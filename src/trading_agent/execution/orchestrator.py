from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from trading_agent.data.moomoo_market import US_OPTION_LOT_SIZE, parse_us_option_code
from trading_agent.data.sec_edgar import SecEdgarClient
from trading_agent.domain.calendar import market_date
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.positions import MonitoredPosition
from trading_agent.execution.orders import OrderManager
from trading_agent.domain.proposals import OpenPositionProposal
from trading_agent.domain.risk import Mandate, PortfolioState, QuoteSnapshot, RiskGate
from trading_agent.research.catalysts import build_candidate_context, derive_score_inputs
from trading_agent.research.committee import Committee
from trading_agent.research.scoring import score_candidate
from trading_agent.storage.audit import AuditWriter
from trading_agent.storage.positions import PositionStore


@dataclass
class CycleResult:
    halted: bool = False
    circuit_breaker: str | None = None
    cooldown: bool = False
    reconciled_closed: list[str] = field(default_factory=list)
    exits: list[dict[str, Any]] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


class PaperTradingCycle:
    """One deterministic paper-trading cycle.

    Reconciles broker state, runs exits, then routes every entry through the
    deterministic risk gate. The gate is the only path to an order, so the LLM
    can never bypass the mandate. Each ticker and position is handled at most
    once per cycle, so there is no unbounded retry loop.
    """

    def __init__(
        self,
        *,
        mandate: Mandate,
        account_id: int,
        market: Any,
        broker: Any,
        sec_client: SecEdgarClient,
        committee: Committee,
        gate: RiskGate,
        liquidity: LiquidityValidator,
        position_store: PositionStore,
        audit: AuditWriter,
        now_fn: Callable[[], datetime] | None = None,
        trd_env: str = "SIMULATE",
        max_open_positions_override: int | None = None,
        order_poll_interval_seconds: float = 2.0,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        if trd_env not in {"SIMULATE", "REAL"}:
            raise ValueError("trd_env must be 'SIMULATE' or 'REAL'")
        self.mandate = mandate
        self.account_id = account_id
        self.market = market
        self.broker = broker
        self.sec_client = sec_client
        self.committee = committee
        self.gate = gate
        self.liquidity = liquidity
        self.position_store = position_store
        self.audit = audit
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.trd_env = trd_env
        self.max_open_positions_override = max_open_positions_override
        self.orders = OrderManager(
            broker=broker,
            account_id=account_id,
            trd_env=trd_env,
            cancel_after_seconds=mandate.execution.cancel_unfilled_order_seconds,
            poll_interval_seconds=order_poll_interval_seconds,
            sleep_fn=sleep_fn,
        )

    def _max_positions(self) -> int:
        mandate_max = self.mandate.portfolio.max_open_positions
        if self.max_open_positions_override is None:
            return mandate_max
        return min(self.max_open_positions_override, mandate_max)

    def run_once(self, tickers: list[str]) -> CycleResult:
        now = self.now_fn()
        result = CycleResult()

        if self.gate.halt_path.exists():
            result.halted = True
            self.audit.append("cycle_halted", {"at": now.isoformat()})
            return result

        held_codes = self._reconcile(now, result)
        marks = self._run_exits(now, held_codes, result)

        if not self._enforce_circuit_breakers(now, result, marks):
            if self._in_cooldown(now):
                result.cooldown = True
                self.audit.append("cooldown_active", {"at": now.isoformat()})
            else:
                self._run_entries(now, tickers, held_codes, result)

        self.audit.append(
            "cycle_completed",
            {
                "at": now.isoformat(),
                "trd_env": self.trd_env,
                "circuit_breaker": result.circuit_breaker,
                "cooldown": result.cooldown,
                "reconciled_closed": result.reconciled_closed,
                "exits": len(result.exits),
                "entries": len(result.entries),
                "rejected": len(result.rejected),
                "errors": len(result.errors),
            },
        )
        return result

    def _enforce_circuit_breakers(
        self, now: datetime, result: CycleResult, marks: dict[str, float]
    ) -> bool:
        """Trip the kill switch on a breached daily-loss or drawdown stop.

        Includes unrealized P/L on still-open positions (marked from the latest
        quote), so a deep open drawdown halts the session before it is closed.
        Unlike the entry-time gate checks, this writes the HALT file so the whole
        session stops until an operator reviews and resumes.
        """

        daily_realized, realized_drawdown, _ = self._realized_pnl(now)
        unrealized = self._unrealized_pnl(marks)
        daily_total = round(daily_realized + unrealized, 4)
        drawdown = round(max(realized_drawdown, realized_drawdown - unrealized), 4)

        reason: str | None = None
        if daily_total <= -self.mandate.portfolio.daily_loss_stop_usd:
            reason = "daily loss stop"
        elif drawdown >= self.mandate.portfolio.hard_drawdown_stop_usd:
            reason = "hard drawdown stop"
        if reason is None:
            return False

        self.gate.halt_path.parent.mkdir(parents=True, exist_ok=True)
        self.gate.halt_path.write_text(f"circuit breaker: {reason}\n", encoding="utf-8")
        result.circuit_breaker = reason
        self.audit.append(
            "circuit_breaker_tripped",
            {
                "reason": reason,
                "daily_pnl_usd": daily_total,
                "unrealized_pnl_usd": unrealized,
                "drawdown_usd": drawdown,
            },
        )
        return True

    def _unrealized_pnl(self, marks: dict[str, float]) -> float:
        """Mark-to-market P/L of still-open ledger positions."""

        total = 0.0
        for row in self.position_store.open_positions():
            mark = marks.get(row["option_code"])
            if mark is None:
                continue
            total += (mark - row["entry_price"]) * row["contracts"] * row["lot_size"]
        return round(total, 4)

    def _in_cooldown(self, now: datetime) -> bool:
        """Block entries for a cooldown window after the consecutive-loss stop."""

        _, _, consecutive_losses = self._realized_pnl(now)
        if consecutive_losses < self.mandate.portfolio.consecutive_loss_stop:
            return False
        last_loss = self._last_loss_time()
        if last_loss is None:
            return False
        cooldown = timedelta(
            hours=self.mandate.portfolio.cooldown_after_consecutive_losses_hours
        )
        return now < last_loss + cooldown

    def _last_loss_time(self) -> datetime | None:
        latest: datetime | None = None
        for row in self.position_store.all_positions():
            if row["status"] != "closed" or row["exit_price"] is None:
                continue
            pnl = (row["exit_price"] - row["entry_price"]) * row["contracts"] * row["lot_size"]
            if pnl < 0 and row.get("closed_at"):
                closed_at = datetime.fromisoformat(str(row["closed_at"]).replace("Z", "+00:00"))
                if latest is None or closed_at > latest:
                    latest = closed_at
        return latest

    # --- reconciliation -------------------------------------------------
    def _reconcile(self, now: datetime, result: CycleResult) -> set[str]:
        broker_positions = self.broker.positions_query(self.account_id, self.trd_env)
        held_codes = {
            row["code"]
            for row in broker_positions
            if row.get("code") and float(row.get("qty") or 0) > 0
        }
        for ledger in self.position_store.open_positions():
            if ledger["option_code"] not in held_codes:
                self.position_store.mark_closed(
                    ledger["option_code"],
                    close_reason="reconciled: not held at broker",
                    exit_price=None,
                )
                result.reconciled_closed.append(ledger["option_code"])
                self.audit.append(
                    "position_reconciled_closed",
                    {"option_code": ledger["option_code"], "at": now.isoformat()},
                )
        return held_codes

    # --- exits ----------------------------------------------------------
    def _run_exits(
        self, now: datetime, held_codes: set[str], result: CycleResult
    ) -> dict[str, float]:
        from trading_agent.execution.monitor import PositionMonitor

        monitor = PositionMonitor(
            self.mandate.options.force_close_before_expiry_trading_days,
            self.mandate.execution.stale_quote_seconds,
        )
        monitored: list[MonitoredPosition] = []
        ledger_by_code: dict[str, dict[str, Any]] = {}
        marks: dict[str, float] = {}
        for ledger in self.position_store.open_positions():
            code = ledger["option_code"]
            if code not in held_codes:
                continue
            try:
                _, expiry, _, _ = parse_us_option_code(code)
                quote = self.market.option_quote(
                    option_code=code, expiry=expiry, lot_size=ledger["lot_size"], now=now
                )
            except Exception as exc:  # noqa: BLE001
                self._record_error(result, "exit_quote", code, exc)
                continue
            ledger_by_code[code] = ledger
            position = MonitoredPosition(
                option_code=code,
                option_side=ledger["option_side"],
                entry_price=ledger["entry_price"],
                contracts=ledger["contracts"],
                lot_size=ledger["lot_size"],
                expiry=date.fromisoformat(ledger["expiry"]),
                take_profit_pct=ledger["take_profit_pct"],
                stop_loss_pct=ledger["stop_loss_pct"],
                time_stop=date.fromisoformat(ledger["time_stop"]),
                bid=quote.bid,
                ask=quote.ask,
                observed_at=quote.observed_at,
            )
            monitored.append(position)
            marks[code] = position.mark_price()

        for signal in monitor.evaluate(monitored, now):
            ledger = ledger_by_code[signal.option_code]
            limit = signal.mark_price if signal.mark_price > 0 else ledger["entry_price"]
            try:
                fill = self.orders.place_and_await(
                    option_code=signal.option_code,
                    contracts=ledger["contracts"],
                    limit_price=limit,
                    side="sell",
                )
            except Exception as exc:  # noqa: BLE001
                self._record_error(result, "exit_order", signal.option_code, exc)
                continue
            self.audit.append("order_placed", {"option_code": signal.option_code, "side": "sell"})
            if not fill.filled:
                # Leave the position open; it will be retried next cycle.
                self.audit.append(
                    "order_cancelled",
                    {"option_code": signal.option_code, "side": "sell", "status": fill.status},
                )
                continue
            exit_price = fill.dealt_avg_price
            realized = round(
                (exit_price - ledger["entry_price"]) * ledger["contracts"] * ledger["lot_size"],
                4,
            )
            self.position_store.mark_closed(
                signal.option_code, close_reason=signal.reason, exit_price=exit_price
            )
            del marks[signal.option_code]
            self.audit.append(
                "order_filled",
                {"option_code": signal.option_code, "side": "sell", "price": exit_price},
            )
            self.audit.append(
                "position_closed",
                {
                    "option_code": signal.option_code,
                    "reason": signal.reason,
                    "realized_pnl_usd": realized,
                    "exit_price": exit_price,
                },
            )
            result.exits.append({"option_code": signal.option_code, "reason": signal.reason})
        return marks

    # --- entries --------------------------------------------------------
    def _run_entries(
        self, now: datetime, tickers: list[str], held_codes: set[str], result: CycleResult
    ) -> None:
        for ticker in tickers:
            if len(held_codes) >= self._max_positions():
                break
            try:
                opened_code = self._try_enter(now, ticker, held_codes, result)
            except Exception as exc:  # noqa: BLE001
                self._record_error(result, "entry", ticker, exc)
                continue
            if opened_code is not None:
                held_codes.add(opened_code)

    def _try_enter(
        self, now: datetime, ticker: str, held_codes: set[str], result: CycleResult
    ) -> str | None:
        evidence = self.sec_client.fetch_evidence(ticker)
        context = build_candidate_context(ticker, evidence, now)
        scores = score_candidate(derive_score_inputs(context.evidence, now))
        output = self.committee.run(context, scores)
        self.audit.append(
            "committee_run",
            {"ticker": ticker, "decision": output.decision, "output": output.model_dump(mode="json")},
        )
        if output.decision != "open_position" or output.proposal is None:
            return None

        proposal = output.proposal
        _, expiry, _, _ = parse_us_option_code(proposal.option_code)

        # Guard against a hallucinated contract: it must be a real listed option.
        if not self.market.is_listed_option(proposal.option_code):
            self.audit.append(
                "proposal_rejected",
                {"option_code": proposal.option_code, "stage": "not_listed"},
            )
            result.rejected.append(
                {"ticker": ticker, "stage": "not_listed", "reasons": ["option not listed"]}
            )
            return None

        quote = self.market.option_quote(
            option_code=proposal.option_code,
            expiry=expiry,
            lot_size=US_OPTION_LOT_SIZE,
            now=now,
        )
        liquidity_result = self.liquidity.validate(quote, proposal.contracts, now)
        self.audit.append(
            "candidate_validated",
            {"ticker": ticker, "option_code": proposal.option_code,
             "result": liquidity_result.model_dump(mode="json")},
        )
        if not liquidity_result.passed:
            result.rejected.append(
                {"ticker": ticker, "stage": "liquidity", "reasons": liquidity_result.reasons}
            )
            return None

        portfolio = self._portfolio_state(now, held_codes, proposal)
        decision = self.gate.evaluate_open(proposal, quote, portfolio, now)
        self.audit.append(
            "proposal_checked",
            {"option_code": proposal.option_code, "decision": decision.model_dump(mode="json")},
        )
        if not decision.approved:
            result.rejected.append(
                {"ticker": ticker, "stage": "risk_gate", "reasons": decision.reasons}
            )
            return None

        fill = self.orders.place_and_await(
            option_code=proposal.option_code,
            contracts=proposal.contracts,
            limit_price=proposal.limit_price,
            side="buy",
        )
        self.audit.append("order_placed", {"option_code": proposal.option_code, "side": "buy"})
        if not fill.filled:
            self.audit.append(
                "order_cancelled",
                {"option_code": proposal.option_code, "side": "buy", "status": fill.status},
            )
            result.rejected.append(
                {"ticker": ticker, "stage": "unfilled", "reasons": [fill.status]}
            )
            return None

        entry_price = fill.dealt_avg_price
        self.position_store.open_position(
            option_code=proposal.option_code,
            ticker=ticker.upper(),
            option_side=proposal.option_side,
            entry_price=entry_price,
            contracts=proposal.contracts,
            lot_size=quote.lot_size,
            expiry=expiry,
            take_profit_pct=proposal.exit_plan.take_profit_pct,
            stop_loss_pct=proposal.exit_plan.stop_loss_pct,
            time_stop=proposal.exit_plan.time_stop,
        )
        self.audit.append(
            "order_filled",
            {"option_code": proposal.option_code, "side": "buy", "price": entry_price},
        )
        result.entries.append({"ticker": ticker, "option_code": proposal.option_code})
        return proposal.option_code

    # --- helpers --------------------------------------------------------
    def _realized_pnl(self, now: datetime) -> tuple[float, float, int]:
        """Return (daily realized P/L, drawdown, consecutive losses) from the ledger."""

        today = market_date(now)
        daily_pnl = 0.0
        total_realized = 0.0
        consecutive_losses = 0
        closed = [
            r
            for r in self.position_store.all_positions()
            if r["status"] == "closed" and r["exit_price"] is not None
        ]
        for row in closed:
            pnl = (row["exit_price"] - row["entry_price"]) * row["contracts"] * row["lot_size"]
            total_realized += pnl
            if row.get("closed_at") and self._date_of(row["closed_at"]) == today:
                daily_pnl += pnl
            consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
        return round(daily_pnl, 4), round(max(0.0, -total_realized), 4), consecutive_losses

    def _portfolio_state(
        self, now: datetime, held_codes: set[str], proposal: OpenPositionProposal
    ) -> PortfolioState:
        ledger_all = self.position_store.all_positions()
        today = market_date(now)
        premium = sum(
            r["entry_price"] * r["contracts"] * r["lot_size"]
            for r in ledger_all
            if r["status"] == "open" and r["option_code"] in held_codes
        )
        new_today = sum(1 for r in ledger_all if self._date_of(r["opened_at"]) == today)
        daily_pnl, drawdown, consecutive_losses = self._realized_pnl(now)

        return PortfolioState(
            open_positions=len(held_codes),
            total_premium_at_risk_usd=round(premium, 4),
            new_positions_today=new_today,
            daily_pnl_usd=daily_pnl,
            total_drawdown_usd=drawdown,
            consecutive_losses=consecutive_losses,
            duplicate_order_exists=proposal.option_code in held_codes,
        )

    @staticmethod
    def _date_of(iso: str) -> date:
        return market_date(datetime.fromisoformat(str(iso).replace("Z", "+00:00")))

    def _record_error(
        self, result: CycleResult, stage: str, ref: str, exc: Exception
    ) -> None:
        detail = {"stage": stage, "ref": ref, "error": str(exc)}
        result.errors.append(detail)
        self.audit.append("cycle_step_failed", detail)
