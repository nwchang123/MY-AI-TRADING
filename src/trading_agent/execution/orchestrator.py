from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from trading_agent.data.iv_history import IV30History
from trading_agent.data.moomoo_market import US_OPTION_LOT_SIZE, parse_us_option_code
from trading_agent.data.sec_edgar import SecEdgarClient
from trading_agent.domain.calendar import (
    market_date,
    minutes_since_open,
    parse_iso,
    subtract_trading_days,
)
from trading_agent.domain.montecarlo import (
    MAX_USABLE_IV,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_theta,
    black_scholes_vega,
    iv_rank,
    kelly_criterion,
    stable_seed,
    win_probability,
)
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.positions import MonitoredPosition
from trading_agent.execution.orders import classify_order_status
from trading_agent.execution.orders import OrderManager
from trading_agent.domain.proposals import OpenPositionProposal, OptionCandidate
from trading_agent.domain.risk import Mandate, PortfolioState, QuoteSnapshot, RiskGate
from trading_agent.research.catalysts import build_candidate_context, derive_score_inputs
from trading_agent.research.committee import Committee, CommitteeOutput
from trading_agent.research.llm import usage_delta
from trading_agent.research.redflags import detect_red_flags, nondirectional_critical_flags
from trading_agent.storage.budget import DailyTokenBudget
from trading_agent.storage.decisions import DecisionCache, decision_digest
from trading_agent.research.scoring import score_candidate
from trading_agent.storage.audit import AuditWriter
from trading_agent.storage.positions import PositionStore

# Upper bound on contracts shown to the committee per ticker, so the prompt
# stays small. Split per side so a directional thesis always has contracts to
# express: a bearish name with only calls in the list would force a hold even
# though the put side is exactly what the catalyst calls for.
_MAX_CANDIDATES_PER_SIDE = 10


def _parse_window_end(window: str | None) -> date | None:
    """End date of a committee 'YYYY-MM-DD/YYYY-MM-DD' catalyst window.

    Tolerates a single date (no slash) and any malformed string (returns None),
    so a model formatting slip degrades to "no catalyst exit" rather than
    raising during entry.
    """

    if not window:
        return None
    candidate = window.split("/")[-1].strip()
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


@dataclass(frozen=True)
class _ExitQuote:
    bid: float
    ask: float
    observed_at: datetime
    is_delayed: bool


@dataclass
class CycleResult:
    halted: bool = False
    circuit_breaker: str | None = None
    cooldown: bool = False
    reconciled_closed: list[str] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)
    open_orders_cancelled: list[str] = field(default_factory=list)
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
        news_client: Any | None = None,
        decision_cache: DecisionCache | None = None,
        llm_budget: DailyTokenBudget | None = None,
        moomoo_market: Any | None = None,
        earnings_client: Any | None = None,
        price_history: Any | None = None,
        earnings_calendar: Any | None = None,
        iv_history: IV30History | None = None,
        on_position_closed: Callable[[str, str, float | None], None] | None = None,
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
        self.news_client = news_client
        self.decision_cache = decision_cache
        self.llm_budget = llm_budget
        self.moomoo_market = moomoo_market
        self.earnings_client = earnings_client
        self.price_history = price_history
        # Unified upcoming-earnings source so the option DTE filter and the
        # pre-earnings exit use the SAME calendar the universe scanner selected
        # on (avoids a Finnhub-vs-yfinance date mismatch). Falls back to
        # earnings_client.next_earnings_date when not supplied.
        self.earnings_calendar = earnings_calendar
        self.iv_history = iv_history
        self._earnings_date_cache: dict[str, date | None] = {}
        # Refreshed at the start of every cycle: when the account compounds,
        # these carry the equity-scaled risk caps; otherwise they alias the
        # injected gate/liquidity unchanged.
        self.active_mandate = mandate
        self.on_position_closed = on_position_closed
        self.active_gate = gate
        self.active_liquidity = liquidity
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
        return self.run_once_lazy(lambda: tickers)

    def run_once_lazy(self, tickers_fn: Callable[[], list[str]]) -> CycleResult:
        try:
            return self._run_once_lazy(tickers_fn)
        finally:
            self._close_adapters()

    def _run_once_lazy(self, tickers_fn: Callable[[], list[str]]) -> CycleResult:
        now = self.now_fn()
        result = CycleResult()

        if self.gate.halt_path.exists():
            result.halted = True
            self.audit.append("cycle_halted", {"at": now.isoformat()})
            return result

        self._refresh_sizing()
        held_codes = self._reconcile(now, result)
        open_orders_ok = self._reconcile_open_orders(now, result)
        marks = self._run_exits(now, held_codes, result)

        # Refresh held_codes from the position store to get an accurate count
        # after exits. Positions closed during _run_exits are now marked as
        # closed in the store, so re-reading gives the correct open set.
        held_codes = {
            r["option_code"]
            for r in self.position_store.open_positions()
        } & held_codes

        if not self._enforce_circuit_breakers(now, result, marks):
            if self._in_cooldown(now):
                result.cooldown = True
                self.audit.append("cooldown_active", {"at": now.isoformat()})
            elif not open_orders_ok:
                result.rejected.append(
                    {
                        "ticker": "*",
                        "stage": "open_order_reconcile",
                        "reasons": ["open orders could not be reconciled"],
                    }
                )
            else:
                capacity_block = self._entry_capacity_block(now, held_codes)
                if capacity_block is not None:
                    self.audit.append(
                        "entries_skipped",
                        {"reason": capacity_block, "at": now.isoformat()},
                    )
                else:
                    self._run_entries(now, tickers_fn(), held_codes, result)

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

    def _close_adapters(self) -> None:
        seen: set[int] = set()
        for adapter in (self.market, self.moomoo_market, self.broker):
            if adapter is None or id(adapter) in seen:
                continue
            seen.add(id(adapter))
            close = getattr(adapter, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                self.audit.append(
                    "adapter_close_failed",
                    {"adapter": type(adapter).__name__, "error": str(exc)},
                )

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
        if daily_total <= -self.active_mandate.portfolio.daily_loss_stop_usd:
            reason = "daily loss stop"
        elif drawdown >= self.active_mandate.portfolio.hard_drawdown_stop_usd:
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
                closed_at = parse_iso(row["closed_at"])
                if latest is None or closed_at > latest:
                    latest = closed_at
        return latest

    # --- reconciliation -------------------------------------------------
    def _reconcile(self, now: datetime, result: CycleResult) -> set[str]:
        broker_positions = self.broker.positions_query(self.account_id, self.trd_env)
        held_rows = [
            row
            for row in broker_positions
            if row.get("code") and float(row.get("qty") or 0) > 0
        ]
        held_codes = {row["code"] for row in held_rows}
        for ledger in self.position_store.open_positions():
            if ledger["option_code"] not in held_codes:
                self.position_store.mark_closed(
                    ledger["option_code"],
                    close_reason="reconciled: not held at broker",
                    exit_price=None,
                )
                if self.on_position_closed is not None:
                    self.on_position_closed(
                        ledger["option_code"], "reconciled: not held at broker", None
                    )
                result.reconciled_closed.append(ledger["option_code"])
                self.audit.append(
                    "position_reconciled_closed",
                    {"option_code": ledger["option_code"], "at": now.isoformat()},
                )
        self._adopt_orphans(now, held_rows, result)
        return held_codes

    def _reconcile_open_orders(self, now: datetime, result: CycleResult) -> bool:
        """Cancel broker orders left behind by a crashed previous cycle.

        The broker is authoritative for submitted-but-not-final orders. If a
        process dies after broker acceptance but before local ledger write, the
        next cycle first cancels any still-pending orders, then position
        reconciliation/adoption handles anything that already filled.
        """

        method = getattr(self.broker, "open_orders_query", None)
        if method is None:
            return True
        try:
            rows = method(self.account_id, self.trd_env)
        except Exception as exc:  # noqa: BLE001
            self._record_error(result, "open_orders_query", "*", exc)
            return False

        for row in rows:
            status = str(row.get("order_status") or row.get("status") or "")
            classification = classify_order_status(status)
            if classification in {"filled", "dead"}:
                continue
            order_id = row.get("order_id")
            if order_id is None:
                self.audit.append(
                    "open_order_reconcile_skipped",
                    {"reason": "missing_order_id", "order": row, "at": now.isoformat()},
                )
                continue
            order_id = str(order_id)
            try:
                self.broker.cancel_order(self.account_id, order_id, self.trd_env)
            except Exception as exc:  # noqa: BLE001
                self._record_error(result, "open_order_cancel", order_id, exc)
                return False
            result.open_orders_cancelled.append(order_id)
            self.audit.append(
                "open_order_cancelled",
                {
                    "order_id": order_id,
                    "option_code": row.get("code") or row.get("option_code"),
                    "status": status,
                    "at": now.isoformat(),
                },
            )
        return True

    def _adopt_orphans(
        self, now: datetime, held_rows: list[dict[str, Any]], result: CycleResult
    ) -> None:
        """Adopt broker option positions missing from the local ledger.

        A crash between order fill and ledger write leaves a position the exit
        engine would otherwise never manage. Adopted positions get the mandate's
        standard exit plan (+100/-50, time stop at expiry -- the forced-close
        window still fires first). Non-option holdings are ignored.

        Orphans are rejected if adopting would violate the portfolio mandate
        (position count, total premium, single-position cost).
        """

        ledger_open = {r["option_code"] for r in self.position_store.open_positions()}
        current_premium = sum(
            r["entry_price"] * r["contracts"] * r["lot_size"]
            for r in self.position_store.open_positions()
        )
        current_count = len(ledger_open)
        for row in held_rows:
            code = row["code"]
            if code in ledger_open:
                continue
            try:
                _, expiry, side, _ = parse_us_option_code(code)
            except ValueError:
                continue  # stock or non-US-option holding: not ours to manage
            entry = float(row.get("cost_price") or 0.0)
            if entry <= 0:
                entry = float(row.get("nominal_price") or 0.0)
            if entry <= 0:
                continue
            contracts = int(float(row.get("qty") or 1))
            orphan_cost = entry * contracts * US_OPTION_LOT_SIZE

            # --- risk gate checks for orphan adoption ---
            portfolio = self.active_mandate.portfolio
            max_positions = self._max_positions()
            reject_reason = None
            if current_count + 1 > max_positions:
                reject_reason = f"adopting {code} would exceed max open positions ({current_count}+1 > {max_positions})"
            elif portfolio.max_single_position_cost_usd > 0 and orphan_cost > portfolio.max_single_position_cost_usd:
                reject_reason = f"adopting {code} cost ${orphan_cost:.0f} exceeds single position limit ${portfolio.max_single_position_cost_usd:.0f}"
            elif current_premium + orphan_cost > portfolio.max_total_premium_at_risk_usd:
                reject_reason = f"adopting {code} would exceed total premium at risk (${current_premium:.0f}+${orphan_cost:.0f} > ${portfolio.max_total_premium_at_risk_usd:.0f})"

            if reject_reason:
                self.audit.append(
                    "orphan_adopt_rejected",
                    {"option_code": code, "reason": reject_reason, "at": now.isoformat()},
                )
                continue

            self.position_store.open_position(
                option_code=code,
                ticker=parse_us_option_code(code)[0],
                option_side=side,
                entry_price=entry,
                contracts=contracts,
                lot_size=US_OPTION_LOT_SIZE,
                expiry=expiry,
                take_profit_pct=100.0,
                stop_loss_pct=50.0,
                time_stop=expiry,
            )
            current_count += 1
            current_premium += orphan_cost
            result.adopted.append(code)
            self.audit.append(
                "position_adopted",
                {"option_code": code, "entry_price": entry, "at": now.isoformat()},
            )

    # --- exits ----------------------------------------------------------
    def _run_exits(
        self, now: datetime, held_codes: set[str], result: CycleResult
    ) -> dict[str, float]:
        from trading_agent.execution.monitor import PositionMonitor

        monitor = PositionMonitor(
            self.active_mandate.options.force_close_before_expiry_trading_days,
            self.active_mandate.execution.stale_quote_seconds,
            self.active_mandate.execution.delayed_quote_max_age_seconds,
            self.active_mandate.options.iv_crush_exit_drop_pct,
            self.active_mandate.options.max_theta_decay_pct_per_day,
            self.active_mandate.options.trailing_profit_activation_pct,
            self.active_mandate.options.trailing_profit_giveback_pct,
        )
        monitored: list[MonitoredPosition] = []
        ledger_by_code: dict[str, dict[str, Any]] = {}
        marks: dict[str, float] = {}
        for ledger in self.position_store.open_positions():
            code = ledger["option_code"]
            if code not in held_codes:
                continue
            try:
                ticker, expiry, side, strike = parse_us_option_code(code)
                quote = self._exit_quote(
                    ticker=ticker,
                    option_code=code,
                    expiry=expiry,
                    lot_size=ledger["lot_size"],
                    now=now,
                )
            except Exception as exc:  # noqa: BLE001
                self._record_error(result, "exit_quote", code, exc)
                continue
            previous_peak = float(ledger.get("peak_bid") or 0.0)
            peak_bid = max(previous_peak, float(quote.bid or 0.0))
            if peak_bid > previous_peak:
                self.position_store.update_peak_bid(code, peak_bid)
            ledger_by_code[code] = ledger
            window_end = ledger.get("catalyst_window_end")
            pre_earnings = ledger.get("pre_earnings_exit_date")
            current_iv = self._current_option_iv(ticker, code, expiry, now)
            theta_iv = current_iv or ledger.get("entry_iv")
            theta_decay_pct = self._theta_decay_pct_per_day(
                ticker=ticker,
                side=side,
                strike=strike,
                expiry=expiry,
                iv=float(theta_iv or 0.0),
                mark=quote.bid,
                now=now,
            )
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
                catalyst_window_end=date.fromisoformat(window_end) if window_end else None,
                pre_earnings_exit_date=(
                    date.fromisoformat(pre_earnings) if pre_earnings else None
                ),
                entry_iv=ledger.get("entry_iv"),
                current_iv=current_iv,
                theta_decay_pct_per_day=theta_decay_pct,
                peak_bid=peak_bid if peak_bid > 0 else None,
                bid=quote.bid,
                ask=quote.ask,
                observed_at=quote.observed_at,
                is_delayed=quote.is_delayed,
            )
            monitored.append(position)
            marks[code] = position.mark_price()

        position_by_code = {p.option_code: p for p in monitored}
        for signal in monitor.evaluate(monitored, now):
            ledger = ledger_by_code[signal.option_code]
            limit = signal.mark_price if signal.mark_price > 0 else ledger["entry_price"]
            # Sell ladder: try the mid first, then the bid as a last resort
            # before the limit (stop-loss). The bid rung recovers roughly half
            # the spread on fills that would happen anyway; without it a
            # mid→limit gap can leave an ITM position unhedged if the spread
            # is wide and the signal fires near the bid.
            position = position_by_code.get(signal.option_code)
            ladder = [limit]
            if position is not None and position.bid > 0:
                ladder = self._exit_sell_ladder(
                    position=position,
                    limit=limit,
                    max_chase_pct=self.active_mandate.execution.max_limit_chase_pct,
                )
            if position is not None:
                self.audit.append(
                    "exit_signal",
                    self._exit_signal_payload(
                        position=position,
                        signal=signal,
                        limit=limit,
                        ladder=ladder,
                        now=now,
                        trailing_activation_pct=(
                            self.active_mandate.options.trailing_profit_activation_pct
                        ),
                        trailing_giveback_pct=(
                            self.active_mandate.options.trailing_profit_giveback_pct
                        ),
                    ),
                )
            try:
                fill = self.orders.place_and_await(
                    option_code=signal.option_code,
                    contracts=ledger["contracts"],
                    limit_price=limit,
                    side="sell",
                    price_ladder=ladder,
                    on_order_submitted=lambda payload: self.audit.append(
                        "order_submitted", payload
                    ),
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
                self._net_pnl(
                    ledger["entry_price"], exit_price,
                    ledger["contracts"], ledger["lot_size"],
                ),
                4,
            )
            self.position_store.mark_closed(
                signal.option_code, close_reason=signal.reason, exit_price=exit_price
            )
            if self.on_position_closed is not None:
                self.on_position_closed(signal.option_code, signal.reason, exit_price)
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

    def _exit_quote(
        self,
        *,
        ticker: str,
        option_code: str,
        expiry: date,
        lot_size: int,
        now: datetime,
    ) -> _ExitQuote:
        try:
            quote = self.market.option_quote(
                option_code=option_code, expiry=expiry, lot_size=lot_size, now=now
            )
            return _ExitQuote(
                bid=float(quote.bid or 0.0),
                ask=float(quote.ask or 0.0),
                observed_at=quote.observed_at,
                is_delayed=quote.is_delayed,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._is_missing_ask_error(exc):
                raise
            fallback = self._exit_quote_from_chain(
                ticker=ticker, option_code=option_code, expiry=expiry, now=now
            )
            if fallback is None:
                raise
            self.audit.append(
                "exit_quote_degraded",
                {
                    "option_code": option_code,
                    "reason": str(exc),
                    "bid": round(fallback.bid, 4),
                    "ask": round(fallback.ask, 4),
                    "observed_at": fallback.observed_at.isoformat(),
                    "is_delayed": fallback.is_delayed,
                },
            )
            return fallback

    @staticmethod
    def _is_missing_ask_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "no ask" in message or (
            "ask" in message and "greater than 0" in message
        )

    def _exit_quote_from_chain(
        self,
        *,
        ticker: str,
        option_code: str,
        expiry: date,
        now: datetime,
    ) -> _ExitQuote | None:
        chain = self.market.option_chain(ticker, expiry, expiry, "ALL")
        row = next((r for r in chain if r.get("code") == option_code), None)
        if row is None:
            return None
        bid = float(row.get("bid") or row.get("bid_price") or 0.0)
        ask = float(row.get("ask") or row.get("ask_price") or 0.0)
        if bid <= 0:
            return None
        observed = row.get("observed_at")
        observed_at = parse_iso(observed) if observed else now
        return _ExitQuote(
            bid=bid,
            ask=max(ask, 0.0),
            observed_at=observed_at,
            is_delayed=bool(row.get("is_delayed", True)),
        )

    @staticmethod
    def _exit_sell_ladder(
        *,
        position: MonitoredPosition,
        limit: float,
        max_chase_pct: float,
    ) -> list[float]:
        """Sell-to-close ladder from patient to marketable.

        Exit quotes can be delayed. A sell order parked exactly at the delayed
        bid can miss a fast-moving live market, so the final rung is allowed to
        concede up to ``max_chase_pct`` below the observed bid. This preserves
        the existing mid/bid attempts while making a triggered exit much more
        likely to actually leave the book.
        """

        bid = round(float(position.bid), 2)
        ask = round(float(position.ask), 2)
        base_limit = round(float(limit), 2)
        if bid <= 0:
            return [base_limit]

        chase_pct = max(float(max_chase_pct or 0.0), 0.0)
        aggressive = round(bid * (1.0 - chase_pct / 100.0), 2)
        if aggressive <= 0 and bid > 0:
            aggressive = 0.01

        ladder: list[float] = []
        prices = (bid, base_limit, aggressive)
        if ask > bid:
            prices = (round((bid + ask) / 2, 2), bid, base_limit, aggressive)
        for price in prices:
            if price > 0 and all(abs(price - seen) >= 0.005 for seen in ladder):
                ladder.append(price)
        return ladder

    @staticmethod
    def _exit_signal_payload(
        *,
        position: MonitoredPosition,
        signal: Any,
        limit: float,
        ladder: list[float],
        now: datetime,
        trailing_activation_pct: float = 0.0,
        trailing_giveback_pct: float = 0.0,
    ) -> dict[str, Any]:
        entry = float(position.entry_price)
        take_profit_price = entry * (1 + position.take_profit_pct / 100.0)
        stop_loss_price = entry * (1 - position.stop_loss_pct / 100.0)
        payload: dict[str, Any] = {
            "option_code": position.option_code,
            "reason": signal.reason,
            "mark_price": signal.mark_price,
            "pnl_pct": signal.pnl_pct,
            "entry_price": round(entry, 4),
            "contracts": position.contracts,
            "lot_size": position.lot_size,
            "bid": round(float(position.bid), 4),
            "ask": round(float(position.ask), 4),
            "limit_price": round(float(limit), 4),
            "price_ladder": [round(float(price), 4) for price in ladder],
            "take_profit_pct": position.take_profit_pct,
            "take_profit_price": round(take_profit_price, 4),
            "stop_loss_pct": position.stop_loss_pct,
            "stop_loss_price": round(max(stop_loss_price, 0.0), 4),
            "observed_at": position.observed_at.isoformat(),
            "quote_age_seconds": round((now - position.observed_at).total_seconds(), 3),
            "is_delayed": position.is_delayed,
        }
        if position.peak_bid is not None:
            peak_bid = float(position.peak_bid)
            payload["peak_bid"] = round(peak_bid, 4)
            if peak_bid > entry:
                peak_profit = peak_bid - entry
                payload["peak_profit_pct"] = round(peak_profit / entry * 100.0, 4)
                if trailing_activation_pct > 0 and trailing_giveback_pct > 0:
                    payload["trailing_profit_activation_pct"] = trailing_activation_pct
                    payload["trailing_profit_giveback_pct"] = trailing_giveback_pct
                    payload["trailing_activation_price"] = round(
                        entry * (1 + trailing_activation_pct / 100.0), 4
                    )
                    payload["trailing_trigger_price"] = round(
                        entry
                        + peak_profit * (1 - trailing_giveback_pct / 100.0),
                        4,
                    )
        return payload

    def _current_option_iv(
        self, ticker: str, option_code: str, expiry: date, now: datetime
    ) -> float | None:
        try:
            chain = self.market.option_chain(ticker, now.date(), expiry, "ALL")
        except Exception:  # noqa: BLE001 - exit Greeks are best-effort
            return None
        for row in chain:
            if row.get("code") != option_code:
                continue
            iv = float(row.get("iv") or 0.0)
            return iv if iv > 0 else None
        return None

    def _theta_decay_pct_per_day(
        self,
        *,
        ticker: str,
        side: str,
        strike: float,
        expiry: date,
        iv: float,
        mark: float,
        now: datetime,
    ) -> float | None:
        if iv <= 0 or mark <= 0:
            return None
        spot = 0.0
        method = getattr(self.market, "underlying_snapshot", None)
        if method is not None:
            try:
                spot = float((method(ticker) or {}).get("price") or 0.0)
            except Exception:  # noqa: BLE001 - exit Greeks are best-effort
                spot = 0.0
        if spot <= 0:
            return None
        dte = max((expiry - now.date()).days, 1)
        theta = black_scholes_theta(
            side=side,
            spot=spot,
            strike=strike,
            t_years=dte / 365.0,
            iv=iv,
        ) / 365.0
        if theta >= 0:
            return 0.0
        return round(abs(theta) / mark * 100.0, 3)

    # --- entries --------------------------------------------------------
    def _vix(self, now: datetime) -> float | None:
        """Current VIX level, or None when the feed has no index snapshot.

        Used both by the cycle-wide entry block and as the value handed to the
        deterministic risk gate (so the mandate's VIX cap is enforced inside the
        gate too, not only here). A None return is audited as ``vix_unavailable``.
        """

        method = getattr(self.market, "underlying_snapshot", None)
        if method is None:
            return None
        try:
            vix = float((method("_VIX") or {}).get("price") or 0.0)
        except Exception:  # noqa: BLE001 - missing data never blocks
            vix = 0.0
        if vix <= 0:
            # Fail loud: a panic-regime guard that silently can't read the VIX is
            # worse than no guard -- it looks active but isn't. Audit it so the
            # gap is visible (the feed may lack an index snapshot).
            self.audit.append("vix_unavailable", {"at": now.isoformat()})
            return None
        return vix

    def _entry_regime_block(self, now: datetime) -> str | None:
        """Cycle-wide reason to skip ALL new entries, or None. Exits still run.

        This is a cheap pre-filter that avoids spending LLM tokens when the whole
        market is in a panic regime or just opened. The same VIX/minutes rules
        are ALSO enforced inside the deterministic risk gate (per-proposal), so a
        caller that bypasses this orchestrator still cannot trade through them.
        """

        window = self.active_mandate.execution.no_entry_minutes_after_open
        if window > 0:
            minutes = minutes_since_open(now)
            if minutes is not None and minutes < window:
                return f"first {window} minutes after the open (widest spreads)"

        vix_cap = self.active_mandate.portfolio.max_vix_for_entries
        if vix_cap > 0:
            vix = self._vix(now)
            if vix is not None and vix > vix_cap:
                return f"VIX {vix:.1f} above the {vix_cap:.0f} entry cap"
        return None

    def _entry_capacity_block(self, now: datetime, held_codes: set[str]) -> str | None:
        """Cheap reason to skip universe selection before spending scan/API work."""

        open_rows = [
            row
            for row in self.position_store.open_positions()
            if row["option_code"] in held_codes
        ]
        max_positions = self._max_positions()
        if len(open_rows) >= max_positions:
            return f"maximum open positions reached ({len(open_rows)}/{max_positions})"

        premium = sum(
            row["entry_price"] * row["contracts"] * row["lot_size"]
            for row in open_rows
        )
        premium_cap = self.active_mandate.portfolio.max_total_premium_at_risk_usd
        if premium_cap > 0 and premium >= premium_cap:
            return (
                f"total premium at risk at limit "
                f"(${premium:.0f}/${premium_cap:.0f})"
            )

        today = market_date(now)
        new_today = sum(
            1
            for opened_at in self.position_store.opened_at_values()
            if self._date_of(opened_at) == today
        )
        max_new = self.active_mandate.portfolio.max_new_positions_per_day
        if new_today >= max_new:
            return f"daily new-position limit reached ({new_today}/{max_new})"
        return None

    def _run_entries(
        self, now: datetime, tickers: list[str], held_codes: set[str], result: CycleResult
    ) -> None:
        block = self._entry_regime_block(now)
        if block is not None:
            self.audit.append(
                "entries_skipped", {"reason": block, "at": now.isoformat()}
            )
            result.rejected.append(
                {"ticker": "*", "stage": "regime", "reasons": [block]}
            )
            return
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

    def _eligible_candidates(
        self,
        now: datetime,
        ticker: str,
        result: CycleResult,
        spot: float = 0.0,
        iv_rank_val: float | None = None,
    ) -> list[OptionCandidate]:
        """Mandate-eligible contracts from the live chain, for the committee.

        Pulls the option chain in the mandate DTE window and keeps only contracts
        that pass the deterministic liquidity validator, so the committee can only
        ever choose a real, tradeable contract instead of guessing one blind. The
        list keeps the most liquid ``_MAX_CANDIDATES_PER_SIDE`` calls AND puts so
        a bearish thesis is never starved of a contract to express it.
        """

        today = now.date()
        start = today + timedelta(days=self.mandate.options.min_dte)
        end = today + timedelta(days=self.mandate.options.max_dte)
        # Pre-earnings (IV-ramp) play: the option must OUTLIVE the print to carry
        # its event vega through the hold, so drop contracts expiring on/before
        # the earnings date. This contract filter belongs to the IV-ramp ENTRY
        # strategy ONLY (gated on earnings_window_max_days), NOT the pre-earnings
        # exit safety: when only the exit is kept, requiring expiry > earnings is
        # both pointless (the position is closed before the print anyway) and
        # catastrophic in earnings season -- every in-window expiry falls on/
        # before the next print, emptying the candidate list for the whole
        # universe.
        earnings_date = None
        if self.active_mandate.options.earnings_window_max_days > 0:
            earnings_date = self._next_earnings_date(ticker)
        try:
            chain = self.market.option_chain(ticker, start, end, "ALL")
        except Exception as exc:  # noqa: BLE001
            self._record_error(result, "option_chain", ticker, exc)
            return []

        candidates: list[OptionCandidate] = []
        for row in chain:
            ask = float(row.get("ask") or 0.0)
            if ask <= 0:
                continue
            if earnings_date is not None and row["expiry"] <= earnings_date:
                continue
            quote = QuoteSnapshot(
                option_code=row["code"],
                bid=float(row.get("bid") or 0.0),
                ask=ask,
                open_interest=int(row.get("open_interest") or 0),
                daily_volume=int(row.get("daily_volume") or 0),
                lot_size=US_OPTION_LOT_SIZE,
                expiry=row["expiry"],
                observed_at=now,
                is_delayed=True,
            )
            verdict = self.active_liquidity.validate(
                quote, self.active_mandate.options.contracts_per_order, now
            )
            if not verdict.passed:
                continue
            iv = float(row.get("iv") or 0.0)
            # Baseline math POP at the standard +100/-50 grid, plus delta and the
            # breakeven move, shown to the committee alongside each candidate so
            # the AIs see what the math says before claiming an edge. Fewer
            # paths: a guide, not the gate.
            mc_pop: float | None = None
            delta: float | None = None
            gamma: float | None = None
            vega: float | None = None
            theta: float | None = None
            theta_decay_pct: float | None = None
            if spot > 0 and 0 < iv <= MAX_USABLE_IV:
                t_years = max(verdict.dte, 1) / 365.0
                mc_pop = round(
                    win_probability(
                        side=row["side"],
                        spot=spot,
                        strike=row["strike"],
                        dte_days=verdict.dte,
                        iv=iv,
                        entry_price=ask,
                        take_profit_pct=100.0,
                        stop_loss_pct=50.0,
                        paths=500,
                        seed=stable_seed(row["code"]),
                    ),
                    3,
                )
                delta = round(
                    black_scholes_delta(
                        side=row["side"],
                        spot=spot,
                        strike=row["strike"],
                        t_years=t_years,
                        iv=iv,
                    ),
                    3,
                )
                gamma = round(
                    black_scholes_gamma(
                        spot=spot,
                        strike=row["strike"],
                        t_years=t_years,
                        iv=iv,
                    ),
                    4,
                )
                vega = round(
                    black_scholes_vega(
                        spot=spot,
                        strike=row["strike"],
                        t_years=t_years,
                        iv=iv,
                    )
                    * 0.01,
                    4,
                )
                theta = round(
                    black_scholes_theta(
                        side=row["side"],
                        spot=spot,
                        strike=row["strike"],
                        t_years=t_years,
                        iv=iv,
                    )
                    / 365.0,
                    4,
                )
                if ask > 0 and theta < 0:
                    theta_decay_pct = round(abs(theta) / ask * 100.0, 3)
            candidates.append(
                OptionCandidate(
                    option_code=row["code"],
                    option_side=row["side"],
                    strike=row["strike"],
                    expiry=row["expiry"],
                    bid=quote.bid,
                    ask=ask,
                    open_interest=quote.open_interest,
                    daily_volume=quote.daily_volume,
                    iv=iv,
                    dte=verdict.dte,
                    estimated_contract_cost_usd=verdict.estimated_contract_cost_usd,
                    mc_pop=mc_pop,
                    delta=delta,
                    gamma=gamma,
                    vega=vega,
                    theta=theta,
                    theta_decay_pct_per_day=theta_decay_pct,
                    iv_rank=iv_rank_val,
                    breakeven_move_pct=self._breakeven_move_pct(
                        row["side"], spot, row["strike"], ask
                    ),
                )
            )
        return self._top_per_side(candidates)

    @staticmethod
    def _breakeven_move_pct(
        side: str, spot: float, strike: float, ask: float
    ) -> float | None:
        """Signed % the underlying must move by expiry to break even (None w/o spot).

        A long call breaks even at strike+premium (needs an up move); a long put
        at strike-premium (needs a down move). The sign tells the committee the
        required direction, the magnitude how far OTM the strike is.
        """

        if spot <= 0:
            return None
        breakeven = strike + ask if side == "call" else strike - ask
        return round((breakeven - spot) / spot * 100.0, 2)

    @staticmethod
    def _top_per_side(candidates: list[OptionCandidate]) -> list[OptionCandidate]:
        """Most-liquid N calls + N puts, each ranked by open interest."""

        by_oi = sorted(candidates, key=lambda c: c.open_interest, reverse=True)
        calls = [c for c in by_oi if c.option_side == "call"][:_MAX_CANDIDATES_PER_SIDE]
        puts = [c for c in by_oi if c.option_side == "put"][:_MAX_CANDIDATES_PER_SIDE]
        return calls + puts

    def _gather_evidence(self, ticker: str, result: CycleResult) -> list[Any]:
        """SEC filings (with 8-K bodies) plus news headlines for one ticker.

        A news-feed failure is recorded but never blocks the entry: filings are
        the primary evidence and headlines are an enrichment. 8-K bodies are
        fetched so the committee reads what the filing SAYS, not just that it
        exists. News is queried by the registered company name when available,
        so single-word tickers (TE, BULL) stop pulling unrelated headlines.
        """

        def fetch_sec() -> list[Any]:
            return list(self.sec_client.fetch_evidence(ticker, fetch_bodies=True))

        def fetch_news() -> list[Any]:
            if self.news_client is None:
                return []
            query_name = None
            try:
                query_name = self.sec_client.company_name(ticker)
            except Exception:  # noqa: BLE001 - name lookup is best-effort
                query_name = None
            return list(self.news_client.fetch_evidence(ticker, query_name=query_name))

        def fetch_earnings() -> list[Any]:
            if self.earnings_client is None:
                return []
            # Upcoming earnings inside the holding window feeds the existing
            # earnings_iv_crush red flag before the print, not after it.
            return list(self.earnings_client.fetch_evidence(ticker))

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                "sec_fetch": pool.submit(fetch_sec),
                "news_fetch": pool.submit(fetch_news),
                "earnings_fetch": pool.submit(fetch_earnings),
            }
            results: dict[str, list[Any]] = {}
            sec_error: Exception | None = None
            for stage, future in futures.items():
                try:
                    results[stage] = future.result()
                except Exception as exc:  # noqa: BLE001
                    if stage == "sec_fetch":
                        sec_error = exc
                    else:
                        self._record_error(result, stage, ticker, exc)
            if sec_error is not None:
                raise sec_error
        evidence: list[Any] = []
        evidence.extend(results.get("sec_fetch", []))
        evidence.extend(results.get("news_fetch", []))
        evidence.extend(results.get("earnings_fetch", []))
        return evidence

    def _underlying_snapshot(
        self, ticker: str, result: CycleResult
    ) -> dict[str, Any] | None:
        """Get real-time stock price from Moomoo, fallback to delayed CBOE/Yahoo.

        When Moomoo provides the price but lacks iv30, we enrich the snapshot
        from the CBOE/Yahoo fallback so IV Rank can be computed.
        """

        def fetch_moomoo() -> dict[str, Any] | None:
            if self.moomoo_market is None:
                return None
            return self.moomoo_market.stock_snapshot(ticker)

        def fetch_delayed() -> dict[str, Any] | None:
            method = getattr(self.market, "underlying_snapshot", None)
            if method is None:
                return None
            return method(ticker) or None

        with ThreadPoolExecutor(max_workers=2) as pool:
            moomoo_future = pool.submit(fetch_moomoo)
            delayed_future = pool.submit(fetch_delayed)
            try:
                moomoo_snap = moomoo_future.result()
            except Exception as exc:  # noqa: BLE001
                moomoo_snap = None
                self._record_error(result, "moomoo_stock_snapshot", ticker, exc)
            try:
                cboe_snap = delayed_future.result()
            except Exception:  # noqa: BLE001
                cboe_snap = None

        # Prefer Moomoo price (real-time), always enrich with CBOE iv30 if missing.
        snapshot = moomoo_snap or cboe_snap
        if snapshot is None:
            return None

        if not snapshot.get("iv30") and cboe_snap:
            snapshot["iv30"] = cboe_snap.get("iv30", 0)
            snapshot["iv30_change"] = cboe_snap.get("iv30_change", 0)

        source = "moomoo_realtime" if moomoo_snap else "cboe_delayed"
        self.audit.append("stock_snapshot_source", {"ticker": ticker, "source": source})
        return snapshot

    def _price_context(self, ticker: str, result: CycleResult) -> dict[str, Any] | None:
        """Delayed daily-bar technical context for the committee (best-effort)."""

        if self.price_history is None:
            return None
        try:
            return self.price_history.context(ticker) or None
        except Exception as exc:  # noqa: BLE001 - context never blocks an entry
            self._record_error(result, "price_context", ticker, exc)
            return None

    def _next_earnings_date(self, ticker: str):
        """Next earnings date for ``ticker`` from the unified calendar (cached).

        Prefers ``earnings_calendar`` (the SAME source the universe scanner
        selected on), falling back to ``earnings_client``. Best-effort: any
        failure or unknown date yields None. Cached per cycle so the candidate
        DTE filter and the pre-earnings exit share one lookup.
        """

        key = ticker.strip().upper()
        if key in self._earnings_date_cache:
            return self._earnings_date_cache[key]
        result = None
        for source in (self.earnings_calendar, self.earnings_client):
            getter = getattr(source, "next_earnings_date", None) if source else None
            if getter is None:
                continue
            try:
                result = getter(ticker)
            except Exception:  # noqa: BLE001 - absence is not an error
                result = None
            if result is not None:
                break
        self._earnings_date_cache[key] = result
        return result

    def _pre_earnings_exit_date(self, ticker: str) -> date | None:
        """Exit date for the pre-earnings (IV-ramp) play, or None when disabled.

        K trading days before the next earnings print so the position is out
        BEFORE the report. None when the strategy is off or no date is known
        (the position then falls back to its ordinary exits).
        """

        k = self.active_mandate.options.pre_earnings_exit_trading_days
        if k <= 0:
            return None
        earnings_day = self._next_earnings_date(ticker)
        if earnings_day is None:
            return None
        return subtract_trading_days(earnings_day, k)

    def _iv_rank_value(self, ticker: str, snapshot: dict | None) -> float | None:
        current_iv30 = float((snapshot or {}).get("iv30") or 0.0)
        if current_iv30 <= 0 or self.iv_history is None:
            return None
        extremes = self.iv_history.extremes(ticker)
        if extremes is None:
            return None
        return iv_rank(
            current_iv=current_iv30,
            iv_52w_high=extremes[0],
            iv_52w_low=extremes[1],
        )

    def _try_enter(
        self, now: datetime, ticker: str, held_codes: set[str], result: CycleResult
    ) -> str | None:
        snapshot = self._underlying_snapshot(ticker, result)

        # Record iv30 for IV Rank history (always, even without candidates).
        if self.iv_history is not None and snapshot:
            current_iv30 = float(snapshot.get("iv30") or 0)
            if current_iv30 > 0:
                self.iv_history.record(ticker, current_iv30, today=market_date(now))

        iv_rank_val = self._iv_rank_value(ticker, snapshot)
        spot = float((snapshot or {}).get("price") or 0.0)
        candidates = self._eligible_candidates(
            now, ticker, result, spot=spot, iv_rank_val=iv_rank_val
        )
        if not candidates:
            self.audit.append(
                "no_eligible_contracts", {"ticker": ticker, "at": now.isoformat()}
            )
            result.rejected.append(
                {
                    "ticker": ticker,
                    "stage": "no_eligible_contracts",
                    "reasons": ["no mandate-eligible contract in the option chain"],
                }
            )
            return None

        price_context = self._price_context(ticker, result)

        # Pre-committee deterministic screen: gather evidence and check red
        # flags BEFORE invoking the expensive LLM committee.  A ticker with
        # critical red flags (dilution, stale catalyst) or weak catalyst score
        # is rejected here at zero LLM cost, saving ~40k tokens per run.
        evidence = self._gather_evidence(ticker, result)
        context = build_candidate_context(ticker, evidence, now)
        scores = score_candidate(derive_score_inputs(context.evidence, now))
        red_flags = detect_red_flags(context, scores)
        blockers = nondirectional_critical_flags(red_flags)
        if blockers:
            self.audit.append(
                "pre_screen_reject",
                {"ticker": ticker, "reasons": [f.code for f in blockers]},
            )
            result.rejected.append(
                {
                    "ticker": ticker,
                    "stage": "pre_screen",
                    "reasons": [
                        f"blocked by deterministic red flag(s): "
                        + ", ".join(f.code for f in blockers)
                    ],
                }
            )
            return None

        # Skip committee when catalyst score is too weak to justify LLM cost.
        if scores.total < 20.0:
            self.audit.append(
                "pre_screen_reject",
                {"ticker": ticker, "reasons": ["weak_catalyst_score"],
                 "catalyst_score": scores.total},
            )
            result.rejected.append(
                {
                    "ticker": ticker,
                    "stage": "pre_screen",
                    "reasons": [
                        f"catalyst score {scores.total:.1f} below threshold 20.0"
                    ],
                }
            )
            return None

        # Skip committee when ALL candidates have IV above the mandate ceiling.
        iv_ceiling = self.active_mandate.options.max_entry_iv
        if iv_ceiling > 0 and candidates:
            all_over = all(c.iv > iv_ceiling for c in candidates if c.iv > 0)
            if all_over:
                self.audit.append(
                    "pre_screen_reject",
                    {"ticker": ticker, "reasons": ["all_candidates_iv_over_ceiling"],
                     "iv_ceiling": iv_ceiling},
                )
                result.rejected.append(
                    {
                        "ticker": ticker,
                        "stage": "pre_screen",
                        "reasons": [
                            f"all candidate IVs exceed ceiling {iv_ceiling:.0%}"
                        ],
                    }
                )
                return None

        output = self._get_or_run_committee(
            now, ticker, candidates, snapshot, price_context, result,
            pre_context=context, pre_scores=scores, iv_rank_val=iv_rank_val,
        )
        if output is None or output.decision != "open_position" or output.proposal is None:
            return None

        candidate, expiry = self._validate_proposal(
            ticker, output, candidates, now, spot, result
        )
        if candidate is None:
            return None

        return self._execute_entry(
            now, ticker, held_codes, output.proposal, candidate, expiry, spot, result
        )

    def _get_or_run_committee(
        self,
        now: datetime,
        ticker: str,
        candidates: list,
        snapshot: dict | None,
        price_context: dict | None,
        result: CycleResult,
        pre_context=None,
        pre_scores=None,
        iv_rank_val: float | None = None,
    ):
        """Return committee decision from cache or a fresh LLM run."""
        if pre_context is not None and pre_scores is not None:
            context = pre_context
            scores = pre_scores
        else:
            evidence = self._gather_evidence(ticker, result)
            context = build_candidate_context(ticker, evidence, now)
            scores = score_candidate(derive_score_inputs(context.evidence, now))

        digest = decision_digest(
            ticker,
            [e.evidence_id for e in context.evidence],
            [c.option_code for c in candidates],
        )
        output = None
        if self.decision_cache is not None:
            cached = self.decision_cache.get(ticker, digest, now)
            if cached is not None:
                try:
                    output = CommitteeOutput.model_validate_json(cached)
                except ValueError:
                    output = None
                if output is not None:
                    self.audit.append(
                        "committee_cache_hit",
                        {"ticker": ticker, "decision": output.decision,
                         "digest": digest[:16]},
                    )
        if output is not None:
            return output

        today = market_date(now)
        if self.llm_budget is not None and self.llm_budget.remaining(today) <= 0:
            self.audit.append(
                "llm_budget_exhausted",
                {"ticker": ticker, "used": self.llm_budget.used(today)},
            )
            result.rejected.append(
                {
                    "ticker": ticker,
                    "stage": "llm_budget",
                    "reasons": ["daily LLM token budget exhausted"],
                }
            )
            return None

        before = self.committee.usage_total()
        output = self.committee.run(
            context, scores, candidates=candidates, market_snapshot=snapshot,
            price_context=price_context, iv_rank=iv_rank_val,
        )
        usage = usage_delta(before, self.committee.usage_total())
        self.audit.append(
            "committee_run",
            {"ticker": ticker, "decision": output.decision, "output": output.model_dump(mode="json")},
        )
        self.audit.append(
            "llm_usage",
            {
                "ticker": ticker,
                **usage.model_dump(mode="json"),
                "total_tokens": usage.total_tokens,
            },
        )
        if self.llm_budget is not None:
            self.llm_budget.add(usage.total_tokens, today)
        if self.decision_cache is not None:
            self.decision_cache.put(ticker, digest, output.model_dump_json(), now)
        return output

    def _validate_proposal(
        self,
        ticker: str,
        output,
        candidates: list,
        now: datetime,
        spot: float,
        result: CycleResult,
    ):
        """Validate proposal against candidate list and Monte Carlo POP.

        Returns (candidate, expiry) on success, (None, "") on rejection.
        """
        proposal = output.proposal
        candidate = next(
            (c for c in candidates if c.option_code == proposal.option_code), None
        )
        if candidate is None:
            self.audit.append(
                "proposal_rejected",
                {"option_code": proposal.option_code, "stage": "not_in_candidates"},
            )
            result.rejected.append(
                {
                    "ticker": ticker,
                    "stage": "not_in_candidates",
                    "reasons": ["chosen contract not in candidate list"],
                }
            )
            return None, ""

        mc_floor = self.active_mandate.options.min_monte_carlo_pop
        if mc_floor > 0:
            if spot > 0 and 0 < candidate.iv <= MAX_USABLE_IV:
                hold_days = (proposal.exit_plan.time_stop - now.date()).days
                pop = win_probability(
                    side=candidate.option_side,
                    spot=spot,
                    strike=candidate.strike,
                    dte_days=candidate.dte,
                    iv=candidate.iv,
                    entry_price=proposal.limit_price,
                    take_profit_pct=proposal.exit_plan.take_profit_pct,
                    stop_loss_pct=proposal.exit_plan.stop_loss_pct,
                    hold_days=hold_days if hold_days > 0 else None,
                    seed=stable_seed(proposal.option_code),
                    as_of=now.date(),
                )
                self.audit.append(
                    "monte_carlo_pop",
                    {
                        "ticker": ticker,
                        "option_code": proposal.option_code,
                        "pop": round(pop, 4),
                        "floor": mc_floor,
                        "spot": spot,
                        "iv": candidate.iv,
                        "dte": candidate.dte,
                        "entry": proposal.limit_price,
                    },
                )
                if pop < mc_floor:
                    result.rejected.append(
                        {
                            "ticker": ticker,
                            "stage": "monte_carlo_pop",
                            "reasons": [
                                f"baseline POP {pop:.2f} below floor {mc_floor:.2f}"
                            ],
                        }
                    )
                    return None, ""
            else:
                self.audit.append(
                    "monte_carlo_pop_unavailable",
                    {"ticker": ticker, "option_code": proposal.option_code,
                     "spot": spot, "iv": candidate.iv},
                )

        return candidate, candidate.expiry

    def _execute_entry(
        self,
        now: datetime,
        ticker: str,
        held_codes: set[str],
        proposal,
        candidate,
        expiry: str,
        spot: float,
        result: CycleResult,
    ):
        """Validate liquidity, risk gate, place order, and record position."""
        quote = self.market.option_quote(
            option_code=proposal.option_code,
            expiry=expiry,
            lot_size=US_OPTION_LOT_SIZE,
            now=now,
        )
        liquidity_result = self.active_liquidity.validate(quote, proposal.contracts, now)
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

        portfolio = self._portfolio_state(now, held_codes, proposal, candidate, quote.lot_size)
        self.audit.append(
            "portfolio_greeks",
            {
                "ticker": ticker,
                "option_code": proposal.option_code,
                "total_delta": portfolio.total_delta,
                "total_gamma": portfolio.total_gamma,
                "total_vega": portfolio.total_vega,
                "total_theta": portfolio.total_theta,
                "proposal_delta": portfolio.proposal_delta,
                "proposal_gamma": portfolio.proposal_gamma,
                "proposal_vega": portfolio.proposal_vega,
                "proposal_theta": portfolio.proposal_theta,
                "after_delta": round(
                    portfolio.total_delta + (portfolio.proposal_delta or 0.0), 4
                ),
                "after_gamma": round(
                    portfolio.total_gamma + (portfolio.proposal_gamma or 0.0), 4
                ),
                "after_vega": round(
                    portfolio.total_vega + (portfolio.proposal_vega or 0.0), 4
                ),
                "after_theta": round(
                    portfolio.total_theta + (portfolio.proposal_theta or 0.0), 4
                ),
            },
        )
        decision = self.active_gate.evaluate_open(proposal, quote, portfolio, now)
        self.audit.append(
            "proposal_checked",
            {"option_code": proposal.option_code, "decision": decision.model_dump(mode="json")},
        )
        if not decision.approved:
            result.rejected.append(
                {"ticker": ticker, "stage": "risk_gate", "reasons": decision.reasons}
            )
            return None

        buy_ladder = [proposal.limit_price]
        if quote.ask > quote.bid > 0:
            mid = round((quote.bid + quote.ask) / 2, 2)
            if mid < proposal.limit_price:
                buy_ladder = [mid, proposal.limit_price]
        fill = self.orders.place_and_await(
            option_code=proposal.option_code,
            contracts=proposal.contracts,
            limit_price=proposal.limit_price,
            side="buy",
            price_ladder=buy_ladder,
            on_order_submitted=lambda payload: self.audit.append(
                "order_submitted", payload
            ),
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
            catalyst_window_end=_parse_window_end(proposal.expected_catalyst_window),
            pre_earnings_exit_date=self._pre_earnings_exit_date(ticker),
            entry_spot=spot if spot > 0 else None,
            entry_iv=candidate.iv,
            entry_delta=candidate.delta,
            entry_gamma=candidate.gamma,
            entry_vega=candidate.vega,
            entry_theta=candidate.theta,
            entry_theta_decay_pct_per_day=candidate.theta_decay_pct_per_day,
            entry_iv_rank=candidate.iv_rank,
        )
        self.audit.append(
            "order_filled",
            {"option_code": proposal.option_code, "side": "buy", "price": entry_price},
        )
        result.entries.append({"ticker": ticker, "option_code": proposal.option_code})
        return proposal.option_code

    # --- helpers --------------------------------------------------------
    def _closed_rows(self) -> list[dict[str, Any]]:
        """Closed ledger rows in close order (ISO timestamps sort correctly)."""

        return self.position_store.closed_positions()

    def _net_pnl(
        self, entry_price: float, exit_price: float, contracts: int, lot_size: int
    ) -> float:
        """Realized P/L net of broker commissions (a round trip = 2 sides).

        Paper keeps commission at 0 so the ledger mirrors the broker sim; live
        sets it so compounding, the loss stops, and reports see NET results.
        """

        gross = (exit_price - entry_price) * contracts * lot_size
        fees = 2 * self.mandate.execution.commission_per_contract_usd * contracts
        return gross - fees

    def _row_pnl(self, row: dict[str, Any]) -> float:
        return self._net_pnl(
            row["entry_price"], row["exit_price"], row["contracts"], row["lot_size"]
        )

    def _equity_and_peak(self) -> tuple[float, float]:
        """Realized equity and its high-water mark, replayed from the ledger.

        Realized-only on purpose: open-position swings must not inflate sizing.
        """

        equity = self.mandate.account.initial_capital_usd
        peak = equity
        for row in self._closed_rows():
            equity += self._row_pnl(row)
            peak = max(peak, equity)
        return round(equity, 4), round(peak, 4)

    def _refresh_sizing(self) -> None:
        """Re-derive the equity-scaled risk caps at the start of a cycle."""

        equity, peak = self._equity_and_peak()
        active = self.mandate.scaled_for_equity(equity, peak)
        self.active_mandate = active
        if active is self.mandate:
            self.active_gate = self.gate
            self.active_liquidity = self.liquidity
            return
        self.active_gate = RiskGate(active, self.gate.root_dir)
        self.active_liquidity = LiquidityValidator(active.options, active.execution)
        self.audit.append(
            "risk_caps_scaled",
            {
                "equity_usd": equity,
                "peak_equity_usd": peak,
                "max_contract_cost_usd": active.options.max_contract_cost_usd,
                "max_total_premium_at_risk_usd": active.portfolio.max_total_premium_at_risk_usd,
                "daily_loss_stop_usd": active.portfolio.daily_loss_stop_usd,
                "hard_drawdown_stop_usd": active.portfolio.hard_drawdown_stop_usd,
            },
        )

    def _realized_pnl(self, now: datetime) -> tuple[float, float, int]:
        """Return (daily realized P/L, drawdown, consecutive losses) from the ledger.

        A compounding account measures drawdown from the equity peak (the hard
        stop means "this far off the high-water mark"); a fixed account keeps
        the original below-initial-capital meaning.
        """

        today = market_date(now)
        daily_pnl = 0.0
        consecutive_losses = 0
        equity = self.mandate.account.initial_capital_usd
        peak = equity
        for row in self._closed_rows():
            pnl = self._row_pnl(row)
            equity += pnl
            peak = max(peak, equity)
            if row.get("closed_at") and self._date_of(row["closed_at"]) == today:
                daily_pnl += pnl
            consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
        reference = peak if self.mandate.account.compounding else (
            self.mandate.account.initial_capital_usd
        )
        drawdown = max(0.0, reference - equity)
        return round(daily_pnl, 4), round(drawdown, 4), consecutive_losses

    def _portfolio_state(
        self,
        now: datetime,
        held_codes: set[str],
        proposal: OpenPositionProposal,
        candidate: OptionCandidate | None = None,
        lot_size: int = US_OPTION_LOT_SIZE,
    ) -> PortfolioState:
        open_rows = self.position_store.open_positions()
        today = market_date(now)
        premium = sum(
            r["entry_price"] * r["contracts"] * r["lot_size"]
            for r in open_rows
            if r["option_code"] in held_codes
        )
        new_today = sum(
            1
            for opened_at in self.position_store.opened_at_values()
            if self._date_of(opened_at) == today
        )
        daily_pnl, drawdown, consecutive_losses = self._realized_pnl(now)
        totals = self._portfolio_greeks(open_rows, held_codes)
        proposal_greeks = (
            self._candidate_greek_exposure(candidate, proposal.contracts, lot_size)
            if candidate is not None
            else {}
        )

        return PortfolioState(
            open_positions=len(held_codes),
            total_premium_at_risk_usd=round(premium, 4),
            new_positions_today=new_today,
            daily_pnl_usd=daily_pnl,
            total_drawdown_usd=drawdown,
            consecutive_losses=consecutive_losses,
            duplicate_order_exists=proposal.option_code in held_codes,
            underlying_already_held=self._same_underlying_held(
                proposal.option_code, held_codes
            ),
            vix=self._vix(now),
            minutes_since_open=minutes_since_open(now),
            total_delta=totals["delta"],
            total_gamma=totals["gamma"],
            total_vega=totals["vega"],
            total_theta=totals["theta"],
            proposal_delta=proposal_greeks.get("delta"),
            proposal_gamma=proposal_greeks.get("gamma"),
            proposal_vega=proposal_greeks.get("vega"),
            proposal_theta=proposal_greeks.get("theta"),
            proposal_abs_delta=abs(candidate.delta) if candidate and candidate.delta is not None else None,
            proposal_gamma_per_contract=(
                candidate.gamma if candidate is not None else None
            ),
            proposal_theta_decay_pct_per_day=(
                candidate.theta_decay_pct_per_day if candidate is not None else None
            ),
            proposal_iv_rank=candidate.iv_rank if candidate is not None else None,
        )

    @staticmethod
    def _portfolio_greeks(
        ledger_rows: list[dict[str, Any]], held_codes: set[str]
    ) -> dict[str, float]:
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for row in ledger_rows:
            if row["status"] != "open" or row["option_code"] not in held_codes:
                continue
            multiplier = float(row["contracts"] * row["lot_size"])
            totals["delta"] += float(row.get("entry_delta") or 0.0) * multiplier
            totals["gamma"] += float(row.get("entry_gamma") or 0.0) * multiplier
            totals["vega"] += float(row.get("entry_vega") or 0.0) * multiplier
            totals["theta"] += float(row.get("entry_theta") or 0.0) * multiplier
        return {key: round(value, 4) for key, value in totals.items()}

    @staticmethod
    def _candidate_greek_exposure(
        candidate: OptionCandidate, contracts: int, lot_size: int
    ) -> dict[str, float]:
        multiplier = float(contracts * lot_size)
        return {
            "delta": round(float(candidate.delta or 0.0) * multiplier, 4),
            "gamma": round(float(candidate.gamma or 0.0) * multiplier, 4),
            "vega": round(float(candidate.vega or 0.0) * multiplier, 4),
            "theta": round(float(candidate.theta or 0.0) * multiplier, 4),
        }

    @staticmethod
    def _same_underlying_held(option_code: str, held_codes: set[str]) -> bool:
        """True when any held option shares the proposal's underlying ticker."""

        try:
            proposal_root = parse_us_option_code(option_code)[0]
        except ValueError:
            return False
        for held in held_codes:
            if held == option_code:
                continue
            try:
                if parse_us_option_code(held)[0] == proposal_root:
                    return True
            except ValueError:
                continue  # non-option holdings (e.g. stock) never conflict
        return False

    @staticmethod
    def _date_of(iso: str) -> date:
        return market_date(parse_iso(iso))

    def _record_error(
        self, result: CycleResult, stage: str, ref: str, exc: Exception
    ) -> None:
        detail = {"stage": stage, "ref": ref, "error": str(exc)}
        result.errors.append(detail)
        self.audit.append("cycle_step_failed", detail)
