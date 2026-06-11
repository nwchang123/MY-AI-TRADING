from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from trading_agent.data.moomoo_market import US_OPTION_LOT_SIZE, parse_us_option_code
from trading_agent.data.sec_edgar import SecEdgarClient
from trading_agent.domain.calendar import market_date, minutes_since_open
from trading_agent.domain.montecarlo import MAX_USABLE_IV, stable_seed, win_probability
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.positions import MonitoredPosition
from trading_agent.execution.orders import OrderManager
from trading_agent.domain.proposals import OpenPositionProposal, OptionCandidate
from trading_agent.domain.risk import Mandate, PortfolioState, QuoteSnapshot, RiskGate
from trading_agent.research.catalysts import build_candidate_context, derive_score_inputs
from trading_agent.research.committee import Committee, CommitteeOutput
from trading_agent.research.llm import usage_delta
from trading_agent.storage.budget import DailyTokenBudget
from trading_agent.storage.decisions import DecisionCache, decision_digest
from trading_agent.research.scoring import score_candidate
from trading_agent.storage.audit import AuditWriter
from trading_agent.storage.positions import PositionStore

# Upper bound on contracts shown to the committee per ticker, so the prompt stays
# small. The $25 cost cap already filters most chains to well under this.
_MAX_CANDIDATES = 20


@dataclass
class CycleResult:
    halted: bool = False
    circuit_breaker: str | None = None
    cooldown: bool = False
    reconciled_closed: list[str] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)
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
        # Refreshed at the start of every cycle: when the account compounds,
        # these carry the equity-scaled risk caps; otherwise they alias the
        # injected gate/liquidity unchanged.
        self.active_mandate = mandate
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
        now = self.now_fn()
        result = CycleResult()

        if self.gate.halt_path.exists():
            result.halted = True
            self.audit.append("cycle_halted", {"at": now.isoformat()})
            return result

        self._refresh_sizing()
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
                closed_at = datetime.fromisoformat(str(row["closed_at"]).replace("Z", "+00:00"))
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
                result.reconciled_closed.append(ledger["option_code"])
                self.audit.append(
                    "position_reconciled_closed",
                    {"option_code": ledger["option_code"], "at": now.isoformat()},
                )
        self._adopt_orphans(now, held_rows, result)
        return held_codes

    def _adopt_orphans(
        self, now: datetime, held_rows: list[dict[str, Any]], result: CycleResult
    ) -> None:
        """Adopt broker option positions missing from the local ledger.

        A crash between order fill and ledger write leaves a position the exit
        engine would otherwise never manage. Adopted positions get the mandate's
        standard exit plan (+100/-50, time stop at expiry -- the forced-close
        window still fires first). Non-option holdings are ignored.
        """

        ledger_open = {r["option_code"] for r in self.position_store.open_positions()}
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
                # No usable cost basis: mark from the nominal price so the exit
                # engine at least has a reference; worst case the stop fires.
                entry = float(row.get("nominal_price") or 0.0)
            if entry <= 0:
                continue
            self.position_store.open_position(
                option_code=code,
                ticker=parse_us_option_code(code)[0],
                option_side=side,
                entry_price=entry,
                contracts=int(float(row.get("qty") or 1)),
                lot_size=US_OPTION_LOT_SIZE,
                expiry=expiry,
                take_profit_pct=100.0,
                stop_loss_pct=50.0,
                time_stop=expiry,
            )
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
            self.mandate.options.force_close_before_expiry_trading_days,
            self.mandate.execution.stale_quote_seconds,
            self.mandate.execution.delayed_quote_max_age_seconds,
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
                is_delayed=quote.is_delayed,
            )
            monitored.append(position)
            marks[code] = position.mark_price()

        position_by_code = {p.option_code: p for p in monitored}
        for signal in monitor.evaluate(monitored, now):
            ledger = ledger_by_code[signal.option_code]
            limit = signal.mark_price if signal.mark_price > 0 else ledger["entry_price"]
            # Sell ladder: try the mid first, fall back to the bid -- recovers
            # roughly half the spread on fills that would happen anyway.
            position = position_by_code.get(signal.option_code)
            ladder = [limit]
            if position is not None and position.ask > position.bid > 0:
                ladder = [round((position.bid + position.ask) / 2, 2), limit]
            try:
                fill = self.orders.place_and_await(
                    option_code=signal.option_code,
                    contracts=ledger["contracts"],
                    limit_price=limit,
                    side="sell",
                    price_ladder=ladder,
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
    def _entry_regime_block(self, now: datetime) -> str | None:
        """Cycle-wide reason to skip ALL new entries, or None. Exits still run."""

        window = self.active_mandate.execution.no_entry_minutes_after_open
        if window > 0:
            minutes = minutes_since_open(now)
            if minutes is not None and minutes < window:
                return f"first {window} minutes after the open (widest spreads)"

        vix_cap = self.active_mandate.portfolio.max_vix_for_entries
        if vix_cap > 0:
            method = getattr(self.market, "underlying_snapshot", None)
            if method is not None:
                try:
                    vix = float((method("_VIX") or {}).get("price") or 0.0)
                except Exception:  # noqa: BLE001 - missing data never blocks
                    vix = 0.0
                if vix > vix_cap:
                    return f"VIX {vix:.1f} above the {vix_cap:.0f} entry cap"
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
        self, now: datetime, ticker: str, result: CycleResult, spot: float = 0.0
    ) -> list[OptionCandidate]:
        """Mandate-eligible contracts from the live chain, for the committee.

        Pulls the option chain in the mandate DTE window and keeps only contracts
        that pass the deterministic liquidity validator, so the committee can only
        ever choose a real, tradeable contract instead of guessing one blind.
        Bounded to the most liquid ``_MAX_CANDIDATES`` to keep the prompt small.
        """

        today = now.date()
        start = today + timedelta(days=self.mandate.options.min_dte)
        end = today + timedelta(days=self.mandate.options.max_dte)
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
            # Baseline math POP at the standard +100/-50 grid, shown to the
            # committee alongside each candidate so the AIs see what the math
            # says before claiming an edge. Fewer paths: a guide, not the gate.
            mc_pop: float | None = None
            if spot > 0 and 0 < iv <= MAX_USABLE_IV:
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
                )
            )
        candidates.sort(key=lambda c: c.open_interest, reverse=True)
        return candidates[:_MAX_CANDIDATES]

    def _gather_evidence(self, ticker: str, result: CycleResult) -> list[Any]:
        """SEC filings plus (optional) news headlines for one ticker.

        A news-feed failure is recorded but never blocks the entry: filings are
        the primary evidence and headlines are an enrichment.
        """

        evidence = list(self.sec_client.fetch_evidence(ticker))
        if self.news_client is not None:
            try:
                evidence.extend(self.news_client.fetch_evidence(ticker))
            except Exception as exc:  # noqa: BLE001
                self._record_error(result, "news_fetch", ticker, exc)
        return evidence

    def _underlying_snapshot(
        self, ticker: str, result: CycleResult
    ) -> dict[str, Any] | None:
        """Delayed price/IV context for the committee briefing (best-effort)."""

        method = getattr(self.market, "underlying_snapshot", None)
        if method is None:
            return None
        try:
            return method(ticker) or None
        except Exception as exc:  # noqa: BLE001
            self._record_error(result, "underlying_snapshot", ticker, exc)
            return None

    def _try_enter(
        self, now: datetime, ticker: str, held_codes: set[str], result: CycleResult
    ) -> str | None:
        # Build the real, mandate-eligible contract shortlist FIRST: if nothing is
        # tradeable, skip the SEC fetch and the committee entirely (saves tokens).
        # The underlying snapshot comes from the same cached chain payload, so
        # fetching it up front for the MC annotation costs no extra request.
        snapshot = self._underlying_snapshot(ticker, result)
        spot = float((snapshot or {}).get("price") or 0.0)
        candidates = self._eligible_candidates(now, ticker, result, spot=spot)
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

        evidence = self._gather_evidence(ticker, result)
        context = build_candidate_context(ticker, evidence, now)
        scores = score_candidate(derive_score_inputs(context.evidence, now))

        # When the evidence and the eligible contract list are unchanged since
        # the committee last reasoned about this ticker, reuse that decision
        # instead of spending five LLM calls re-deriving it. Liquidity and the
        # risk gate still re-validate fresh quotes downstream on every pass.
        digest = decision_digest(
            ticker,
            [e.evidence_id for e in context.evidence],
            [c.option_code for c in candidates],
        )
        output: CommitteeOutput | None = None
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
        if output is None:
            # Hard daily token ceiling: once exhausted, no more committee
            # passes until the next market day (cache hits still work).
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
                context, scores, candidates=candidates, market_snapshot=snapshot
            )
            usage = usage_delta(before, self.committee.usage_total())
            self.audit.append(
                "committee_run",
                {"ticker": ticker, "decision": output.decision, "output": output.model_dump(mode="json")},
            )
            self.audit.append("llm_usage", {"ticker": ticker, **usage.model_dump(mode="json")})
            if self.llm_budget is not None:
                self.llm_budget.add(usage.total_tokens, today)
            if self.decision_cache is not None:
                self.decision_cache.put(ticker, digest, output.model_dump_json(), now)
        if output.decision != "open_position" or output.proposal is None:
            return None

        proposal = output.proposal
        # The committee constrains the code to the candidate list; this backstops
        # against a contract that is not real/tradeable.
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
            return None
        expiry = candidate.expiry

        # Deterministic third vote: the no-edge Monte Carlo baseline with the
        # proposal's ACTUAL exit plan and limit price. The AI floor (0.55) asks
        # "do you believe in the edge"; this low floor asks "is the ticket
        # structurally hopeless even with one". Skipped (and audited) when the
        # spot or a usable IV is unavailable rather than blocking on missing data.
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
                    return None
            else:
                self.audit.append(
                    "monte_carlo_pop_unavailable",
                    {"ticker": ticker, "option_code": proposal.option_code,
                     "spot": spot, "iv": candidate.iv},
                )

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

        portfolio = self._portfolio_state(now, held_codes, proposal)
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

        # Buy ladder: bid at the mid first, then chase to the proposal limit.
        # Half-spread on a $0.10-0.25 ticket is ~5% of the position -- fills
        # captured at the mid go straight into expectancy.
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
    def _closed_rows(self) -> list[dict[str, Any]]:
        """Closed ledger rows in close order (ISO timestamps sort correctly)."""

        closed = [
            r
            for r in self.position_store.all_positions()
            if r["status"] == "closed" and r["exit_price"] is not None
        ]
        closed.sort(key=lambda r: str(r.get("closed_at") or ""))
        return closed

    @staticmethod
    def _row_pnl(row: dict[str, Any]) -> float:
        return (row["exit_price"] - row["entry_price"]) * row["contracts"] * row["lot_size"]

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
            underlying_already_held=self._same_underlying_held(
                proposal.option_code, held_codes
            ),
        )

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
        return market_date(datetime.fromisoformat(str(iso).replace("Z", "+00:00")))

    def _record_error(
        self, result: CycleResult, stage: str, ref: str, exc: Exception
    ) -> None:
        detail = {"stage": stage, "ref": ref, "error": str(exc)}
        result.errors.append(detail)
        self.audit.append("cycle_step_failed", detail)
