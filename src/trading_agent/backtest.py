from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trading_agent.domain.calendar import market_date
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.positions import MonitoredPosition
from trading_agent.domain.proposals import OpenPositionProposal
from trading_agent.domain.risk import Mandate, PortfolioState, QuoteSnapshot, RiskGate
from trading_agent.execution.monitor import PositionMonitor


class BacktestStep(BaseModel):
    """One point-in-time replay step with quotes and an optional proposal."""

    model_config = ConfigDict(extra="forbid")

    now: datetime
    quote: QuoteSnapshot | None = None
    quotes: list[QuoteSnapshot] = Field(default_factory=list)
    proposal: OpenPositionProposal | None = None
    label: str | None = None


class BacktestScenario(BaseModel):
    """Offline backtest v0 input.

    The runner does not fetch market data or call the LLM. Feed it point-in-time
    quotes and proposals captured elsewhere so the live risk and exit rules can
    be replayed deterministically.
    """

    model_config = ConfigDict(extra="forbid")

    initial_capital_usd: float | None = Field(default=None, gt=0)
    steps: list[BacktestStep] = Field(min_length=1)


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: dict[str, Any]
    events: list[dict[str, Any]]
    open_positions: list[dict[str, Any]]
    closed_trades: list[dict[str, Any]]


@dataclass
class _OpenTrade:
    option_code: str
    ticker: str
    option_side: str
    entry_price: float
    contracts: int
    lot_size: int
    expiry: Any
    take_profit_pct: float
    stop_loss_pct: float
    time_stop: Any
    opened_at: datetime


@dataclass
class _ClosedTrade:
    option_code: str
    ticker: str
    entry_price: float
    exit_price: float
    contracts: int
    lot_size: int
    opened_at: datetime
    closed_at: datetime
    reason: str

    @property
    def pnl_usd(self) -> float:
        return round(
            (self.exit_price - self.entry_price) * self.contracts * self.lot_size,
            4,
        )


def load_backtest_scenario(path: Path) -> BacktestScenario:
    return BacktestScenario.model_validate_json(path.read_text(encoding="utf-8"))


def run_backtest_file(
    path: Path, *, mandate: Mandate, root_dir: Path
) -> BacktestResult:
    return run_backtest(load_backtest_scenario(path), mandate=mandate, root_dir=root_dir)


def run_backtest(
    scenario: BacktestScenario, *, mandate: Mandate, root_dir: Path
) -> BacktestResult:
    """Replay proposals and option quotes through the live guard components."""

    _require_chronological_steps(scenario.steps)
    initial_capital = scenario.initial_capital_usd or mandate.account.initial_capital_usd
    gate = RiskGate(mandate, root_dir)
    liquidity = LiquidityValidator(mandate.options, mandate.execution)
    monitor = PositionMonitor(
        mandate.options.force_close_before_expiry_trading_days,
        mandate.execution.stale_quote_seconds,
    )

    open_trades: list[_OpenTrade] = []
    closed_trades: list[_ClosedTrade] = []
    last_quotes: dict[str, QuoteSnapshot] = {}
    events: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = [
        {"time": "initial", "equity_usd": round(initial_capital, 4)}
    ]

    for step in scenario.steps:
        _require_aware(step.now, "step.now")
        step_quotes = _quotes_by_code(step)
        last_quotes.update(step_quotes)

        _run_exit_step(
            step=step,
            quotes=step_quotes,
            monitor=monitor,
            open_trades=open_trades,
            closed_trades=closed_trades,
            events=events,
        )
        if step.proposal is not None:
            _run_entry_step(
                step=step,
                quote=step_quotes.get(step.proposal.option_code),
                gate=gate,
                liquidity=liquidity,
                mandate=mandate,
                open_trades=open_trades,
                closed_trades=closed_trades,
                events=events,
            )
        equity_curve.append(
            {
                "time": step.now.isoformat(),
                "equity_usd": round(
                    initial_capital
                    + _realized_pnl(closed_trades)
                    + _unrealized_pnl(open_trades, last_quotes),
                    4,
                ),
            }
        )

    summary = _summary(
        initial_capital,
        open_trades,
        closed_trades,
        events,
        equity_curve,
        last_quotes,
    )
    return BacktestResult(
        summary=summary,
        events=events,
        open_positions=[_open_payload(trade, last_quotes.get(trade.option_code)) for trade in open_trades],
        closed_trades=[_closed_payload(trade) for trade in closed_trades],
    )


def _quotes_by_code(step: BacktestStep) -> dict[str, QuoteSnapshot]:
    quotes = {quote.option_code: quote for quote in step.quotes}
    if step.quote is not None:
        quotes[step.quote.option_code] = step.quote
    return quotes


def _run_exit_step(
    *,
    step: BacktestStep,
    quotes: dict[str, QuoteSnapshot],
    monitor: PositionMonitor,
    open_trades: list[_OpenTrade],
    closed_trades: list[_ClosedTrade],
    events: list[dict[str, Any]],
) -> None:
    monitored: list[MonitoredPosition] = []
    by_code: dict[str, _OpenTrade] = {}
    for trade in open_trades:
        quote = quotes.get(trade.option_code)
        if quote is None:
            continue
        monitored.append(
            MonitoredPosition(
                option_code=trade.option_code,
                option_side=trade.option_side,  # type: ignore[arg-type]
                entry_price=trade.entry_price,
                contracts=trade.contracts,
                lot_size=trade.lot_size,
                expiry=trade.expiry,
                take_profit_pct=trade.take_profit_pct,
                stop_loss_pct=trade.stop_loss_pct,
                time_stop=trade.time_stop,
                bid=quote.bid,
                ask=quote.ask,
                observed_at=quote.observed_at,
            )
        )
        by_code[trade.option_code] = trade

    for signal in monitor.evaluate(monitored, step.now):
        trade = by_code[signal.option_code]
        closed = _ClosedTrade(
            option_code=trade.option_code,
            ticker=trade.ticker,
            entry_price=trade.entry_price,
            exit_price=signal.mark_price,
            contracts=trade.contracts,
            lot_size=trade.lot_size,
            opened_at=trade.opened_at,
            closed_at=step.now,
            reason=signal.reason,
        )
        open_trades.remove(trade)
        closed_trades.append(closed)
        _append_event(
            events,
            step.now,
            "position_closed",
            {
                "ticker": trade.ticker,
                "option_code": trade.option_code,
                "reason": signal.reason,
                "exit_price": signal.mark_price,
                "realized_pnl_usd": closed.pnl_usd,
            },
        )


def _run_entry_step(
    *,
    step: BacktestStep,
    quote: QuoteSnapshot | None,
    gate: RiskGate,
    liquidity: LiquidityValidator,
    mandate: Mandate,
    open_trades: list[_OpenTrade],
    closed_trades: list[_ClosedTrade],
    events: list[dict[str, Any]],
) -> None:
    proposal = step.proposal
    assert proposal is not None
    if quote is None:
        _reject(events, step.now, proposal, "missing_quote", ["missing quote"])
        return

    liquidity_result = liquidity.validate(quote, proposal.contracts, step.now)
    if not liquidity_result.passed:
        _reject(events, step.now, proposal, "liquidity", liquidity_result.reasons)
        return

    decision = gate.evaluate_open(
        proposal,
        quote,
        _portfolio_state(step.now, open_trades, closed_trades, proposal),
        step.now,
    )
    if not decision.approved:
        _reject(events, step.now, proposal, "risk_gate", decision.reasons)
        return

    if quote.ask > proposal.limit_price:
        _reject(
            events,
            step.now,
            proposal,
            "unfilled",
            ["ask is above the limit price"],
            {"ask": quote.ask, "limit_price": proposal.limit_price},
        )
        return

    trade = _OpenTrade(
        option_code=proposal.option_code,
        ticker=proposal.ticker.upper(),
        option_side=proposal.option_side,
        entry_price=quote.ask,
        contracts=proposal.contracts,
        lot_size=quote.lot_size,
        expiry=quote.expiry,
        take_profit_pct=proposal.exit_plan.take_profit_pct,
        stop_loss_pct=proposal.exit_plan.stop_loss_pct,
        time_stop=proposal.exit_plan.time_stop,
        opened_at=step.now,
    )
    open_trades.append(trade)
    _append_event(
        events,
        step.now,
        "position_opened",
        {
            "ticker": trade.ticker,
            "option_code": trade.option_code,
            "entry_price": trade.entry_price,
            "contracts": trade.contracts,
            "premium_at_risk_usd": round(
                trade.entry_price * trade.contracts * trade.lot_size
                + mandate.options.fee_buffer_usd,
                4,
            ),
        },
    )


def _portfolio_state(
    now: datetime,
    open_trades: list[_OpenTrade],
    closed_trades: list[_ClosedTrade],
    proposal: OpenPositionProposal,
) -> PortfolioState:
    today = market_date(now)
    return PortfolioState(
        open_positions=len(open_trades),
        total_premium_at_risk_usd=round(
            sum(t.entry_price * t.contracts * t.lot_size for t in open_trades),
            4,
        ),
        new_positions_today=sum(1 for t in open_trades if market_date(t.opened_at) == today)
        + sum(1 for t in closed_trades if market_date(t.opened_at) == today),
        daily_pnl_usd=round(
            sum(t.pnl_usd for t in closed_trades if market_date(t.closed_at) == today),
            4,
        ),
        total_drawdown_usd=round(max(0.0, -_realized_pnl(closed_trades)), 4),
        consecutive_losses=_consecutive_losses(closed_trades),
        duplicate_order_exists=any(t.option_code == proposal.option_code for t in open_trades),
    )


def _summary(
    initial_capital: float,
    open_trades: list[_OpenTrade],
    closed_trades: list[_ClosedTrade],
    events: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    last_quotes: dict[str, QuoteSnapshot],
) -> dict[str, Any]:
    realized = _realized_pnl(closed_trades)
    unrealized = _unrealized_pnl(open_trades, last_quotes)
    wins = sum(1 for trade in closed_trades if trade.pnl_usd > 0)
    losses = sum(1 for trade in closed_trades if trade.pnl_usd < 0)
    rejected = [event for event in events if event["event_type"] == "entry_rejected"]
    rejected_by_stage = Counter(event["payload"].get("stage", "unknown") for event in rejected)
    trade_pnls = [trade.pnl_usd for trade in closed_trades]

    return {
        "initial_capital_usd": round(initial_capital, 4),
        "ending_equity_usd": round(initial_capital + realized + unrealized, 4),
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": round(unrealized, 4),
        "max_drawdown_usd": _max_drawdown(equity_curve),
        "trades_opened": sum(1 for event in events if event["event_type"] == "position_opened"),
        "trades_closed": len(closed_trades),
        "open_positions": len(open_trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(closed_trades), 4) if closed_trades else None,
        "avg_trade_pnl_usd": round(sum(trade_pnls) / len(trade_pnls), 4)
        if trade_pnls
        else None,
        "max_consecutive_losses": _max_consecutive_losses(closed_trades),
        "rejections": len(rejected),
        "rejected_by_stage": dict(sorted(rejected_by_stage.items())),
        "equity_curve": equity_curve,
    }


def _unrealized_pnl(
    open_trades: list[_OpenTrade], quotes: dict[str, QuoteSnapshot]
) -> float:
    total = 0.0
    for trade in open_trades:
        quote = quotes.get(trade.option_code)
        if quote is None:
            continue
        total += (quote.bid - trade.entry_price) * trade.contracts * trade.lot_size
    return round(total, 4)


def _realized_pnl(closed_trades: list[_ClosedTrade]) -> float:
    return round(sum(trade.pnl_usd for trade in closed_trades), 4)


def _max_drawdown(equity_curve: list[dict[str, Any]]) -> float:
    peak: float | None = None
    max_drawdown = 0.0
    for point in equity_curve:
        equity = float(point["equity_usd"])
        peak = equity if peak is None else max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return round(max_drawdown, 4)


def _consecutive_losses(closed_trades: list[_ClosedTrade]) -> int:
    count = 0
    for trade in closed_trades:
        count = count + 1 if trade.pnl_usd < 0 else 0
    return count


def _max_consecutive_losses(closed_trades: list[_ClosedTrade]) -> int:
    best = 0
    current = 0
    for trade in closed_trades:
        current = current + 1 if trade.pnl_usd < 0 else 0
        best = max(best, current)
    return best


def _open_payload(trade: _OpenTrade, quote: QuoteSnapshot | None) -> dict[str, Any]:
    mark = quote.bid if quote is not None else None
    return {
        "ticker": trade.ticker,
        "option_code": trade.option_code,
        "entry_price": trade.entry_price,
        "mark_price": mark,
        "unrealized_pnl_usd": round(
            (mark - trade.entry_price) * trade.contracts * trade.lot_size, 4
        )
        if mark is not None
        else None,
        "opened_at": trade.opened_at.isoformat(),
    }


def _closed_payload(trade: _ClosedTrade) -> dict[str, Any]:
    return {
        "ticker": trade.ticker,
        "option_code": trade.option_code,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "realized_pnl_usd": trade.pnl_usd,
        "opened_at": trade.opened_at.isoformat(),
        "closed_at": trade.closed_at.isoformat(),
        "reason": trade.reason,
    }


def _reject(
    events: list[dict[str, Any]],
    now: datetime,
    proposal: OpenPositionProposal,
    stage: str,
    reasons: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "ticker": proposal.ticker.upper(),
        "option_code": proposal.option_code,
        "stage": stage,
        "reasons": reasons,
    }
    if extra:
        payload.update(extra)
    _append_event(events, now, "entry_rejected", payload)


def _append_event(
    events: list[dict[str, Any]], now: datetime, event_type: str, payload: dict[str, Any]
) -> None:
    events.append(
        {
            "time": now.isoformat(),
            "event_type": event_type,
            "payload": json.loads(json.dumps(payload, default=str)),
        }
    )


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")


def _require_chronological_steps(steps: list[BacktestStep]) -> None:
    previous: datetime | None = None
    for step in steps:
        _require_aware(step.now, "step.now")
        if previous is not None and step.now < previous:
            raise ValueError("backtest steps must be chronological")
        previous = step.now
