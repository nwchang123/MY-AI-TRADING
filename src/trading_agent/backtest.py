from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class CostModel(BaseModel):
    """Per-order trading-cost assumptions for a small-capital account.

    All fields are optional and default to zero; when ``costs`` is omitted from a
    scenario the runner falls back to a flat per-side fee of the mandate's
    ``fee_buffer_usd`` (the prior behaviour). Spread is always modelled implicitly
    by filling buys at the ask and marking/closing longs at the bid.
    """

    model_config = ConfigDict(extra="forbid")

    # Per-side broker commission: max(min, per_contract * contracts) + platform.
    commission_per_contract_usd: float = Field(default=0.0, ge=0)
    commission_min_usd: float = Field(default=0.0, ge=0)
    platform_fee_per_order_usd: float = Field(default=0.0, ge=0)
    # One-off promo (e.g. a commission-free card): the first N USD of commissions
    # are waived, then full cost resumes -- so the report shows the steady state.
    commission_waiver_usd: float = Field(default=0.0, ge=0)
    # Adverse marketable-fill slippage per share. On entry it is capped by the
    # proposal limit (a limit order never pays above it); on exit it can push the
    # fill below the bid (a forced close behaves like a marketable order).
    slippage_usd_per_share: float = Field(default=0.0, ge=0)
    # If the bid-ask spread at exit exceeds this, the close does not fill that
    # step (mirrors the live cancel-and-retry); 0 disables the check.
    max_exit_fill_spread_pct: float = Field(default=0.0, ge=0)


class BacktestScenario(BaseModel):
    """Offline backtest v0 input.

    The runner does not fetch market data or call the LLM. Feed it point-in-time
    quotes and proposals captured elsewhere so the live risk and exit rules can
    be replayed deterministically.
    """

    model_config = ConfigDict(extra="forbid")

    initial_capital_usd: float | None = Field(default=None, gt=0)
    costs: CostModel | None = None
    steps: list[BacktestStep] = Field(min_length=1)


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: dict[str, Any]
    events: list[dict[str, Any]]
    open_positions: list[dict[str, Any]]
    closed_trades: list[dict[str, Any]]


class BacktestSweepConfig(BaseModel):
    """Batch parameter grid over the same deterministic replay scenario."""

    model_config = ConfigDict(extra="forbid")

    scenario: BacktestScenario
    take_profit_pct: list[float] = Field(default_factory=lambda: [100.0])
    stop_loss_pct: list[float] = Field(default_factory=lambda: [50.0])
    max_bid_ask_spread_pct: list[float] | None = None
    min_estimated_win_probability: list[float] | None = None

    @field_validator("take_profit_pct", "stop_loss_pct")
    @classmethod
    def validate_positive_grid(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("sweep grid cannot be empty")
        if any(value <= 0 for value in values):
            raise ValueError("sweep grid values must be positive")
        return values

    @field_validator("max_bid_ask_spread_pct")
    @classmethod
    def validate_spread_grid(cls, values: list[float] | None) -> list[float] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("max_bid_ask_spread_pct cannot be empty")
        if any(value <= 0 for value in values):
            raise ValueError("max_bid_ask_spread_pct values must be positive")
        return values

    @field_validator("min_estimated_win_probability")
    @classmethod
    def validate_probability_grid(cls, values: list[float] | None) -> list[float] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("min_estimated_win_probability cannot be empty")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("min_estimated_win_probability values must be in [0, 1]")
        return values


class BacktestSweepRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    parameters: dict[str, float]
    summary: dict[str, Any]


class BacktestSweepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best: BacktestSweepRun | None
    runs: list[BacktestSweepRun]


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
    entry_fee_usd: float


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
    entry_fee_usd: float
    exit_fee_usd: float

    @property
    def gross_pnl_usd(self) -> float:
        return round(
            (self.exit_price - self.entry_price) * self.contracts * self.lot_size,
            4,
        )

    @property
    def fees_usd(self) -> float:
        return round(self.entry_fee_usd + self.exit_fee_usd, 4)

    @property
    def pnl_usd(self) -> float:
        return round(self.gross_pnl_usd - self.fees_usd, 4)


@dataclass
class _CostEngine:
    """Applies the scenario's cost assumptions; tracks the commission waiver."""

    per_contract: float
    minimum: float
    platform: float
    slippage: float
    waiver_remaining: float
    max_exit_fill_spread_pct: float

    @classmethod
    def build(cls, costs: CostModel | None, mandate: Mandate) -> "_CostEngine":
        if costs is None:
            # Back-compat: a flat per-side fee from the risk buffer, no slippage,
            # no waiver, every exit fills.
            return cls(0.0, mandate.options.fee_buffer_usd, 0.0, 0.0, 0.0, 0.0)
        return cls(
            costs.commission_per_contract_usd,
            costs.commission_min_usd,
            costs.platform_fee_per_order_usd,
            costs.slippage_usd_per_share,
            costs.commission_waiver_usd,
            costs.max_exit_fill_spread_pct,
        )

    def commission(self, contracts: int) -> float:
        """One side's commission, after consuming any remaining waiver."""

        raw = round(max(self.minimum, self.per_contract * contracts) + self.platform, 4)
        if self.waiver_remaining > 0:
            waived = min(raw, self.waiver_remaining)
            self.waiver_remaining = round(self.waiver_remaining - waived, 4)
            return round(raw - waived, 4)
        return raw

    def entry_fill(self, ask: float, limit_price: float) -> float:
        # A limit buy never pays above the limit, so slippage is capped there.
        return round(min(limit_price, ask + self.slippage), 4)

    def exit_fill(self, bid: float) -> float:
        return round(max(0.0, bid - self.slippage), 4)

    def exit_allowed(self, bid: float, ask: float) -> bool:
        if self.max_exit_fill_spread_pct <= 0:
            return True
        return _spread_pct(bid, ask) <= self.max_exit_fill_spread_pct


def load_backtest_scenario(path: Path) -> BacktestScenario:
    return BacktestScenario.model_validate_json(path.read_text(encoding="utf-8"))


def load_backtest_sweep_config(path: Path) -> BacktestSweepConfig:
    return BacktestSweepConfig.model_validate_json(path.read_text(encoding="utf-8"))


def run_backtest_file(
    path: Path, *, mandate: Mandate, root_dir: Path
) -> BacktestResult:
    return run_backtest(load_backtest_scenario(path), mandate=mandate, root_dir=root_dir)


def run_backtest_sweep_file(
    path: Path, *, mandate: Mandate, root_dir: Path
) -> BacktestSweepResult:
    return run_backtest_sweep(
        load_backtest_sweep_config(path), mandate=mandate, root_dir=root_dir
    )


def run_backtest_sweep(
    config: BacktestSweepConfig, *, mandate: Mandate, root_dir: Path
) -> BacktestSweepResult:
    """Run a TP/SL and gate-threshold grid against one offline scenario."""

    take_profit_grid = _unique_floats(config.take_profit_pct)
    stop_loss_grid = _unique_floats(config.stop_loss_pct)
    spread_grid = _unique_floats(config.max_bid_ask_spread_pct or [
        mandate.options.max_bid_ask_spread_pct
    ])
    win_grid = _unique_floats(config.min_estimated_win_probability or [
        mandate.options.min_estimated_win_probability
    ])
    runs: list[BacktestSweepRun] = []
    for take_profit in take_profit_grid:
        for stop_loss in stop_loss_grid:
            scenario = _scenario_with_exit_grid(
                config.scenario,
                take_profit_pct=take_profit,
                stop_loss_pct=stop_loss,
            )
            for spread_cap in spread_grid:
                for win_floor in win_grid:
                    run_mandate = mandate.model_copy(
                        update={
                            "options": mandate.options.model_copy(
                                update={
                                    "max_bid_ask_spread_pct": spread_cap,
                                    "min_estimated_win_probability": win_floor,
                                }
                            )
                        }
                    )
                    result = run_backtest(
                        scenario, mandate=run_mandate, root_dir=root_dir
                    )
                    runs.append(
                        BacktestSweepRun(
                            rank=0,
                            parameters={
                                "take_profit_pct": take_profit,
                                "stop_loss_pct": stop_loss,
                                "max_bid_ask_spread_pct": spread_cap,
                                "min_estimated_win_probability": win_floor,
                            },
                            summary=result.summary,
                        )
                    )

    ranked = sorted(
        runs,
        key=lambda run: (
            float(run.summary.get("ending_equity_usd") or 0.0),
            -float(run.summary.get("max_drawdown_usd") or 0.0),
            float(run.summary.get("profit_factor") or 0.0),
        ),
        reverse=True,
    )
    ranked = [
        run.model_copy(update={"rank": idx + 1})
        for idx, run in enumerate(ranked)
    ]
    return BacktestSweepResult(best=ranked[0] if ranked else None, runs=ranked)


def _unique_floats(values: list[float]) -> list[float]:
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(float(value), 10)
        if key in seen:
            continue
        seen.add(key)
        unique.append(float(value))
    return unique


def run_backtest(
    scenario: BacktestScenario, *, mandate: Mandate, root_dir: Path
) -> BacktestResult:
    """Replay proposals and option quotes through the live guard components."""

    _require_chronological_steps(scenario.steps)
    initial_capital = scenario.initial_capital_usd or mandate.account.initial_capital_usd
    gate = RiskGate(mandate, root_dir)
    liquidity = LiquidityValidator(mandate.options, mandate.execution)
    cost = _CostEngine.build(scenario.costs, mandate)
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
            cost=cost,
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
                cost=cost,
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
        cost,
    )
    return BacktestResult(
        summary=summary,
        events=events,
        open_positions=[_open_payload(trade, last_quotes.get(trade.option_code)) for trade in open_trades],
        closed_trades=[_closed_payload(trade) for trade in closed_trades],
    )


def _scenario_with_exit_grid(
    scenario: BacktestScenario,
    *,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> BacktestScenario:
    steps: list[BacktestStep] = []
    for step in scenario.steps:
        proposal = step.proposal
        if proposal is not None:
            proposal = proposal.model_copy(
                update={
                    "exit_plan": proposal.exit_plan.model_copy(
                        update={
                            "take_profit_pct": take_profit_pct,
                            "stop_loss_pct": stop_loss_pct,
                        }
                    )
                }
            )
        steps.append(step.model_copy(update={"proposal": proposal}, deep=True))
    return scenario.model_copy(update={"steps": steps}, deep=True)


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
    cost: _CostEngine,
    open_trades: list[_OpenTrade],
    closed_trades: list[_ClosedTrade],
    events: list[dict[str, Any]],
) -> None:
    monitored: list[MonitoredPosition] = []
    by_code: dict[str, _OpenTrade] = {}
    quote_by_code: dict[str, QuoteSnapshot] = {}
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
        quote_by_code[trade.option_code] = quote

    for signal in monitor.evaluate(monitored, step.now):
        trade = by_code[signal.option_code]
        exit_quote = quote_by_code[signal.option_code]
        if not cost.exit_allowed(exit_quote.bid, exit_quote.ask):
            # Spread too wide to get filled this step; the live engine cancels and
            # retries, so the position lingers and is revisited next step.
            _append_event(
                events,
                step.now,
                "exit_unfilled",
                {
                    "ticker": trade.ticker,
                    "option_code": trade.option_code,
                    "reason": signal.reason,
                    "spread_pct": _spread_pct(exit_quote.bid, exit_quote.ask),
                },
            )
            continue
        exit_price = cost.exit_fill(signal.mark_price)
        closed = _ClosedTrade(
            option_code=trade.option_code,
            ticker=trade.ticker,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            contracts=trade.contracts,
            lot_size=trade.lot_size,
            opened_at=trade.opened_at,
            closed_at=step.now,
            reason=signal.reason,
            entry_fee_usd=trade.entry_fee_usd,
            exit_fee_usd=cost.commission(trade.contracts),
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
                "exit_price": exit_price,
                "gross_pnl_usd": closed.gross_pnl_usd,
                "fees_usd": closed.fees_usd,
                "realized_pnl_usd": closed.pnl_usd,
                "spread_pct": _spread_pct(exit_quote.bid, exit_quote.ask),
            },
        )


def _run_entry_step(
    *,
    step: BacktestStep,
    quote: QuoteSnapshot | None,
    gate: RiskGate,
    liquidity: LiquidityValidator,
    cost: _CostEngine,
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
        entry_price=cost.entry_fill(quote.ask, proposal.limit_price),
        contracts=proposal.contracts,
        lot_size=quote.lot_size,
        expiry=quote.expiry,
        take_profit_pct=proposal.exit_plan.take_profit_pct,
        stop_loss_pct=proposal.exit_plan.stop_loss_pct,
        time_stop=proposal.exit_plan.time_stop,
        opened_at=step.now,
        entry_fee_usd=cost.commission(proposal.contracts),
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
            "entry_fee_usd": trade.entry_fee_usd,
            "premium_at_risk_usd": round(
                trade.entry_price * trade.contracts * trade.lot_size
                + trade.entry_fee_usd,
                4,
            ),
            "spread_pct": _spread_pct(quote.bid, quote.ask),
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
    cost: _CostEngine,
) -> dict[str, Any]:
    realized = _realized_pnl(closed_trades)
    unrealized = _unrealized_pnl(open_trades, last_quotes)
    wins = sum(1 for trade in closed_trades if trade.pnl_usd > 0)
    losses = sum(1 for trade in closed_trades if trade.pnl_usd < 0)
    rejected = [event for event in events if event["event_type"] == "entry_rejected"]
    rejected_by_stage = Counter(event["payload"].get("stage", "unknown") for event in rejected)
    trade_pnls = [trade.pnl_usd for trade in closed_trades]
    win_pnl = sum(trade.pnl_usd for trade in closed_trades if trade.pnl_usd > 0)
    loss_pnl = sum(trade.pnl_usd for trade in closed_trades if trade.pnl_usd < 0)
    fees_paid = round(
        sum(trade.entry_fee_usd for trade in open_trades)
        + sum(trade.fees_usd for trade in closed_trades),
        4,
    )
    entry_spreads = _event_values(events, "position_opened", "spread_pct")
    exit_spreads = _event_values(events, "position_closed", "spread_pct")
    exits_unfilled = sum(1 for event in events if event["event_type"] == "exit_unfilled")
    ending_equity = round(initial_capital + realized + unrealized, 4)

    return {
        "initial_capital_usd": round(initial_capital, 4),
        "ending_equity_usd": ending_equity,
        "return_pct": round((ending_equity - initial_capital) / initial_capital * 100, 4),
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": round(unrealized, 4),
        "fees_paid_usd": fees_paid,
        "commission_waiver_remaining_usd": round(cost.waiver_remaining, 4),
        "exits_unfilled": exits_unfilled,
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
        "profit_factor": round(win_pnl / abs(loss_pnl), 4) if loss_pnl < 0 else None,
        "best_trade_pnl_usd": max(trade_pnls) if trade_pnls else None,
        "worst_trade_pnl_usd": min(trade_pnls) if trade_pnls else None,
        "max_consecutive_losses": _max_consecutive_losses(closed_trades),
        "rejections": len(rejected),
        "rejected_by_stage": dict(sorted(rejected_by_stage.items())),
        "avg_entry_spread_pct": _mean(entry_spreads),
        "avg_exit_spread_pct": _mean(exit_spreads),
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
        total += (
            (quote.bid - trade.entry_price) * trade.contracts * trade.lot_size
            - trade.entry_fee_usd
        )
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
        "entry_fee_usd": trade.entry_fee_usd,
        "gross_unrealized_pnl_usd": round(
            (mark - trade.entry_price) * trade.contracts * trade.lot_size, 4
        )
        if mark is not None
        else None,
        "unrealized_pnl_usd": round(
            (mark - trade.entry_price) * trade.contracts * trade.lot_size
            - trade.entry_fee_usd,
            4,
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
        "gross_pnl_usd": trade.gross_pnl_usd,
        "fees_usd": trade.fees_usd,
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


def _spread_pct(bid: float, ask: float) -> float:
    if ask <= bid or bid <= 0:
        return float("inf")
    return round(((ask - bid) / ((ask + bid) / 2)) * 100, 4)


def _event_values(
    events: list[dict[str, Any]], event_type: str, payload_key: str
) -> list[float]:
    values: list[float] = []
    for event in events:
        if event.get("event_type") != event_type:
            continue
        value = event.get("payload", {}).get(payload_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
