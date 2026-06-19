from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from trading_agent.domain.proposals import OpenPositionProposal


class AccountMandate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_capital_usd: float = Field(gt=0)
    live_mode_requires_operator_flag: bool
    withdrawal_capability: Literal["forbidden"]
    # When true, the dollar risk caps scale with realized equity instead of
    # staying pinned to initial_capital_usd: wins compound into larger sizing,
    # losses automatically de-risk. Counts and percentages never scale.
    compounding: bool = False


class UniverseMandate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: Literal["US"]
    min_underlying_price_usd: float = Field(gt=0)
    min_market_cap_usd: float = Field(gt=0)
    max_market_cap_usd: float = Field(gt=0)
    min_average_daily_turnover_usd: float = Field(gt=0)
    require_listed_equity: bool
    reject_otc: bool
    reject_halted: bool
    excluded_industries: list[str] = Field(default_factory=lambda: ["Shell Companies"])
    watchlist: list[str] = Field(default_factory=list)


class OptionsMandate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_opening_actions: list[Literal["buy_call", "buy_put", "buy_spread", "sell_spread"]]
    allowed_closing_actions: list[Literal["sell_to_close"]]
    contracts_per_order: int = Field(gt=0)
    min_dte: int = Field(gt=0)
    max_dte: int = Field(gt=0)
    min_option_bid_usd: float = Field(gt=0)
    max_contract_cost_usd: float = Field(gt=0)
    max_bid_ask_spread_pct: float = Field(gt=0)
    min_open_interest: int = Field(ge=0)
    min_daily_volume: int = Field(ge=0)
    use_limit_orders_only: bool
    reject_auto_exercise: bool
    force_close_before_expiry_trading_days: int = Field(ge=0)
    fee_buffer_usd: float = Field(ge=0)
    # Minimum estimated win probability for an entry. Both model lineages
    # (committee analysts AND the cross-provider adversary roles) must estimate
    # at least this probability of the trade reaching take-profit; the most
    # pessimistic estimate is the binding one. 0 disables the check.
    min_estimated_win_probability: float = Field(default=0.0, ge=0, le=1)
    # Floor for the deterministic Monte Carlo baseline POP (no-edge GBM with
    # sticky IV). Set LOW: it only rejects structurally hopeless tickets that
    # no plausible catalyst edge could rescue. 0 disables the check.
    min_monte_carlo_pop: float = Field(default=0.0, ge=0, le=1)
    # Soft-veto control. 0 (default) keeps the legacy behavior where a skeptic or
    # risk_manager VETO is an absolute block. When > 0, a veto no longer hard-
    # blocks: each standing veto instead docks the binding win-probability by this
    # amount (and drops that role's own estimate from the min), so the
    # portfolio_manager can override a veto only with conviction high enough to
    # still clear ``min_estimated_win_probability``. E.g. penalty 0.10 + floor
    # 0.55 => one veto needs PM confidence >= 0.65, two vetoes >= 0.75.
    veto_win_prob_penalty: float = Field(default=0.0, ge=0, le=1)
    # --- Phase 1: pre-catalyst (earnings) IV-ramp selection ---
    # When earnings_window_max_days > 0, the universe is picked by UPCOMING
    # earnings proximity instead of realized volume spikes: only names whose next
    # earnings date is in [min, max] calendar days out are eligible, so we buy
    # BEFORE the IV ramp and exit before the print. 0 (default) keeps the legacy
    # volume-ratio ranking, so existing behavior is unchanged until enabled.
    earnings_window_min_days: int = Field(default=0, ge=0)
    earnings_window_max_days: int = Field(default=0, ge=0)
    # Exit this many TRADING days before the earnings date so a position never
    # holds through the print (the event IV crush is exactly what the run-up play
    # avoids). 0 (default) disables the pre-earnings exit.
    pre_earnings_exit_trading_days: int = Field(default=0, ge=0)
    # Skip contracts whose implied volatility (a FRACTION: 1.5 = 150%) already
    # exceeds this -- a name that has already ramped is the peak-IV trap we are
    # trying to avoid. 0 (default) disables the IV ceiling.
    max_entry_iv: float = Field(default=0.0, ge=0)


class PortfolioMandate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_open_positions: int = Field(gt=0)
    max_total_premium_at_risk_usd: float = Field(gt=0)
    max_single_position_cost_usd: float = Field(default=0, ge=0)
    max_new_positions_per_day: int = Field(gt=0)
    daily_loss_stop_usd: float = Field(gt=0)
    hard_drawdown_stop_usd: float = Field(gt=0)
    consecutive_loss_stop: int = Field(gt=0)
    cooldown_after_consecutive_losses_hours: int = Field(gt=0)
    # Market-regime guard: no NEW entries while the VIX is above this level
    # (panic regimes blow out small-cap option spreads and inflate IV; buying
    # premium into them is structurally bad). Exits still run. 0 disables.
    max_vix_for_entries: float = Field(default=0.0, ge=0)


class ExecutionMandate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stale_quote_seconds: int = Field(gt=0)
    # Free option-data feeds (CBOE/Tradier) are ~15 min delayed, so a real-time
    # threshold would reject every quote. Quotes flagged ``is_delayed`` use this
    # wider age limit instead, which still catches a frozen/weekend feed.
    delayed_quote_max_age_seconds: int = Field(default=1200, gt=0)
    cancel_unfilled_order_seconds: int = Field(gt=0)
    max_limit_chase_pct: float = Field(ge=0)
    kill_switch_file: str = Field(min_length=1)
    # No NEW entries during the first N minutes after the open, where option
    # spreads are at their widest. Exits still run. 0 disables.
    no_entry_minutes_after_open: int = Field(default=0, ge=0)
    # Hard daily LLM token budget: once today's metered usage crosses this,
    # the committee is skipped until the next market day. Protects the API
    # balance from a runaway loop. 0 disables.
    max_daily_llm_tokens: int = Field(default=0, ge=0)
    # Broker commission per contract PER SIDE, deducted from realized ledger
    # P/L (a round trip costs 2x this). Keep 0 in paper -- Moomoo SIMULATE
    # charges no fees, and the local ledger must mirror the broker sim. Set it
    # for live so compounding, the loss stops, and reports reflect NET P/L.
    commission_per_contract_usd: float = Field(default=0.0, ge=0)

    def max_quote_age_seconds(self, is_delayed: bool) -> int:
        """Staleness ceiling for a quote, widened for delayed data feeds."""

        return (
            self.delayed_quote_max_age_seconds if is_delayed else self.stale_quote_seconds
        )


# Compounding never scales the caps below this fraction of their configured
# values, so a deep drawdown cannot shrink the limits into nonsense (the hard
# drawdown stop halts the session long before this floor matters).
COMPOUND_SCALE_FLOOR = 0.1


class Mandate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: AccountMandate
    universe: UniverseMandate
    options: OptionsMandate
    portfolio: PortfolioMandate
    execution: ExecutionMandate

    @classmethod
    def load(cls, path: Path) -> "Mandate":
        with path.open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def scaled_for_equity(
        self, equity: float, peak_equity: float | None = None
    ) -> "Mandate":
        """Risk caps proportional to realized equity (compounding accounts).

        At $200 equity on a $100 mandate, the $25 contract cap becomes $50 and
        the $60 premium-at-risk cap becomes $120; after losses they shrink the
        same way, so sizing de-risks automatically. The hard drawdown stop
        scales with PEAK equity instead, preserving its meaning of "this far
        off the high-water mark". Counts (positions, contracts) and percentage
        limits are never scaled. Returns ``self`` unchanged when the account
        does not compound.
        """

        if not self.account.compounding:
            return self
        initial = self.account.initial_capital_usd
        scale = max(equity / initial, COMPOUND_SCALE_FLOOR)
        peak = peak_equity if peak_equity is not None else equity
        drawdown_scale = max(peak / initial, COMPOUND_SCALE_FLOOR)
        if scale == 1.0 and drawdown_scale == 1.0:
            return self

        options = self.options.model_copy(
            update={
                "max_contract_cost_usd": round(
                    self.options.max_contract_cost_usd * scale, 4
                )
            }
        )
        portfolio = self.portfolio.model_copy(
            update={
                "max_total_premium_at_risk_usd": round(
                    self.portfolio.max_total_premium_at_risk_usd * scale, 4
                ),
                "daily_loss_stop_usd": round(
                    self.portfolio.daily_loss_stop_usd * scale, 4
                ),
                "hard_drawdown_stop_usd": round(
                    self.portfolio.hard_drawdown_stop_usd * drawdown_scale, 4
                ),
            }
        )
        return self.model_copy(update={"options": options, "portfolio": portfolio})


class QuoteSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_code: str
    bid: float = Field(ge=0)
    ask: float = Field(gt=0)
    open_interest: int = Field(ge=0)
    daily_volume: int = Field(ge=0)
    lot_size: int = Field(gt=0)
    expiry: date
    observed_at: datetime
    # True when sourced from a delayed feed (CBOE/Tradier). Selects the wider
    # staleness ceiling in the gate, liquidity validator, and position monitor.
    is_delayed: bool = False


class PortfolioState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open_positions: int = Field(ge=0)
    total_premium_at_risk_usd: float = Field(ge=0)
    new_positions_today: int = Field(ge=0)
    daily_pnl_usd: float
    total_drawdown_usd: float = Field(ge=0)
    consecutive_losses: int = Field(ge=0)
    duplicate_order_exists: bool = False
    # True when any open position is on the proposal's underlying (different
    # strike/expiry included): with a 2-position book, two contracts on one
    # name concentrates the whole account in a single ticker.
    underlying_already_held: bool = False


class RiskDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    reasons: list[str]
    estimated_contract_cost_usd: float
    spread_pct: float


class RiskGate:
    def __init__(self, mandate: Mandate, root_dir: Path):
        self.mandate = mandate
        self.root_dir = root_dir.resolve()

    @property
    def halt_path(self) -> Path:
        return self.root_dir / self.mandate.execution.kill_switch_file

    def evaluate_open(
        self,
        proposal: OpenPositionProposal,
        quote: QuoteSnapshot,
        portfolio: PortfolioState,
        now: datetime | None = None,
    ) -> RiskDecision:
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        reasons: list[str] = []
        spread_pct = self._spread_pct(quote.bid, quote.ask)
        estimated_cost = (
            proposal.limit_price * quote.lot_size * proposal.contracts
            + self.mandate.options.fee_buffer_usd
        )

        if self.halt_path.exists():
            reasons.append("kill switch is active")
        if quote.option_code != proposal.option_code:
            reasons.append("proposal option code does not match quote")
        if proposal.contracts != self.mandate.options.contracts_per_order:
            reasons.append("contract quantity violates mandate")
        if quote.bid < self.mandate.options.min_option_bid_usd:
            reasons.append("option bid is below minimum")
        if spread_pct > self.mandate.options.max_bid_ask_spread_pct:
            reasons.append("bid-ask spread is too wide")
        if quote.open_interest < self.mandate.options.min_open_interest:
            reasons.append("open interest is below minimum")
        if quote.daily_volume < self.mandate.options.min_daily_volume:
            reasons.append("daily option volume is below minimum")

        quote_age = (current_time - quote.observed_at).total_seconds()
        max_age = self.mandate.execution.max_quote_age_seconds(quote.is_delayed)
        if quote_age < 0 or quote_age > max_age:
            reasons.append("quote is stale")

        dte = (quote.expiry - current_time.date()).days
        if not self.mandate.options.min_dte <= dte <= self.mandate.options.max_dte:
            reasons.append("days to expiry violate mandate")

        # Deterministic backstop for the committee's own probability check: the
        # proposal's confidence is the PM's estimated win probability, and a
        # proposal below the mandate floor can never reach the broker even if
        # the committee-side check were bypassed.
        if proposal.confidence < self.mandate.options.min_estimated_win_probability:
            reasons.append("estimated win probability below mandate minimum")

        if proposal.limit_price > proposal.max_limit_price:
            reasons.append("limit price exceeds proposal maximum")
        maximum_chase = quote.ask * (1 + self.mandate.execution.max_limit_chase_pct / 100)
        if proposal.max_limit_price > maximum_chase:
            reasons.append("proposal maximum exceeds allowed quote chase")
        if estimated_cost > self.mandate.options.max_contract_cost_usd:
            reasons.append("contract cost exceeds mandate")

        if portfolio.open_positions >= self.mandate.portfolio.max_open_positions:
            reasons.append("maximum open positions reached")
        if (
            portfolio.total_premium_at_risk_usd + estimated_cost
            > self.mandate.portfolio.max_total_premium_at_risk_usd
        ):
            reasons.append("total premium at risk would exceed mandate")
        if (
            self.mandate.portfolio.max_single_position_cost_usd > 0
            and estimated_cost > self.mandate.portfolio.max_single_position_cost_usd
        ):
            reasons.append("single position cost exceeds mandate limit")
        if portfolio.new_positions_today >= self.mandate.portfolio.max_new_positions_per_day:
            reasons.append("daily new-position limit reached")
        if portfolio.daily_pnl_usd <= -self.mandate.portfolio.daily_loss_stop_usd:
            reasons.append("daily loss stop is active")
        if portfolio.total_drawdown_usd >= self.mandate.portfolio.hard_drawdown_stop_usd:
            reasons.append("hard drawdown stop is active")
        if portfolio.consecutive_losses >= self.mandate.portfolio.consecutive_loss_stop:
            reasons.append("consecutive-loss cooldown is active")
        if portfolio.duplicate_order_exists:
            reasons.append("duplicate order already exists")
        if portfolio.underlying_already_held:
            reasons.append("an open position already exists on this underlying")

        return RiskDecision(
            approved=not reasons,
            reasons=reasons,
            estimated_contract_cost_usd=round(estimated_cost, 4),
            spread_pct=round(spread_pct, 4),
        )

    @staticmethod
    def _spread_pct(bid: float, ask: float) -> float:
        if ask <= bid or bid <= 0:
            return float("inf")
        return ((ask - bid) / ((ask + bid) / 2)) * 100

