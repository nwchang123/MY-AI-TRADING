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
    # Pre-move accumulation bias: down-rank a high-volume name by this factor times
    # its absolute % day-move, so volume-ratio selection favors names where volume
    # is building BEFORE the price move (catalyst not yet priced in) over already-
    # spiked names the committee vetoes as "priced in". score = volume_ratio /
    # (1 + penalty * |change_pct|). 0 = legacy pure volume_ratio ranking.
    premove_change_penalty: float = Field(default=0.0, ge=0)


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
    # Pre-screen catalyst gate: candidates whose deterministic catalyst score
    # (range -25..+75; catalyst*50 + operations*25 - contradictions*25) is below
    # this never reach the LLM committee. This is a COST filter (skip hopeless
    # names to save ~33k tokens/pass), NOT the quality gate -- the committee +
    # win-prob/MC floors decide quality downstream. Set too high and the committee
    # never sees anything: at 20.0 the whole 2026-06-18..24 run scored a max of
    # 17.8 (median 10.2) so ZERO candidates passed and ZERO entries opened for 6
    # sessions. Default 20.0 preserves the historical (live) behavior.
    min_catalyst_score_for_committee: float = Field(default=20.0)
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
    # Hard Greeks filters for long-premium entries. 0 disables each threshold.
    min_abs_delta: float = Field(default=0.0, ge=0, le=1)
    max_gamma_per_contract: float = Field(default=0.0, ge=0)
    max_theta_decay_pct_per_day: float = Field(default=0.0, ge=0)
    max_iv_rank_for_long_premium: float = Field(default=0.0, ge=0, le=1)
    # Exit long premium when current option IV has fallen this many percent from
    # entry IV. 25 means entry 80% IV exits at/below 60% IV. 0 disables.
    iv_crush_exit_drop_pct: float = Field(default=0.0, ge=0, le=100)
    # Profit-protection trailing exit. Once the best observed bid is at least
    # this many percent above entry, exit if that open profit gives back
    # ``trailing_profit_giveback_pct``. Both 0 disables the rule.
    trailing_profit_activation_pct: float = Field(default=0.0, ge=0)
    trailing_profit_giveback_pct: float = Field(default=0.0, ge=0, le=100)
    # Default exit ladder the committee briefs against when no LLM proposal
    # exists yet (used for the Kelly sizing hint on the candidate block).
    # Keep aligned with the prompt's exit_plan example so the model isn't
    # nudged toward a different TP/SL than the brief assumes. Defaults preserve
    # the historical +100 / -50 baseline.
    default_take_profit_pct: float = Field(default=100.0, gt=0)
    default_stop_loss_pct: float = Field(default=50.0, gt=0, le=100)


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
    # Optional portfolio Greeks caps. 0 disables each cap. The gate evaluates
    # the open book plus the proposed new trade.
    max_portfolio_abs_delta: float = Field(default=0.0, ge=0)
    max_portfolio_gamma: float = Field(default=0.0, ge=0)
    max_portfolio_vega: float = Field(default=0.0, ge=0)
    max_portfolio_theta_decay_usd_per_day: float = Field(default=0.0, ge=0)


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
    # Local committee decision cache TTL. Evidence IDs are still hashed into the
    # key, so fresh filings/news force a re-evaluation; this TTL mostly controls
    # how long unchanged reject/hold decisions avoid repeat LLM burns.
    decision_cache_ttl_hours: float = Field(default=6.0, gt=0)
    # Freeze each ticker's Google News headline set for this long. The RSS feed
    # reshuffles its top-N between requests, so re-fetching every cycle churns the
    # news evidence IDs and busts the thesis-keyed decision cache even when no new
    # story broke -- the dominant cache miss that drained the token budget. Keep
    # this >= rejection_bench_hours so a name re-competing after its bench still
    # hits the thesis cache. SEC filings stay real-time regardless, so a fresh
    # 8-K still forces a re-run. 0 disables (always fetch live).
    news_cache_ttl_hours: float = Field(default=4.0, ge=0)
    # Universe selection can bench recent reject/hold names before fetching more
    # evidence. Keep this shorter than the decision cache if you want names to
    # compete for slots again while still allowing a later thesis-cache hit. 0
    # disables the pre-selection bench entirely.
    rejection_bench_hours: float = Field(default=2.0, ge=0)
    # Decouple cadence: exits/risk are monitored every tick (cheap, no LLM),
    # but new-entry EVALUATION (universe + committee, the token-hungry part)
    # only runs this often. At a 60s tick a value of 300 means the committee
    # fires every ~5 min instead of every minute, cutting token burn ~5x while
    # stops/take-profits still react each tick. 0 evaluates entries every tick.
    entry_evaluation_interval_seconds: int = Field(default=0, ge=0)
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
    # --- market-regime guards (Optional: None = caller could not read it). ---
    # These live in the deterministic gate, not the orchestrator, so the
    # mandate's "no entries above VIX X / in the first N minutes after the
    # open" rules can never be bypassed by a different caller (e.g. the
    # ``proposal-check`` CLI or a future execution path). None means "no
    # reading available" -- the check is skipped rather than silently passing,
    # because a market-data outage should not change whether a panic-regime
    # guard fires; the orchestrator audits the unavailability separately.
    vix: float | None = None
    minutes_since_open: float | None = None
    # Current open-book Greeks from the local ledger, plus the proposed trade's
    # Greeks. Delta is signed share-equivalent; gamma/vega are dollar-greek
    # equivalents after multiplying by contracts*lot_size; theta is USD/day.
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    proposal_delta: float | None = None
    proposal_gamma: float | None = None
    proposal_vega: float | None = None
    proposal_theta: float | None = None
    proposal_abs_delta: float | None = None
    proposal_gamma_per_contract: float | None = None
    proposal_theta_decay_pct_per_day: float | None = None
    proposal_iv_rank: float | None = None


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

        if (
            self.mandate.options.min_abs_delta > 0
            and portfolio.proposal_abs_delta is not None
            and portfolio.proposal_abs_delta < self.mandate.options.min_abs_delta
        ):
            reasons.append("absolute delta is below mandate minimum")
        if (
            self.mandate.options.max_gamma_per_contract > 0
            and portfolio.proposal_gamma_per_contract is not None
            and portfolio.proposal_gamma_per_contract
            > self.mandate.options.max_gamma_per_contract
        ):
            reasons.append("gamma exceeds mandate limit")
        if (
            self.mandate.options.max_theta_decay_pct_per_day > 0
            and portfolio.proposal_theta_decay_pct_per_day is not None
            and portfolio.proposal_theta_decay_pct_per_day
            > self.mandate.options.max_theta_decay_pct_per_day
        ):
            reasons.append("theta decay exceeds mandate limit")
        if (
            self.mandate.options.max_iv_rank_for_long_premium > 0
            and portfolio.proposal_iv_rank is not None
            and portfolio.proposal_iv_rank
            > self.mandate.options.max_iv_rank_for_long_premium
        ):
            reasons.append("IV rank exceeds mandate limit for long premium")

        after_delta = portfolio.total_delta + (portfolio.proposal_delta or 0.0)
        after_gamma = portfolio.total_gamma + (portfolio.proposal_gamma or 0.0)
        after_vega = portfolio.total_vega + (portfolio.proposal_vega or 0.0)
        after_theta_decay = abs(portfolio.total_theta + (portfolio.proposal_theta or 0.0))
        if (
            self.mandate.portfolio.max_portfolio_abs_delta > 0
            and abs(after_delta) > self.mandate.portfolio.max_portfolio_abs_delta
        ):
            reasons.append("portfolio delta exposure would exceed mandate limit")
        if (
            self.mandate.portfolio.max_portfolio_gamma > 0
            and after_gamma > self.mandate.portfolio.max_portfolio_gamma
        ):
            reasons.append("portfolio gamma exposure would exceed mandate limit")
        if (
            self.mandate.portfolio.max_portfolio_vega > 0
            and after_vega > self.mandate.portfolio.max_portfolio_vega
        ):
            reasons.append("portfolio vega exposure would exceed mandate limit")
        if (
            self.mandate.portfolio.max_portfolio_theta_decay_usd_per_day > 0
            and after_theta_decay
            > self.mandate.portfolio.max_portfolio_theta_decay_usd_per_day
        ):
            reasons.append("portfolio theta decay would exceed mandate limit")

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

        # Market-regime guards. These are mandate rules, so they belong in the
        # deterministic gate rather than only in the orchestrator -- any caller
        # that hands the gate a VIX reading or a minutes-since-open value gets
        # the same protection. A None reading (data unavailable) is audited by
        # the caller and skipped here, not treated as a pass or a block.
        vix_cap = self.mandate.portfolio.max_vix_for_entries
        if vix_cap > 0 and portfolio.vix is not None and portfolio.vix > vix_cap:
            reasons.append(
                f"VIX {portfolio.vix:.1f} above the {vix_cap:.0f} entry cap"
            )
        open_window = self.mandate.execution.no_entry_minutes_after_open
        if (
            open_window > 0
            and portfolio.minutes_since_open is not None
            and portfolio.minutes_since_open < open_window
        ):
            reasons.append(
                f"within the first {open_window} minutes after the open"
                " (widest spreads)"
            )

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

