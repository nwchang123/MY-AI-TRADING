from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OptionCandidate(BaseModel):
    """A real, listed option contract that already passes the mandate's
    liquidity/DTE/cost checks. The committee picks one of these by option_code
    instead of guessing a contract blind, so a proposal can only ever name a
    contract that actually exists and is tradeable."""

    model_config = ConfigDict(extra="forbid")

    option_code: str = Field(min_length=1)
    option_side: Literal["call", "put"]
    strike: float = Field(gt=0)
    expiry: date
    bid: float = Field(ge=0)
    ask: float = Field(gt=0)
    open_interest: int = Field(ge=0)
    daily_volume: int = Field(ge=0)
    iv: float = Field(ge=0, default=0.0)
    dte: int
    estimated_contract_cost_usd: float = Field(ge=0)
    # Deterministic Monte Carlo baseline POP at the standard +100/-50 exit
    # grid (no-edge GBM). None when the underlying spot or IV was unavailable.
    mc_pop: float | None = Field(default=None, ge=0, le=1)
    # Black-Scholes delta (signed: puts are negative). How much the option
    # actually tracks the underlying. None when spot/IV were unavailable.
    delta: float | None = Field(default=None, ge=-1, le=1)
    # Black-Scholes gamma per $1 underlying move, vega per 1 IV percentage point,
    # and theta per calendar day. None when spot/IV were unavailable.
    gamma: float | None = Field(default=None, ge=0)
    vega: float | None = Field(default=None, ge=0)
    theta: float | None = Field(default=None)
    theta_decay_pct_per_day: float | None = Field(default=None, ge=0)
    iv_rank: float | None = Field(default=None, ge=0, le=1)
    # Signed % move in the underlying needed to break even at expiry (positive =
    # up, negative = down). Surfaces how far OTM a contract is. None w/o spot.
    breakeven_move_pct: float | None = Field(default=None)


class ExitPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    take_profit_pct: float = Field(gt=0)
    stop_loss_pct: float = Field(gt=0, le=100)
    time_stop: date

    @field_validator("take_profit_pct", "stop_loss_pct", mode="before")
    @classmethod
    def normalize_pct(cls, v: object) -> object:
        """The committee LLM is inconsistent about how it expresses exit
        percentages: sometimes a signed fraction (-0.5 = "stop at -50%"),
        sometimes a magnitude (50.0). Both mean the same trade, but the schema
        wants a positive magnitude in [0, 100], so a -0.5 used to fail validation
        and kill an otherwise-valid proposal. Coerce to that convention: drop the
        sign, and scale a sub-unit fraction up to a percentage."""
        if isinstance(v, (int, float)):
            v = abs(float(v))
            if 0 < v <= 1:
                v *= 100.0
        return v


class OpenPositionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["open_position"]
    ticker: str = Field(min_length=1)
    option_code: str = Field(min_length=1)
    option_side: Literal["call", "put"]
    action: Literal["buy_to_open"]
    contracts: int = Field(gt=0)
    limit_price: float = Field(gt=0)
    max_limit_price: float = Field(gt=0)
    thesis: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    expected_catalyst_window: str

    @field_validator("expected_catalyst_window")
    @classmethod
    def validate_catalyst_window(cls, v: str) -> str:
        parts = v.split("/")
        if len(parts) != 2:
            raise ValueError("expected_catalyst_window must be 'YYYY-MM-DD/YYYY-MM-DD'")
        from datetime import datetime
        for part in parts:
            datetime.strptime(part, "%Y-%m-%d")
        return v

    exit_plan: ExitPlan
    invalidation: list[str] = Field(min_length=1)


class SpreadLeg(BaseModel):
    """One leg of a multi-leg spread."""

    model_config = ConfigDict(extra="forbid")

    option_code: str = Field(min_length=1)
    option_side: Literal["call", "put"]
    action: Literal["buy_to_open", "sell_to_open"]
    contracts: int = Field(gt=0)
    limit_price: float = Field(ge=0)


class SpreadProposal(BaseModel):
    """Multi-leg spread proposal (vertical, iron condor, straddle, etc.)."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["open_spread"]
    ticker: str = Field(min_length=1)
    strategy: str = Field(
        min_length=1,
        description="e.g. 'bull_call_spread', 'iron_condor', 'straddle'",
    )
    legs: list[SpreadLeg] = Field(min_length=2, max_length=4)
    net_debit: float = Field(ge=0, description="Max net debit in USD")
    max_profit: float = Field(ge=0, description="Max profit in USD")
    max_loss: float = Field(ge=0, description="Max loss in USD")
    thesis: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    expected_catalyst_window: str
    exit_plan: ExitPlan
    invalidation: list[str] = Field(min_length=1)

    @field_validator("expected_catalyst_window")
    @classmethod
    def validate_catalyst_window(cls, v: str) -> str:
        parts = v.split("/")
        if len(parts) != 2:
            raise ValueError("expected_catalyst_window must be 'YYYY-MM-DD/YYYY-MM-DD'")
        from datetime import datetime
        for part in parts:
            datetime.strptime(part, "%Y-%m-%d")
        return v

