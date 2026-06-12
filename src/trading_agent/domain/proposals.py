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
    # Signed % move in the underlying needed to break even at expiry (positive =
    # up, negative = down). Surfaces how far OTM a contract is. None w/o spot.
    breakeven_move_pct: float | None = Field(default=None)


class ExitPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    take_profit_pct: float = Field(gt=0)
    stop_loss_pct: float = Field(gt=0, le=100)
    time_stop: date


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

