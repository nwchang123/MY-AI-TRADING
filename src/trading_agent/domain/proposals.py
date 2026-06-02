from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    expected_catalyst_window: str = Field(min_length=1)
    exit_plan: ExitPlan
    invalidation: list[str] = Field(min_length=1)

