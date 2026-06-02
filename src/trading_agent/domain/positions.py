from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Re-exported so callers keep importing trading_days_until from this module.
from trading_agent.domain.calendar import trading_days_until  # noqa: F401


class MonitoredPosition(BaseModel):
    """An open long-option position plus its exit plan and latest quote."""

    model_config = ConfigDict(extra="forbid")

    option_code: str = Field(min_length=1)
    option_side: Literal["call", "put"]
    entry_price: float = Field(gt=0)
    contracts: int = Field(gt=0)
    lot_size: int = Field(gt=0)
    expiry: date
    take_profit_pct: float = Field(gt=0)
    stop_loss_pct: float = Field(gt=0, le=100)
    time_stop: date
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    observed_at: datetime

    def mark_price(self) -> float:
        if self.bid > 0 and self.ask > self.bid:
            return (self.bid + self.ask) / 2
        return max(self.bid, 0.0)


class CloseSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_code: str
    reason: str
    mark_price: float
    pnl_pct: float
