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
    # The committee's own estimate of when the catalyst resolves. Once it has
    # passed without hitting take-profit, the edge the thesis rested on is gone,
    # so the position exits rather than bleeding theta to the time stop. None
    # keeps the legacy behavior (time stop only).
    catalyst_window_end: date | None = None
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    observed_at: datetime
    is_delayed: bool = False

    def mark_price(self) -> float:
        # Every monitored position is a long option (mandate allows buy_to_open
        # only), so it is closed by SELLING, which fills at the bid. Marking at
        # the bid -- not the mid -- keeps exit triggers, the sell limit price, and
        # recorded P/L honest about the full spread you actually pay to get out.
        # A non-positive bid means there is no exit liquidity: mark it worthless.
        return max(self.bid, 0.0)


class CloseSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_code: str
    reason: str
    mark_price: float
    pnl_pct: float
