from __future__ import annotations

from datetime import datetime, timezone

from trading_agent.domain.calendar import market_date, trading_days_until
from trading_agent.domain.positions import CloseSignal, MonitoredPosition


class PositionMonitor:
    """Deterministic exit engine for open long-option positions.

    Each position yields at most one close signal. Time-based exits (forced
    close before expiry, time stop) always fire, even on a stale quote, so an
    expiry can never be missed. P/L-based exits (take profit, stop loss) require
    a fresh quote.
    """

    def __init__(
        self,
        force_close_before_expiry_trading_days: int,
        stale_quote_seconds: int,
        delayed_quote_max_age_seconds: int | None = None,
    ):
        self.force_close_days = force_close_before_expiry_trading_days
        self.stale_quote_seconds = stale_quote_seconds
        # Falls back to the real-time threshold when not supplied, so existing
        # callers that pass only stale_quote_seconds keep their behavior.
        self.delayed_quote_max_age_seconds = (
            delayed_quote_max_age_seconds
            if delayed_quote_max_age_seconds is not None
            else stale_quote_seconds
        )

    def evaluate(
        self, positions: list[MonitoredPosition], now: datetime | None = None
    ) -> list[CloseSignal]:
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        signals: list[CloseSignal] = []
        for position in positions:
            signal = self._evaluate_one(position, current_time)
            if signal is not None:
                signals.append(signal)
        return signals

    def _evaluate_one(
        self, position: MonitoredPosition, now: datetime
    ) -> CloseSignal | None:
        mark = position.mark_price()
        pnl_pct = round((mark - position.entry_price) / position.entry_price * 100, 4)
        quote_age = (now - position.observed_at).total_seconds()
        max_age = (
            self.delayed_quote_max_age_seconds
            if position.is_delayed
            else self.stale_quote_seconds
        )
        stale = quote_age < 0 or quote_age > max_age

        reason: str | None = None
        if trading_days_until(position.expiry, now) <= self.force_close_days:
            reason = "forced close before expiry"
        elif (
            position.pre_earnings_exit_date is not None
            and market_date(now) >= position.pre_earnings_exit_date
        ):
            # Pre-earnings (IV-ramp) play: get out BEFORE the print. Fires even on
            # a stale quote -- missing the exit means holding through the event
            # IV crush, which is exactly what this strategy exists to avoid.
            reason = "pre-earnings exit"
        elif (
            position.catalyst_window_end is not None
            and market_date(now) > position.catalyst_window_end
        ):
            # The committee's catalyst window has elapsed without a take-profit:
            # by its own reasoning the edge is gone, so stop paying theta to hold.
            reason = "catalyst window elapsed"
        elif market_date(now) >= position.time_stop:
            reason = "time stop reached"
        elif not stale:
            if pnl_pct >= position.take_profit_pct:
                reason = "take profit"
            elif pnl_pct <= -position.stop_loss_pct:
                reason = "stop loss"

        if reason is None:
            return None
        return CloseSignal(
            option_code=position.option_code,
            reason=reason,
            mark_price=round(mark, 4),
            pnl_pct=pnl_pct,
        )
