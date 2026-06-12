"""Forward earnings dates via yfinance (free, best-effort).

Fills the gap the 8-K item 2.02 path cannot: that 8-K lands AFTER the print,
while a long option bought today can sit through an earnings date inside its
14-45 DTE holding window and eat the IV crush. This client emits an
``earnings_calendar`` EvidenceItem for an upcoming date, which the existing
``earnings_iv_crush`` red flag and the catalyst scorer already consume -- no
downstream changes needed.

Strictly best-effort: yfinance's calendar is unofficial and patchy for small
caps, so any failure or absence yields no evidence rather than an error. A
missing warning must never block the cycle.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from trading_agent.domain.evidence import EvidenceItem

# Matches the mandate's max DTE: an earnings date farther out than the longest
# possible holding period cannot crush a position we are allowed to open.
DEFAULT_WINDOW_DAYS = 45


class YahooEarningsCalendar:
    """Upcoming-earnings evidence source backed by yfinance's calendar."""

    def __init__(
        self,
        *,
        window_days: int = DEFAULT_WINDOW_DAYS,
        now_fn: Callable[[], datetime] | None = None,
        ticker_factory: Callable[[str], Any] | None = None,
    ):
        self.window_days = window_days
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._ticker_factory = ticker_factory or self._yfinance_ticker

    @staticmethod
    def _yfinance_ticker(symbol: str) -> Any:
        import yfinance as yf

        return yf.Ticker(symbol)

    def next_earnings_date(self, ticker: str) -> date | None:
        """The next scheduled earnings date, or None when unknown/unavailable."""

        try:
            calendar = self._ticker_factory(ticker.upper()).calendar or {}
            raw = calendar.get("Earnings Date") or []
        except Exception:  # noqa: BLE001 - unofficial feed: absence, not error
            return None
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        today = self._now_fn().date()
        upcoming = sorted(
            d.date() if isinstance(d, datetime) else d
            for d in raw
            if isinstance(d, (date, datetime))
        )
        for day in upcoming:
            if day >= today:
                return day
        return None

    def fetch_evidence(self, ticker: str) -> list[EvidenceItem]:
        """Zero or one earnings_calendar item for an in-window upcoming date.

        The evidence_id is stable per (ticker, date) so the committee decision
        cache stays valid across cycles while the date is unchanged.
        """

        day = self.next_earnings_date(ticker)
        if day is None:
            return []
        now = self._now_fn()
        if day > now.date() + timedelta(days=self.window_days):
            return []
        symbol = ticker.upper()
        return [
            EvidenceItem(
                evidence_id=f"earnings-{symbol}-{day.isoformat()}",
                ticker=symbol,
                source_type="earnings_calendar",
                source_url=f"https://finance.yahoo.com/quote/{symbol}",
                published_at=now,
                observed_fact=(
                    f"Earnings scheduled on {day.isoformat()} -- inside the option "
                    "holding window; a long option held through the print risks IV "
                    "crush even when the direction is right."
                ),
                retrieved_at=now,
            )
        ]
