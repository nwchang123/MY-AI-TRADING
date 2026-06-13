"""Bulk upcoming-earnings calendar for pre-catalyst (IV-ramp) universe selection.

``data/earnings.py`` answers "when does ONE ticker next report" (per-ticker,
consumed by the earnings_iv_crush red flag). This module answers the universe
scanner's question -- "which of these hundreds of scanned names report between
two dates" -- in one bulk call, so selection can TARGET pre-earnings names
(buy before the IV ramp) instead of chasing post-spike, already-high-IV names.

Finnhub's ``/calendar/earnings`` is the bulk source (free key). When no key is
set, a yfinance per-ticker fallback covers a capped shortlist so the cycle never
stalls on hundreds of lookups. Every failure degrades to an empty result, never
an exception -- a missing calendar must not block trading.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Protocol, runtime_checkable

from trading_agent.data.earnings import YahooEarningsCalendar
from trading_agent.data.http_utils import fetch_with_retry

_FINNHUB_URL = (
    "https://finnhub.io/api/v1/calendar/earnings?from={start}&to={end}&token={token}"
)

# Per-ticker yfinance lookups are slow and rate-limited, so the fallback only
# probes this many highest-priority (already-ranked) names; deeper names are
# simply left unchecked rather than stalling the cycle.
DEFAULT_SHORTLIST_CAP = 60

# How far ahead ``next_earnings_date`` looks (the orchestrator uses it to align
# the option DTE and the pre-earnings exit with the SAME source selection used).
DEFAULT_LOOKAHEAD_DAYS = 90


@runtime_checkable
class EarningsCalendar(Protocol):
    """The bulk-calendar surface the universe scanner + orchestrator depend on."""

    def upcoming(
        self, tickers: list[str], start: date, end: date
    ) -> dict[str, date]: ...

    def next_earnings_date(self, ticker: str) -> date | None: ...


def parse_finnhub_calendar(payload: dict[str, Any]) -> dict[str, date]:
    """Map a Finnhub ``/calendar/earnings`` payload to ``{TICKER: earliest date}``.

    Pure (no network) so it is unit-testable from a fixture. Unparseable rows are
    skipped; when a symbol appears twice the earliest date wins.
    """

    out: dict[str, date] = {}
    for row in payload.get("earningsCalendar") or []:
        symbol = str(row.get("symbol") or "").strip().upper()
        raw = row.get("date")
        if not symbol or not raw:
            continue
        try:
            day = date.fromisoformat(str(raw))
        except ValueError:
            continue
        if symbol not in out or day < out[symbol]:
            out[symbol] = day
    return out


class FinnhubEarningsCalendar:
    """Bulk earnings calendar from Finnhub (one request per date window)."""

    def __init__(
        self,
        token: str,
        *,
        timeout: float = 20.0,
        lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
        now_fn: Callable[[], datetime] | None = None,
        fetch_fn: Callable[[str], bytes] | None = None,
    ):
        if not token:
            raise ValueError("FinnhubEarningsCalendar requires an API token")
        self.token = token
        self.timeout = timeout
        self.lookahead_days = lookahead_days
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._fetch = fetch_fn or self._http_fetch
        self._wide: dict[str, date] | None = None  # cached next_earnings_date window

    def _http_fetch(self, url: str) -> bytes:
        return fetch_with_retry(
            url, timeout=self.timeout, max_retries=2, retry_base_delay=1.0
        )

    def _calendar(self, start: date, end: date) -> dict[str, date]:
        """Full {TICKER: date} for the window, or {} on any failure."""

        url = _FINNHUB_URL.format(
            start=start.isoformat(), end=end.isoformat(), token=self.token
        )
        try:
            payload = json.loads(self._fetch(url))
        except Exception:  # noqa: BLE001 - a missing calendar never blocks a cycle
            return {}
        return {
            t: d
            for t, d in parse_finnhub_calendar(payload).items()
            if start <= d <= end
        }

    def upcoming(
        self, tickers: list[str], start: date, end: date
    ) -> dict[str, date]:
        calendar = self._calendar(start, end)
        wanted = {t.strip().upper() for t in tickers}
        return {t: d for t, d in calendar.items() if t in wanted}

    def next_earnings_date(self, ticker: str) -> date | None:
        # One cached wide-window fetch backs every per-ticker lookup in a cycle,
        # so the orchestrator aligns DTE + the pre-earnings exit with the SAME
        # source the universe scanner used -- without a request per ticker.
        if self._wide is None:
            today = self._now_fn().date()
            self._wide = self._calendar(
                today, today + timedelta(days=self.lookahead_days)
            )
        return self._wide.get(ticker.strip().upper())


class YfinanceEarningsCalendar:
    """Per-ticker fallback over a capped shortlist (no API key required)."""

    def __init__(
        self,
        *,
        shortlist_cap: int = DEFAULT_SHORTLIST_CAP,
        source: YahooEarningsCalendar | None = None,
    ):
        self.shortlist_cap = shortlist_cap
        self.source = source or YahooEarningsCalendar()

    def upcoming(
        self, tickers: list[str], start: date, end: date
    ) -> dict[str, date]:
        out: dict[str, date] = {}
        for ticker in tickers[: self.shortlist_cap]:
            day = self.next_earnings_date(ticker)
            if day is not None and start <= day <= end:
                out[ticker.strip().upper()] = day
        return out

    def next_earnings_date(self, ticker: str) -> date | None:
        try:
            return self.source.next_earnings_date(ticker)
        except Exception:  # noqa: BLE001 - unofficial feed: absence, not error
            return None


def build_earnings_calendar(
    finnhub_token: str = "", *, shortlist_cap: int = DEFAULT_SHORTLIST_CAP
) -> EarningsCalendar:
    """Finnhub bulk calendar when a token is set, else the yfinance shortlist."""

    if finnhub_token:
        return FinnhubEarningsCalendar(finnhub_token)
    return YfinanceEarningsCalendar(shortlist_cap=shortlist_cap)
