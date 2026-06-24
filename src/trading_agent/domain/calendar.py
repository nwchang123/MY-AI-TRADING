from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache

# Regular U.S. equity/option session in Eastern time.
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
# NYSE half days (July 3, day after Thanksgiving, Christmas Eve) close at 13:00.
EARLY_CLOSE = time(13, 0)

try:
    from zoneinfo import ZoneInfo

    MARKET_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo/tzdata unavailable
    MARKET_TZ = None  # type: ignore[assignment]


def market_date(moment: date | datetime) -> date:
    """The U.S. market (Eastern) calendar date for a moment.

    Options expire at the U.S. close, so trading-day math must use the Eastern
    date, not the UTC date, to avoid an off-by-one near midnight UTC.
    """

    if isinstance(moment, datetime):
        if moment.tzinfo is not None and MARKET_TZ is not None:
            return moment.astimezone(MARKET_TZ).date()
        return moment.date()
    return moment


def _easter(year: int) -> date:
    # Anonymous Gregorian algorithm (Meeus/Jones/Butcher).
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    le = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * le) // 451
    month = (h + le - 7 * m + 114) // 31
    day = ((h + le - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(holiday: date) -> date:
    # NYSE: Saturday holidays observed Friday, Sunday holidays observed Monday.
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


@lru_cache(maxsize=32)
def us_market_holidays(year: int) -> frozenset[date]:
    """NYSE/Nasdaq full-day market holidays for a year (observed dates)."""

    holidays = {
        _observed(date(year, 1, 1)),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # MLK Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday (3rd Mon Feb)
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day (last Mon May)
        _nth_weekday(year, 9, 0, 1),  # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),  # Christmas
        _observed(date(year, 7, 4)),  # Independence Day
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))  # Juneteenth
    return frozenset(holidays)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in us_market_holidays(day.year)


def parse_iso(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z``.

    ``datetime.fromisoformat`` (pre-3.11) rejects ``Z``; the position store
    writes ``+00:00`` but external/broker payloads often use ``Z``. This
    normalizes both so callers don't sprinkle ``.replace("Z", "+00:00")``
    across the codebase. Accepts an already-parsed ``datetime`` for ergonomic
    pass-through.
    """

    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


@lru_cache(maxsize=32)
def us_early_close_dates(year: int) -> frozenset[date]:
    """NYSE/Nasdaq 13:00 ET half days (observed dates).

    July 3 and Christmas Eve count only when they are themselves trading days
    (e.g. 2026's July 3 is the observed Independence Day holiday, not a half
    day); the day after Thanksgiving is always a trading Friday.
    """

    days: set[date] = set()
    july3 = date(year, 7, 3)
    if is_trading_day(july3):
        days.add(july3)
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    days.add(thanksgiving + timedelta(days=1))
    dec24 = date(year, 12, 24)
    if is_trading_day(dec24):
        days.add(dec24)
    return frozenset(days)


def market_close_time(day: date) -> time:
    """The session close for a date: 13:00 ET on half days, 16:00 otherwise."""

    return EARLY_CLOSE if day in us_early_close_dates(day.year) else MARKET_CLOSE


def is_market_hours(now: datetime) -> bool:
    """True if ``now`` is within the U.S. session on a trading day, honoring
    13:00 ET early closes. Requires a timezone-aware datetime.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    eastern = now.astimezone(MARKET_TZ) if MARKET_TZ is not None else now
    if not is_trading_day(eastern.date()):
        return False
    return MARKET_OPEN <= eastern.time() <= market_close_time(eastern.date())


def minutes_since_open(now: datetime) -> float | None:
    """Minutes elapsed since the 9:30 ET open, or None outside the session.

    Used to keep new entries out of the opening auction window, where option
    spreads are at their widest.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not is_market_hours(now):
        return None
    eastern = now.astimezone(MARKET_TZ) if MARKET_TZ is not None else now
    open_moment = eastern.replace(
        hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0
    )
    return (eastern - open_moment).total_seconds() / 60.0


def subtract_trading_days(day: date, n: int) -> date:
    """The date ``n`` TRADING days before ``day`` (weekends/holidays skipped).

    Used to schedule a pre-earnings exit ``n`` trading days ahead of the print so
    a position never holds through the report. ``n <= 0`` returns ``day``.
    """

    if n <= 0:
        return day
    cursor = day
    remaining = n
    while remaining > 0:
        cursor -= timedelta(days=1)
        if is_trading_day(cursor):
            remaining -= 1
    return cursor


def trading_days_until(expiry: date, now: date | datetime) -> int:
    """Trading days from the day after ``now`` through ``expiry`` inclusive.

    Excludes weekends and U.S. market holidays. Returns 0 once expiry is today
    or past, so an expired contract always trips a forced close.
    """

    today = market_date(now)
    if expiry <= today:
        return 0
    count = 0
    cursor = today
    while cursor < expiry:
        cursor += timedelta(days=1)
        if is_trading_day(cursor):
            count += 1
    return count
