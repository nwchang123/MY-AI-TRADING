from datetime import date, datetime, time, timezone

from trading_agent.domain.calendar import (
    is_market_hours,
    is_trading_day,
    market_close_time,
    market_date,
    minutes_since_open,
    parse_iso,
    subtract_trading_days,
    trading_days_until,
    us_early_close_dates,
    us_market_holidays,
)


def test_subtract_trading_days_skips_weekend() -> None:
    # 2 trading days before Tue 2026-06-16 is Fri 2026-06-12 (skips the weekend).
    assert subtract_trading_days(date(2026, 6, 16), 2) == date(2026, 6, 12)
    # n<=0 is a no-op.
    assert subtract_trading_days(date(2026, 6, 16), 0) == date(2026, 6, 16)


def test_subtract_trading_days_skips_holiday() -> None:
    # 1 trading day before Mon 2026-07-06 skips the weekend AND Fri 2026-07-03
    # (observed Independence Day holiday) to Thu 2026-07-02.
    assert subtract_trading_days(date(2026, 7, 6), 1) == date(2026, 7, 2)


def test_minutes_since_open_inside_session() -> None:
    # 2026-06-02 is a Tuesday; 13:35 UTC = 9:35 ET during daylight time.
    now = datetime(2026, 6, 2, 13, 35, tzinfo=timezone.utc)
    assert minutes_since_open(now) == 5.0


def test_minutes_since_open_outside_session_is_none() -> None:
    pre_open = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)  # 8:00 ET
    saturday = datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc)
    assert minutes_since_open(pre_open) is None
    assert minutes_since_open(saturday) is None


def test_known_2026_holidays() -> None:
    holidays = us_market_holidays(2026)
    assert date(2026, 1, 1) in holidays  # New Year's Day
    assert date(2026, 4, 3) in holidays  # Good Friday 2026
    assert date(2026, 5, 25) in holidays  # Memorial Day (last Mon May)
    assert date(2026, 6, 19) in holidays  # Juneteenth
    assert date(2026, 7, 3) in holidays  # Independence Day observed (Jul 4 is Sat)
    assert date(2026, 11, 26) in holidays  # Thanksgiving (4th Thu Nov)
    assert date(2026, 12, 25) in holidays  # Christmas


def test_weekend_and_holiday_are_not_trading_days() -> None:
    assert is_trading_day(date(2026, 6, 6)) is False  # Saturday
    assert is_trading_day(date(2026, 6, 19)) is False  # Juneteenth (Friday)
    assert is_trading_day(date(2026, 6, 18)) is True  # Thursday


def test_trading_days_until_skips_holiday() -> None:
    # Wed 2026-06-17 -> Mon 2026-06-22, with Fri 06-19 Juneteenth excluded:
    # Thu 18, Mon 22 = 2 trading days (Fri 19 holiday, Sat/Sun weekend).
    assert trading_days_until(date(2026, 6, 22), date(2026, 6, 17)) == 2


def test_market_date_uses_eastern() -> None:
    # 02:00 UTC on 2026-06-02 is still 2026-06-01 in New York.
    moment = datetime(2026, 6, 2, 2, 0, tzinfo=timezone.utc)
    assert market_date(moment) == date(2026, 6, 1)


def test_early_close_dates_2026() -> None:
    half_days = us_early_close_dates(2026)
    assert date(2026, 11, 27) in half_days  # day after Thanksgiving
    assert date(2026, 12, 24) in half_days  # Christmas Eve (Thursday)
    # July 3 2026 is the OBSERVED July-4 holiday (full close), not a half day.
    assert date(2026, 7, 3) not in half_days


def test_early_close_july3_when_it_trades() -> None:
    # 2025: July 4 is a Friday holiday; July 3 (Thursday) trades and closes early.
    assert date(2025, 7, 3) in us_early_close_dates(2025)
    assert market_close_time(date(2025, 7, 3)) == time(13, 0)
    assert market_close_time(date(2025, 7, 2)) == time(16, 0)


def test_is_market_hours_honors_early_close() -> None:
    # Fri 2026-11-27 (half day): 12:30 ET open, 14:00 ET closed.
    open_moment = datetime(2026, 11, 27, 17, 30, tzinfo=timezone.utc)  # 12:30 ET (EST)
    closed_moment = datetime(2026, 11, 27, 19, 0, tzinfo=timezone.utc)  # 14:00 ET
    assert is_market_hours(open_moment) is True
    assert is_market_hours(closed_moment) is False
    # A normal Friday at 14:00 ET is open.
    normal = datetime(2026, 11, 20, 19, 0, tzinfo=timezone.utc)
    assert is_market_hours(normal) is True


# --- parse_iso ---


def test_parse_iso_accepts_trailing_z() -> None:
    parsed = parse_iso("2026-06-19T13:45:00Z")
    assert parsed == datetime(2026, 6, 19, 13, 45, tzinfo=timezone.utc)


def test_parse_iso_accepts_offset() -> None:
    parsed = parse_iso("2026-06-19T13:45:00+00:00")
    assert parsed == datetime(2026, 6, 19, 13, 45, tzinfo=timezone.utc)


def test_parse_iso_passthrough_datetime() -> None:
    dt = datetime(2026, 6, 19, 13, 45, tzinfo=timezone.utc)
    assert parse_iso(dt) is dt


def test_parse_iso_strips_whitespace() -> None:
    parsed = parse_iso("  2026-06-19T13:45:00Z  ")
    assert parsed == datetime(2026, 6, 19, 13, 45, tzinfo=timezone.utc)
