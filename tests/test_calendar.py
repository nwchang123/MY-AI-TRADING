from datetime import date, datetime, timezone

from trading_agent.domain.calendar import (
    is_trading_day,
    market_date,
    trading_days_until,
    us_market_holidays,
)


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
