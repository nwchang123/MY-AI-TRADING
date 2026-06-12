from datetime import date, datetime, timezone
from types import SimpleNamespace

from trading_agent.data.earnings import YahooEarningsCalendar

NOW = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)


def _client(calendar, now=NOW) -> YahooEarningsCalendar:
    return YahooEarningsCalendar(
        now_fn=lambda: now,
        ticker_factory=lambda symbol: SimpleNamespace(calendar=calendar),
    )


def test_upcoming_earnings_inside_window_yields_evidence() -> None:
    items = _client({"Earnings Date": [date(2026, 6, 25)]}).fetch_evidence("acme")
    assert len(items) == 1
    item = items[0]
    assert item.source_type == "earnings_calendar"
    assert item.ticker == "ACME"
    assert "2026-06-25" in item.observed_fact
    # Stable id keeps the committee decision cache valid across cycles.
    assert item.evidence_id == "earnings-ACME-2026-06-25"


def test_earnings_beyond_window_is_ignored() -> None:
    far = date(2026, 9, 1)  # > 45 days out: cannot crush an allowed position
    assert _client({"Earnings Date": [far]}).fetch_evidence("ACME") == []


def test_past_dates_are_skipped_for_the_next_upcoming() -> None:
    cal = {"Earnings Date": [date(2026, 3, 1), date(2026, 6, 20)]}
    items = _client(cal).fetch_evidence("ACME")
    assert len(items) == 1
    assert "2026-06-20" in items[0].observed_fact


def test_missing_or_broken_calendar_yields_nothing() -> None:
    assert _client({}).fetch_evidence("ACME") == []
    assert _client(None).fetch_evidence("ACME") == []

    def boom(symbol):
        raise RuntimeError("yahoo down")

    client = YahooEarningsCalendar(now_fn=lambda: NOW, ticker_factory=boom)
    assert client.fetch_evidence("ACME") == []


def test_single_datetime_value_is_accepted() -> None:
    # yfinance sometimes returns a bare datetime instead of a list.
    cal = {"Earnings Date": datetime(2026, 6, 18, 21, 0)}
    items = _client(cal).fetch_evidence("ACME")
    assert len(items) == 1
    assert "2026-06-18" in items[0].observed_fact
