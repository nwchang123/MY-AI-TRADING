import json
from datetime import date, datetime, timezone

from trading_agent.data.earnings_calendar import (
    FinnhubEarningsCalendar,
    YfinanceEarningsCalendar,
    build_earnings_calendar,
    parse_finnhub_calendar,
)

START = date(2026, 6, 15)
END = date(2026, 6, 30)


def _finnhub_payload() -> dict:
    return {
        "earningsCalendar": [
            {"symbol": "aaa", "date": "2026-06-20"},
            {"symbol": "BBB", "date": "2026-06-25"},
            {"symbol": "AAA", "date": "2026-06-18"},  # earlier dupe -> wins
            {"symbol": "CCC", "date": "2099-01-01"},  # out of window
            {"symbol": "", "date": "2026-06-20"},  # no symbol -> skipped
            {"symbol": "DDD", "date": "not-a-date"},  # bad date -> skipped
        ]
    }


def test_parse_finnhub_keeps_earliest_and_skips_junk() -> None:
    out = parse_finnhub_calendar(_finnhub_payload())
    assert out["AAA"] == date(2026, 6, 18)  # earliest of the two AAA rows
    assert out["BBB"] == date(2026, 6, 25)
    assert "DDD" not in out and "" not in out


def test_finnhub_upcoming_filters_to_wanted_and_window() -> None:
    cal = FinnhubEarningsCalendar(
        "tok", fetch_fn=lambda url: json.dumps(_finnhub_payload()).encode()
    )
    out = cal.upcoming(["AAA", "BBB", "CCC", "ZZZ"], START, END)
    assert out == {"AAA": date(2026, 6, 18), "BBB": date(2026, 6, 25)}
    # CCC is out of window, ZZZ is not in the calendar


def test_finnhub_upcoming_returns_empty_on_fetch_error() -> None:
    def boom(url: str) -> bytes:
        raise RuntimeError("network down")

    cal = FinnhubEarningsCalendar("tok", fetch_fn=boom)
    assert cal.upcoming(["AAA"], START, END) == {}


def test_finnhub_url_carries_token_and_window() -> None:
    seen: dict[str, str] = {}

    def capture(url: str) -> bytes:
        seen["url"] = url
        return b'{"earningsCalendar": []}'

    FinnhubEarningsCalendar("secret-tok", fetch_fn=capture).upcoming(["AAA"], START, END)
    assert "token=secret-tok" in seen["url"]
    assert "from=2026-06-15" in seen["url"] and "to=2026-06-30" in seen["url"]


class _StubYahoo:
    def __init__(self, dates: dict[str, date]):
        self.dates = dates
        self.queried: list[str] = []

    def next_earnings_date(self, ticker: str):
        self.queried.append(ticker)
        return self.dates.get(ticker.upper())


def test_yfinance_upcoming_respects_window_and_shortlist_cap() -> None:
    stub = _StubYahoo(
        {"AAA": date(2026, 6, 20), "BBB": date(2099, 1, 1), "CCC": date(2026, 6, 22)}
    )
    cal = YfinanceEarningsCalendar(shortlist_cap=2, source=stub)
    out = cal.upcoming(["AAA", "BBB", "CCC"], START, END)
    # CCC is past the shortlist cap of 2 -> never queried; BBB out of window
    assert out == {"AAA": date(2026, 6, 20)}
    assert stub.queried == ["AAA", "BBB"]


def test_build_picks_finnhub_with_token_else_yfinance() -> None:
    assert isinstance(build_earnings_calendar("tok"), FinnhubEarningsCalendar)
    assert isinstance(build_earnings_calendar(""), YfinanceEarningsCalendar)


def test_finnhub_next_earnings_date_caches_one_wide_fetch() -> None:
    calls = {"n": 0}

    def fetch(url: str) -> bytes:
        calls["n"] += 1
        return json.dumps(_finnhub_payload()).encode()

    cal = FinnhubEarningsCalendar(
        "tok",
        fetch_fn=fetch,
        now_fn=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
    )
    assert cal.next_earnings_date("AAA") == date(2026, 6, 18)
    assert cal.next_earnings_date("bbb") == date(2026, 6, 25)  # case-insensitive
    assert cal.next_earnings_date("ZZZ") is None  # not in the calendar
    assert cal.next_earnings_date("CCC") is None  # 2099 is past the 90d lookahead
    assert calls["n"] == 1  # one cached wide-window fetch backs every lookup


def test_yfinance_next_earnings_date_delegates() -> None:
    stub = _StubYahoo({"AAA": date(2026, 6, 20)})
    cal = YfinanceEarningsCalendar(source=stub)
    assert cal.next_earnings_date("AAA") == date(2026, 6, 20)
    assert cal.next_earnings_date("ZZZ") is None
