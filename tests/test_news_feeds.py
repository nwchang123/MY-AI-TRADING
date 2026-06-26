from datetime import datetime, timedelta, timezone

import pytest

from trading_agent.data.news_feeds import (
    CachingNewsClient,
    GoogleNewsClient,
    NewsFeedError,
    parse_google_news_rss,
)

NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>"SOFI" stock - Google News</title>
  <item>
    <title>SoFi announces new partnership with major bank - Reuters</title>
    <link>https://example.com/sofi-partnership</link>
    <pubDate>Wed, 10 Jun 2026 14:30:00 GMT</pubDate>
  </item>
  <item>
    <title>Old story that should be filtered out</title>
    <link>https://example.com/sofi-old</link>
    <pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Missing link is dropped</title>
    <pubDate>Wed, 10 Jun 2026 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Second fresh story - Bloomberg</title>
    <link>https://example.com/sofi-second</link>
    <pubDate>Tue, 09 Jun 2026 09:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""


def test_parse_keeps_fresh_complete_items_only() -> None:
    rows = parse_google_news_rss(RSS, now=NOW)
    assert [r["url"] for r in rows] == [
        "https://example.com/sofi-partnership",
        "https://example.com/sofi-second",
    ]
    assert rows[0]["title"].startswith("SoFi announces")
    assert rows[0]["published_at"] == "2026-06-10T14:30:00+00:00"


def test_parse_respects_max_items() -> None:
    rows = parse_google_news_rss(RSS, now=NOW, max_items=1)
    assert len(rows) == 1


def test_parse_raises_on_garbage() -> None:
    with pytest.raises(NewsFeedError):
        parse_google_news_rss("not xml at all", now=NOW)


def test_client_normalizes_to_evidence_items() -> None:
    client = GoogleNewsClient(now_fn=lambda: NOW, fetch_fn=lambda query: RSS)
    items = client.fetch_evidence("sofi")
    assert len(items) == 2
    first = items[0]
    assert first.ticker == "SOFI"
    assert first.source_type == "news_rss"
    assert first.source_url == "https://example.com/sofi-partnership"
    assert first.observed_fact.startswith("SoFi announces new partnership")
    assert first.evidence_id.startswith("news-")


def test_company_name_anchors_the_query_but_tags_the_ticker() -> None:
    seen: list[str] = []

    def fake_fetch(query: str) -> str:
        seen.append(query)
        return RSS

    client = GoogleNewsClient(now_fn=lambda: NOW, fetch_fn=fake_fetch)
    items = client.fetch_evidence("TE", query_name="Tradeweb Markets Inc")

    # The full company name anchors the search, not the ambiguous "TE"...
    assert seen == ['"Tradeweb Markets Inc" stock']
    # ...but the evidence rows stay tagged with the ticker.
    assert items[0].ticker == "TE"


def test_bare_ticker_query_when_no_name_supplied() -> None:
    seen: list[str] = []
    client = GoogleNewsClient(
        now_fn=lambda: NOW, fetch_fn=lambda q: seen.append(q) or RSS
    )
    client.fetch_evidence("SOFI")
    assert seen == ['"SOFI" stock']


class _StubNews:
    """Inner news client whose result/behavior is driven by ``fn(call_no)``."""

    def __init__(self, fn) -> None:
        self._fn = fn
        self.calls: list[tuple[str, str | None]] = []

    def fetch_evidence(self, ticker: str, query_name: str | None = None):
        self.calls.append((ticker, query_name))
        return self._fn(len(self.calls))


def test_caching_freezes_headline_set_within_ttl() -> None:
    # The core fix: a re-fetch inside the window reuses the first result, so the
    # evidence-id set (and thus the thesis hash) stays stable across cycles even
    # though the live feed would have returned a different list on the 2nd call.
    clock = {"t": NOW}
    stub = _StubNews(lambda n: [f"call-{n}"])
    client = CachingNewsClient(stub, ttl_hours=4.0, now_fn=lambda: clock["t"])

    first = client.fetch_evidence("SOFI")
    clock["t"] = NOW + timedelta(hours=3)
    second = client.fetch_evidence("SOFI")

    assert first == ["call-1"]
    assert second == ["call-1"]  # frozen, not the live "call-2"
    assert len(stub.calls) == 1  # the feed was hit once, not per cycle


def test_caching_refetches_after_ttl_expiry() -> None:
    clock = {"t": NOW}
    stub = _StubNews(lambda n: [f"call-{n}"])
    client = CachingNewsClient(stub, ttl_hours=4.0, now_fn=lambda: clock["t"])

    client.fetch_evidence("SOFI")
    clock["t"] = NOW + timedelta(hours=4, minutes=1)
    refreshed = client.fetch_evidence("SOFI")

    assert refreshed == ["call-2"]  # genuinely fresh news still gets through
    assert len(stub.calls) == 2


def test_caching_keys_on_ticker_and_query_name() -> None:
    clock = {"t": NOW}
    stub = _StubNews(lambda n: [f"call-{n}"])
    client = CachingNewsClient(stub, ttl_hours=4.0, now_fn=lambda: clock["t"])

    client.fetch_evidence("TE", query_name="Tradeweb Markets Inc")
    client.fetch_evidence("TE", query_name="Other Name")
    client.fetch_evidence("SOFI")

    assert len(stub.calls) == 3  # each distinct (ticker, query_name) is its own slot


def test_caching_serves_stale_on_feed_failure() -> None:
    clock = {"t": NOW}

    def fn(n: int):
        if n == 1:
            return ["fresh"]
        raise NewsFeedError("feed down")

    stub = _StubNews(fn)
    client = CachingNewsClient(stub, ttl_hours=4.0, now_fn=lambda: clock["t"])

    client.fetch_evidence("SOFI")
    clock["t"] = NOW + timedelta(hours=5)  # past TTL, forces a re-fetch that fails
    served = client.fetch_evidence("SOFI")

    assert served == ["fresh"]  # stale beats an empty set that would churn the thesis


def test_caching_propagates_failure_with_no_prior_entry() -> None:
    def fn(n: int):
        raise NewsFeedError("feed down")

    client = CachingNewsClient(_StubNews(fn), ttl_hours=4.0, now_fn=lambda: NOW)

    with pytest.raises(NewsFeedError):
        client.fetch_evidence("SOFI")
