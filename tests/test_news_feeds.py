from datetime import datetime, timezone

import pytest

from trading_agent.data.news_feeds import (
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
    client = GoogleNewsClient(now_fn=lambda: NOW, fetch_fn=lambda ticker: RSS)
    items = client.fetch_evidence("sofi")
    assert len(items) == 2
    first = items[0]
    assert first.ticker == "SOFI"
    assert first.source_type == "news_rss"
    assert first.source_url == "https://example.com/sofi-partnership"
    assert first.observed_fact.startswith("SoFi announces new partnership")
    assert first.evidence_id.startswith("news-")
