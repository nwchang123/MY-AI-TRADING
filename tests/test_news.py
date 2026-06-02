from datetime import datetime, timezone

from trading_agent.data.news import news_items_from_raw

RETRIEVED = datetime(2026, 6, 2, 13, 0, tzinfo=timezone.utc)


def test_news_items_from_raw_maps_fields() -> None:
    raw = [
        {
            "url": "https://news.example.com/a",
            "title": "Company wins large supply contract",
            "published_at": "2026-06-01T12:00:00Z",
        }
    ]
    items = news_items_from_raw(raw, "example", retrieved_at=RETRIEVED)
    assert len(items) == 1
    item = items[0]
    assert item.ticker == "EXAMPLE"
    assert item.source_type == "moomoo_news"
    assert item.source_url == "https://news.example.com/a"
    assert item.observed_fact == "Company wins large supply contract"
    assert item.published_at == datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    assert item.evidence_id.startswith("news-")


def test_news_evidence_id_is_stable_for_same_url() -> None:
    raw = [{"url": "https://x.com/1", "title": "t", "published_at": "2026-06-01T00:00:00Z"}]
    a = news_items_from_raw(raw, "X", retrieved_at=RETRIEVED)[0]
    b = news_items_from_raw(raw, "X", retrieved_at=RETRIEVED)[0]
    assert a.evidence_id == b.evidence_id


def test_news_respects_explicit_source_type() -> None:
    raw = [
        {
            "url": "https://ir.example.com/pr",
            "title": "Press release",
            "published_at": "2026-06-01T00:00:00Z",
            "source_type": "press_release",
        }
    ]
    item = news_items_from_raw(raw, "X", retrieved_at=RETRIEVED)[0]
    assert item.source_type == "press_release"
