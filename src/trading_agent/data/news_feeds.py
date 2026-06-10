from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

from trading_agent.data.news import news_items_from_raw
from trading_agent.domain.evidence import EvidenceItem

_USER_AGENT = "moomoo-small-cap-options-agent/0.1 (+news)"
_GOOGLE_NEWS_URL = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

# Headlines older than this cannot be a fresh catalyst (the scorer's catalyst
# horizon is 30 days; anything past half of that is already heavily decayed).
DEFAULT_MAX_AGE_DAYS = 14
DEFAULT_MAX_ITEMS = 10


class NewsFeedError(RuntimeError):
    """Raised when a news feed is unreachable or returns an unusable payload."""


def parse_google_news_rss(
    xml_text: str,
    *,
    now: datetime,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> list[dict[str, Any]]:
    """Normalize a Google News RSS document into raw news rows.

    Pure (no network) so it is testable from a fixture. Rows older than
    ``max_age_days`` or missing a link/title/date are dropped. Returns at most
    ``max_items`` rows shaped for :func:`news_items_from_raw`:
    ``{"url", "title", "published_at" (ISO)}``.
    """

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise NewsFeedError(f"unparseable RSS payload: {exc}") from exc

    cutoff = now - timedelta(days=max_age_days)
    rows: list[dict[str, Any]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if not title or not link or not pub_date:
            continue
        try:
            published = parsedate_to_datetime(pub_date)
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published < cutoff or published > now + timedelta(hours=1):
            continue
        rows.append(
            {
                "url": link,
                "title": title,
                "published_at": published.isoformat(),
                "source_type": "news_rss",
            }
        )
        if len(rows) >= max_items:
            break
    return rows


class GoogleNewsClient:
    """Free per-ticker headline search via Google News RSS (no API key).

    Emits the same :class:`EvidenceItem` rows as the SEC pipeline, so headlines
    flow through dedupe, deterministic scoring, and the committee briefing
    unchanged. Headlines are titles only (no article body): enough for catalyst
    discovery, and every row keeps its source URL so theses stay traceable.
    """

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_items: int = DEFAULT_MAX_ITEMS,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
        now_fn: Callable[[], datetime] | None = None,
        fetch_fn: Callable[[str], str] | None = None,
    ):
        self.timeout = timeout
        self.max_items = max_items
        self.max_age_days = max_age_days
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._fetch = fetch_fn or self._http_fetch

    def _http_fetch(self, ticker: str) -> str:
        query = urllib.parse.quote(f'"{ticker}" stock')
        url = _GOOGLE_NEWS_URL.format(query=query)
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise NewsFeedError(f"news fetch failed for {ticker}: {exc}") from exc

    def fetch_evidence(self, ticker: str) -> list[EvidenceItem]:
        now = self._now_fn()
        rows = parse_google_news_rss(
            self._fetch(ticker),
            now=now,
            max_items=self.max_items,
            max_age_days=self.max_age_days,
        )
        return news_items_from_raw(
            rows, ticker, default_source="news_rss", retrieved_at=now
        )
