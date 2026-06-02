from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from trading_agent.domain.evidence import EvidenceItem, SourceType


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def news_items_from_raw(
    raw_items: list[dict[str, Any]],
    ticker: str,
    *,
    default_source: SourceType = "moomoo_news",
    retrieved_at: datetime | None = None,
) -> list[EvidenceItem]:
    """Normalize raw news rows into traceable evidence.

    Each raw row needs at least ``url``, ``title``, and ``published_at`` (ISO).
    Source-agnostic so any news provider can feed the same pipeline; every row
    keeps its source URL so theses stay traceable.
    """

    retrieved = retrieved_at or datetime.now(timezone.utc)
    items: list[EvidenceItem] = []
    for row in raw_items:
        url = row["url"]
        evidence_id = row.get("evidence_id") or (
            "news-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        )
        items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                ticker=ticker.upper(),
                source_type=row.get("source_type", default_source),
                source_url=url,
                published_at=_parse_dt(row["published_at"]),
                observed_fact=row["title"],
                retrieved_at=retrieved,
            )
        )
    return items
