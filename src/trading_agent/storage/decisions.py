from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_TTL_HOURS = 6.0


def decision_digest(
    ticker: str, evidence_ids: list[str], candidate_codes: list[str]
) -> str:
    payload = json.dumps(
        [ticker.upper(), sorted(evidence_ids), sorted(candidate_codes)],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DecisionCache:
    """Per-ticker cache of the last committee decision with in-memory layer.

    Uses in-memory cache to avoid repeated disk reads within a single cycle.
    Data is flushed to disk on put() calls and can be explicitly flushed.
    """

    def __init__(self, path: Path, *, ttl_hours: float = DEFAULT_TTL_HOURS):
        self.path = path
        self.ttl = timedelta(hours=ttl_hours)
        self._data: dict[str, Any] | None = None

    def _ensure_loaded(self) -> dict[str, Any]:
        """Lazy-load from disk on first access."""
        if self._data is None:
            self._data = self._load_from_disk()
        return self._data

    def _load_from_disk(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def flush(self) -> None:
        """Write in-memory cache to disk."""
        if self._data is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def get(self, ticker: str, digest: str, now: datetime | None = None) -> str | None:
        """Return the cached CommitteeOutput JSON, or None on miss/expiry."""
        current = now or datetime.now(timezone.utc)
        entry = self._ensure_loaded().get(ticker.upper())
        if not entry or entry.get("digest") != digest:
            return None
        try:
            cached_at = datetime.fromisoformat(entry["at"])
        except (KeyError, ValueError):
            return None
        if current - cached_at > self.ttl:
            return None
        return entry.get("output")

    def fresh_rejections(
        self, now: datetime | None = None, *, within_hours: float | None = None
    ) -> set[str]:
        """Tickers whose recently cached decision was anything but an open."""
        current = now or datetime.now(timezone.utc)
        bench = self.ttl
        if within_hours is not None:
            bench = min(bench, timedelta(hours=within_hours))
        rejected: set[str] = set()
        for ticker, entry in self._ensure_loaded().items():
            try:
                cached_at = datetime.fromisoformat(entry["at"])
            except (KeyError, TypeError, ValueError):
                continue
            if current - cached_at > bench:
                continue
            try:
                decision = json.loads(entry.get("output") or "{}").get("decision")
            except json.JSONDecodeError:
                continue
            if decision and decision != "open_position":
                rejected.add(ticker.upper())
        return rejected

    def put(
        self,
        ticker: str,
        digest: str,
        output_json: str,
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(timezone.utc)
        data = self._ensure_loaded()
        data[ticker.upper()] = {
            "digest": digest,
            "at": current.isoformat(),
            "output": output_json,
        }
        self.flush()
