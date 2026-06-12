from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROBE_ELIGIBLE = "eligible"
PROBE_NO_CHAIN = "no_chain"
PROBE_NO_CONTRACT = "no_eligible_contract"
PROBE_ERROR = "error"

DEFAULT_TTL_HOURS: dict[str, float] = {
    PROBE_NO_CHAIN: 24.0,
    PROBE_NO_CONTRACT: 2.0,
}


class ProbeCache:
    """Disk-backed negative cache with in-memory layer for universe probes.

    Uses in-memory cache to avoid repeated disk reads within a single cycle.
    """

    def __init__(
        self,
        path: Path,
        *,
        ttl_hours_by_status: dict[str, float] | None = None,
    ):
        self.path = path
        self.ttls = {
            status: timedelta(hours=hours)
            for status, hours in (ttl_hours_by_status or DEFAULT_TTL_HOURS).items()
        }
        self._data: dict[str, dict[str, str]] | None = None

    def _ensure_loaded(self) -> dict[str, dict[str, str]]:
        """Lazy-load from disk on first access."""
        if self._data is None:
            self._data = self._load_from_disk()
        return self._data

    def _load_from_disk(self) -> dict[str, dict[str, str]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def flush(self) -> None:
        """Write in-memory cache to disk."""
        if self._data is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def get(self, ticker: str, now: datetime | None = None) -> str | None:
        """Return the cached status, or None on miss/expiry."""
        current = now or datetime.now(timezone.utc)
        entry = self._ensure_loaded().get(ticker.upper())
        if not entry:
            return None
        status = entry.get("status")
        ttl = self.ttls.get(str(status))
        if ttl is None:
            return None
        try:
            cached_at = datetime.fromisoformat(entry["at"])
        except (KeyError, ValueError):
            return None
        if current - cached_at > ttl:
            return None
        return status

    def put(self, ticker: str, status: str, now: datetime | None = None) -> None:
        if status not in self.ttls:
            return
        current = now or datetime.now(timezone.utc)
        data = self._ensure_loaded()
        data[ticker.upper()] = {"status": status, "at": current.isoformat()}
        self.flush()
