from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# The run-loop rebuilds the whole cycle every tick, so the cache must live on
# disk to survive process boundaries. Committee decisions are keyed by a digest
# of the INPUTS THAT MATTER to the reasoning: the evidence items and the
# candidate contract codes. Quote drift (bid/ask) deliberately does not change
# the digest -- fresh quotes are re-validated downstream by the liquidity
# check and the risk gate on every pass anyway.

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
    """Per-ticker cache of the last committee decision, keyed by input digest.

    A cache hit means the SEC/news evidence and the eligible contract list are
    unchanged since the committee last reasoned about this ticker, so the
    decision is reused instead of spending five LLM calls re-deriving it.
    Entries expire after ``ttl_hours`` so a stale conviction cannot persist
    across sessions.
    """

    def __init__(self, path: Path, *, ttl_hours: float = DEFAULT_TTL_HOURS):
        self.path = path
        self.ttl = timedelta(hours=ttl_hours)

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def get(self, ticker: str, digest: str, now: datetime | None = None) -> str | None:
        """Return the cached CommitteeOutput JSON, or None on miss/expiry."""

        current = now or datetime.now(timezone.utc)
        entry = self._load().get(ticker.upper())
        if not entry or entry.get("digest") != digest:
            return None
        try:
            cached_at = datetime.fromisoformat(entry["at"])
        except (KeyError, ValueError):
            return None
        if current - cached_at > self.ttl:
            return None
        return entry.get("output")

    def put(
        self,
        ticker: str,
        digest: str,
        output_json: str,
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(timezone.utc)
        data = self._load()
        data[ticker.upper()] = {
            "digest": digest,
            "at": current.isoformat(),
            "output": output_json,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")
