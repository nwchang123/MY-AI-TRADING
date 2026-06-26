from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_agent.storage import atomic_write_json

DEFAULT_TTL_HOURS = 6.0


def decision_digest(
    ticker: str, evidence_ids: list[str], candidate_codes: list[str]
) -> str:
    payload = json.dumps(
        [ticker.upper(), sorted(evidence_ids), sorted(candidate_codes)],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def thesis_digest(ticker: str, evidence_ids: list[str]) -> str:
    """Contract-independent sub-digest: the thesis inputs only.

    A reject/hold turns on the evidence, not on which near-money strike was
    eligible this minute, so isolating the thesis lets diagnostics tell genuine
    evidence changes apart from the option-contract drift that busts the full
    ``decision_digest`` every cycle.
    """
    payload = json.dumps([ticker.upper(), sorted(evidence_ids)], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_digest(candidate_codes: list[str]) -> str:
    """Sub-digest of the eligible option contracts (the intraday-drifting part)."""
    payload = json.dumps(sorted(candidate_codes), separators=(",", ":"))
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
        """Write in-memory cache to disk atomically."""
        if self._data is not None:
            atomic_write_json(self.path, self._data)

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

    def get_by_thesis(
        self, ticker: str, thesis_digest: str, now: datetime | None = None
    ) -> str | None:
        """Reuse a cached HOLD/REJECT decision when only the contracts drifted.

        A reject/hold turns on the thesis (evidence), not on which near-money
        strike was eligible this minute, so an unchanged thesis can reuse the prior
        decision even when the candidate option codes churned -- the dominant
        cache-miss cause (the 4.8% hit rate diagnosed via ``classify_miss``).

        An ``open_position`` decision is NEVER reused this way: it names a specific
        contract that may no longer be eligible, so it still requires the exact
        full-``decision_digest`` match in :meth:`get`. Returns the cached output
        JSON, or None on miss/expiry/open.
        """
        current = now or datetime.now(timezone.utc)
        entry = self._ensure_loaded().get(ticker.upper())
        if not entry or entry.get("thesis") != thesis_digest:
            return None
        try:
            cached_at = datetime.fromisoformat(entry["at"])
        except (KeyError, ValueError):
            return None
        if current - cached_at > self.ttl:
            return None
        output = entry.get("output")
        try:
            decision = json.loads(output or "{}").get("decision")
        except json.JSONDecodeError:
            return None
        if decision == "open_position":
            return None
        return output

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
        *,
        thesis: str | None = None,
        candidates: str | None = None,
    ) -> None:
        current = now or datetime.now(timezone.utc)
        data = self._ensure_loaded()
        entry: dict[str, Any] = {
            "digest": digest,
            "at": current.isoformat(),
            "output": output_json,
        }
        # Sub-digests are optional and only used for miss diagnostics; older
        # entries written before this field simply classify as "legacy_entry".
        if thesis is not None:
            entry["thesis"] = thesis
        if candidates is not None:
            entry["candidates"] = candidates
        data[ticker.upper()] = entry
        self.flush()

    def classify_miss(
        self,
        ticker: str,
        thesis: str,
        candidates: str,
        now: datetime | None = None,
    ) -> str:
        """Why a lookup for ``ticker`` missed -- diagnostic only, no side effects.

        Returns one of: ``no_prior``, ``expired``, ``legacy_entry``,
        ``evidence_changed``, ``candidates_changed``, ``both_changed``,
        ``match``. ``candidates_changed`` means the thesis was unchanged but the
        eligible option contracts drifted -- the churn the full digest cannot
        tolerate, and the expected dominant cause of the low hit rate.
        """
        current = now or datetime.now(timezone.utc)
        entry = self._ensure_loaded().get(ticker.upper())
        if not entry:
            return "no_prior"
        try:
            cached_at = datetime.fromisoformat(entry["at"])
        except (KeyError, ValueError):
            return "no_prior"
        if current - cached_at > self.ttl:
            return "expired"
        stored_thesis = entry.get("thesis")
        stored_candidates = entry.get("candidates")
        if stored_thesis is None or stored_candidates is None:
            return "legacy_entry"
        thesis_match = stored_thesis == thesis
        candidates_match = stored_candidates == candidates
        if thesis_match and candidates_match:
            return "match"
        if thesis_match:
            return "candidates_changed"
        if candidates_match:
            return "evidence_changed"
        return "both_changed"
