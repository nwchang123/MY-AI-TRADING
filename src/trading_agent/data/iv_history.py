"""Persistent IV30 history for computing IV Rank.

Stores one IV30 observation per ticker per day in a local JSON file.
The orchestrator appends after each snapshot; iv_rank() reads the history
to compute the 52-week range.  No external API needed -- the data already
arrives via the CBOE/Yahoo snapshot's ``iv30`` field.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


class IV30History:
    """Per-ticker IV30 history stored on disk as a JSON dict.

    Layout::

        {
          "AAPL": [
            {"date": "2026-06-01", "iv30": 0.28},
            {"date": "2026-06-02", "iv30": 0.31},
            ...
          ],
          "TSLA": [...]
        }

    Only the most recent ``max_days`` entries are kept per ticker.
    ``max_days`` defaults to 371 (~365 calendar days ≈ 252 trading days),
    matching the standard 1-year IV Rank lookback window.
    """

    def __init__(self, path: Path, *, max_days: int = 371) -> None:
        self.path = path
        self.max_days = max_days
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._load()

    # -- persistence --

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=1, sort_keys=True), encoding="utf-8"
        )

    # -- public API --

    def record(self, ticker: str, iv30: float, *, today: date | None = None) -> None:
        """Append today's IV30 for a ticker.  Ignores zero/missing values."""

        if iv30 <= 0:
            return
        day = (today or date.today()).isoformat()
        key = ticker.upper()
        entries = self._data.setdefault(key, [])

        # One observation per day: replace if same date already present.
        if entries and entries[-1].get("date") == day:
            if entries[-1].get("iv30") == iv30:
                return
            entries[-1]["iv30"] = iv30
        else:
            entries.append({"date": day, "iv30": iv30})

        # Trim to max_days.
        if len(entries) > self.max_days:
            self._data[key] = entries[-self.max_days:]
        self._save()

    def values(self, ticker: str) -> list[float]:
        """Return clean IV30 observations for a ticker in stored order."""

        entries = self._data.get(ticker.upper())
        if not entries:
            return []
        return [
            float(e["iv30"])
            for e in entries
            if isinstance(e.get("iv30"), (int, float)) and e["iv30"] > 0
        ]

    def extremes(self, ticker: str) -> tuple[float, float] | None:
        """Return (52w_high, 52w_low) IV30 for a ticker, or None if empty."""

        ivs = self.values(ticker)
        if len(ivs) < 2:
            return None
        return (max(ivs), min(ivs))

    def rank(self, ticker: str, current_iv: float) -> float | None:
        """IV Rank: current IV's position inside the stored high/low range."""

        if current_iv <= 0:
            return None
        ext = self.extremes(ticker)
        if ext is None:
            return None
        high, low = ext
        if high <= low:
            return None
        return round((current_iv - low) / (high - low), 4)

    def percentile(self, ticker: str, current_iv: float) -> float | None:
        """IV Percentile: share of stored observations below current IV."""

        if current_iv <= 0:
            return None
        ivs = self.values(ticker)
        if len(ivs) < 2:
            return None
        below = sum(1 for iv in ivs if iv < current_iv)
        return round(below / len(ivs), 4)
