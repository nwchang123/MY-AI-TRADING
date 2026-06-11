from __future__ import annotations

import json
from datetime import date
from pathlib import Path


class DailyTokenBudget:
    """Persistent daily LLM token meter with a hard ceiling.

    The run-loop rebuilds the cycle (and its LLM clients) every tick, so
    in-process usage counters reset constantly; this file survives process
    boundaries and rolls over automatically on a new market day. A zero or
    negative limit disables the ceiling (remaining is unbounded).
    """

    def __init__(self, path: Path, limit_tokens: int):
        self.path = path
        self.limit = limit_tokens

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def used(self, day: date) -> int:
        data = self._load()
        if data.get("day") != day.isoformat():
            return 0
        return int(data.get("tokens") or 0)

    def remaining(self, day: date) -> int:
        if self.limit <= 0:
            return 2**31
        return max(0, self.limit - self.used(day))

    def add(self, tokens: int, day: date) -> None:
        if tokens <= 0:
            return
        total = self.used(day) + tokens
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"day": day.isoformat(), "tokens": total}), encoding="utf-8"
        )
