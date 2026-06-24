"""Persisted counter for the live-trade ramp.

The first ``LIVE_RAMP_TRADES`` (default 10) completed live trades cap the
agent to one open position.  The operator must explicitly review each trade
(via ``trading-agent review-live``) before the reviewed count advances and
the ramp widens.

Layout (JSON)::

    {"trades_completed": 3, "reviewed_count": 2}
"""

from __future__ import annotations

from pathlib import Path

from trading_agent.storage import atomic_write_json


class LiveTradeCounter:
    """Thin wrapper around a single JSON file.

    All operations are safe to call from a single-process context (the agent
    runs under ``single_instance_lock``).
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    # -- public helpers -----------------------------------------------------

    def load(self) -> _Counter:
        """Return current counts, defaulting both fields to 0."""
        if not self._path.exists():
            return _Counter()
        try:
            from json import load

            with open(self._path, encoding="utf-8") as f:
                data = load(f) or {}
            return _Counter(
                trades_completed=int(data.get("trades_completed", 0)),
                reviewed_count=int(data.get("reviewed_count", 0)),
            )
        except Exception:
            return _Counter()

    def increment_completed(self) -> int:
        """Bump *trades_completed* by 1 (called by orchestrator on mark_closed).

        Returns the new value.
        """
        state = self.load()
        state.trades_completed += 1
        self._save(state)
        return state.trades_completed

    def increment_reviewed(self) -> int:
        """Bump *reviewed_count* by 1 (called by ``review-live`` CLI).

        Returns the new value.
        """
        state = self.load()
        state.reviewed_count += 1
        self._save(state)
        return state.reviewed_count

    # -- internal ------------------------------------------------------------

    def _save(self, state: _Counter) -> None:
        atomic_write_json(
            self._path,
            {
                "trades_completed": state.trades_completed,
                "reviewed_count": state.reviewed_count,
            },
        )


class _Counter:
    """Simple value object — no behaviour, easy to construct."""

    __slots__ = ("trades_completed", "reviewed_count")

    def __init__(
        self, trades_completed: int = 0, reviewed_count: int = 0
    ) -> None:
        self.trades_completed = trades_completed
        self.reviewed_count = reviewed_count
