"""Efficient tail reader for the append-only audit JSONL log.

The full audit file can grow to many MB.  Reading the *last N* events must
not scan the entire file each time.  This module seeks from the end and
returns the trailing events in forward order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Maximum bytes to search backwards when the file has no trailing newlines.
_MAX_TAIL_BYTES = 256 * 1024


def read_recent_audit_events(path: Path, limit: int = 80) -> list[dict[str, Any]]:
    """Return up to *limit* most-recent valid JSONL events from *path*.

    Reads from the tail of the file so large logs don't cause O(N) scans.
    Events are returned in chronological (oldest-first) order.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("rb") as fh:
        # Seek backwards to find enough complete lines.
        fh.seek(0, 2)
        file_size = fh.tell()

        # If the file is small, just read the whole thing.
        if file_size <= _MAX_TAIL_BYTES:
            fh.seek(0)
            raw = fh.read().decode("utf-8", errors="replace")
        else:
            seek_pos = max(0, file_size - _MAX_TAIL_BYTES)
            fh.seek(seek_pos)
            raw = fh.read().decode("utf-8", errors="replace")
            # If we didn't start at byte 0, drop the first (potentially partial) line.
            if seek_pos > 0:
                nl = raw.find("\n")
                if nl != -1:
                    raw = raw[nl + 1 :]

    # Parse lines and keep only valid JSON events.
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Return the most recent *limit* events in chronological order.
    return events[-limit:] if len(events) > limit else events


def read_audit_events_by_date(
    path: Path, target_date: str, limit: int = 200
) -> list[dict[str, Any]]:
    """Read events where ``recorded_at`` starts with *target_date* (YYYY-MM-DD)."""
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            recorded = event.get("recorded_at", "")
            if recorded.startswith(target_date):
                events.append(event)
                if len(events) >= limit:
                    break
    return events
