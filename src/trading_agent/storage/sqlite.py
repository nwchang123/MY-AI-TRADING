from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    option_code TEXT NOT NULL,
    passed INTEGER NOT NULL,
    reasons TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON candidate_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_snapshots_recorded ON candidate_snapshots(recorded_at);
"""


class SnapshotStore:
    """SQLite store for option-candidate liquidity snapshots (plan section 2)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_candidate(
        self,
        *,
        ticker: str,
        option_code: str,
        passed: bool,
        reasons: list[str],
        payload: dict[str, Any],
    ) -> int:
        recorded_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO candidate_snapshots "
                "(recorded_at, ticker, option_code, passed, reasons, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    recorded_at,
                    ticker,
                    option_code,
                    1 if passed else 0,
                    json.dumps(reasons, sort_keys=True),
                    json.dumps(payload, sort_keys=True, default=str),
                ),
            )
            return int(cursor.lastrowid)

    def list_candidates(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM candidate_snapshots ORDER BY id DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "recorded_at": row["recorded_at"],
            "ticker": row["ticker"],
            "option_code": row["option_code"],
            "passed": bool(row["passed"]),
            "reasons": json.loads(row["reasons"]),
            "payload": json.loads(row["payload"]),
        }
