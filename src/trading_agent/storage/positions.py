from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    option_code TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    option_side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    contracts INTEGER NOT NULL,
    lot_size INTEGER NOT NULL,
    expiry TEXT NOT NULL,
    take_profit_pct REAL NOT NULL,
    stop_loss_pct REAL NOT NULL,
    time_stop TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    close_reason TEXT,
    exit_price REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
"""

# Columns added after the original schema shipped. Applied idempotently on init
# so a ledger written by an older build keeps working (SQLite ADD COLUMN is
# non-destructive and cheap). Each entry is (name, column-type).
_ADDED_COLUMNS: list[tuple[str, str]] = [
    ("catalyst_window_end", "TEXT"),
    ("pre_earnings_exit_date", "TEXT"),
]


class PositionStore:
    """Local ledger of opened positions and their exit plans.

    The broker is the source of truth for what is *held*; this ledger holds the
    exit plan and entry price the broker does not, so exits and restart
    reconciliation can be driven deterministically.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(positions)")}
        for name, col_type in _ADDED_COLUMNS:
            if name not in existing:
                conn.execute(f"ALTER TABLE positions ADD COLUMN {name} {col_type}")

    def open_position(
        self,
        *,
        option_code: str,
        ticker: str,
        option_side: str,
        entry_price: float,
        contracts: int,
        lot_size: int,
        expiry: date,
        take_profit_pct: float,
        stop_loss_pct: float,
        time_stop: date,
        catalyst_window_end: date | None = None,
        pre_earnings_exit_date: date | None = None,
    ) -> None:
        opened_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO positions "
                "(option_code, ticker, option_side, entry_price, contracts, lot_size, "
                "expiry, take_profit_pct, stop_loss_pct, time_stop, status, opened_at, "
                "closed_at, close_reason, exit_price, catalyst_window_end, "
                "pre_earnings_exit_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, NULL, NULL, NULL, ?, ?)",
                (
                    option_code,
                    ticker,
                    option_side,
                    entry_price,
                    contracts,
                    lot_size,
                    expiry.isoformat(),
                    take_profit_pct,
                    stop_loss_pct,
                    time_stop.isoformat(),
                    opened_at,
                    catalyst_window_end.isoformat() if catalyst_window_end else None,
                    pre_earnings_exit_date.isoformat()
                    if pre_earnings_exit_date
                    else None,
                ),
            )

    def mark_closed(
        self, option_code: str, *, close_reason: str, exit_price: float | None
    ) -> None:
        closed_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE positions SET status='closed', closed_at=?, close_reason=?, "
                "exit_price=? WHERE option_code=?",
                (closed_at, close_reason, exit_price, option_code),
            )

    def open_positions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='open' ORDER BY opened_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def all_positions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions ORDER BY opened_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, option_code: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE option_code=?", (option_code,)
            ).fetchone()
        return dict(row) if row else None
