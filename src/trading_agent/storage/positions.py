from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# NOTE on the schema: ``trade_id`` is the primary key, NOT ``option_code``.
# A contract that is closed and then re-opened later (same strike/expiry) must
# keep BOTH trade records -- otherwise the realized P/L, drawdown, and
# consecutive-loss accounting (all replayed from closed rows) get silently
# corrupted. The original schema used ``option_code`` as PRIMARY KEY with
# ``INSERT OR REPLACE``, which destroyed the earlier trade on every re-entry.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    option_code TEXT NOT NULL,
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
    exit_price REAL,
    catalyst_window_end TEXT,
    pre_earnings_exit_date TEXT,
    entry_spot REAL,
    entry_iv REAL,
    entry_delta REAL,
    entry_gamma REAL,
    entry_vega REAL,
    entry_theta REAL,
    entry_theta_decay_pct_per_day REAL,
    entry_iv_rank REAL,
    peak_bid REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_option_code ON positions(option_code);
CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at);
"""

# Columns added after the original schema shipped. Applied idempotently on init
# so a ledger written by an older build keeps working (SQLite ADD COLUMN is
# non-destructive and cheap). Each entry is (name, column-type).
_ADDED_COLUMNS: list[tuple[str, str]] = [
    ("catalyst_window_end", "TEXT"),
    ("pre_earnings_exit_date", "TEXT"),
    ("entry_spot", "REAL"),
    ("entry_iv", "REAL"),
    ("entry_delta", "REAL"),
    ("entry_gamma", "REAL"),
    ("entry_vega", "REAL"),
    ("entry_theta", "REAL"),
    ("entry_theta_decay_pct_per_day", "REAL"),
    ("entry_iv_rank", "REAL"),
    ("peak_bid", "REAL"),
]


class PositionStore:
    """Local ledger of opened positions and their exit plans.

    The broker is the source of truth for what is *held*; this ledger holds the
    exit plan and entry price the broker does not, so exits and restart
    reconciliation can be driven deterministically. Each (re-)entry gets its own
    row so a contract traded more than once keeps a complete history.
    """

    def __init__(self, path: Path):
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._ensure_schema(conn)

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create the table or migrate a legacy ``option_code``-PK ledger.

        Old ledgers used ``option_code TEXT PRIMARY KEY``. Those are rebuilt in
        place into the new ``trade_id``-keyed schema, copying every existing row
        so realized-P/L replay and consecutive-loss accounting stay correct.
        """

        existing = {
            row["name"]: row for row in conn.execute("PRAGMA table_info(positions)")
        }
        if not existing:
            conn.executescript(_SCHEMA)
            return

        # Already on the new schema (trade_id PK): just backfill any added cols.
        if "trade_id" in existing and existing["trade_id"]["pk"] == 1:
            for name, col_type in _ADDED_COLUMNS:
                if name not in existing:
                    conn.execute(f"ALTER TABLE positions ADD COLUMN {name} {col_type}")
            return

        # Legacy schema: option_code PRIMARY KEY, possibly missing the added
        # columns. Rebuild into the new shape via a temp table + swap.
        self._migrate_legacy(conn)

    @staticmethod
    def _migrate_legacy(conn: sqlite3.Connection) -> None:
        """Rebuild a legacy ``option_code``-PK table into the new ``trade_id``-PK shape."""

        old_cols = {row["name"] for row in conn.execute("PRAGMA table_info(positions)")}
        needed_extra = [
            name for name, _ in _ADDED_COLUMNS if name not in old_cols
        ]
        # Add any missing added columns first so the copy is uniform.
        for name in [n for n, _ in _ADDED_COLUMNS if n not in old_cols]:
            col_type = dict(_ADDED_COLUMNS)[name]
            conn.execute(f"ALTER TABLE positions ADD COLUMN {name} {col_type}")

        conn.executescript(
            """
            CREATE TABLE positions_new (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                option_code TEXT NOT NULL,
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
                exit_price REAL,
                catalyst_window_end TEXT,
                pre_earnings_exit_date TEXT,
                entry_spot REAL,
                entry_iv REAL,
                entry_delta REAL,
                entry_gamma REAL,
                entry_vega REAL,
                entry_theta REAL,
                entry_theta_decay_pct_per_day REAL,
                entry_iv_rank REAL,
                peak_bid REAL
            );
            """
        )
        # Preserve insertion order via rowid so AUTOINCREMENT trade_ids follow the
        # original opened_at ordering on legacy data.
        conn.execute(
            "INSERT INTO positions_new (option_code, ticker, option_side, entry_price, "
            "contracts, lot_size, expiry, take_profit_pct, stop_loss_pct, time_stop, "
            "status, opened_at, closed_at, close_reason, exit_price, "
            "catalyst_window_end, pre_earnings_exit_date, entry_spot, entry_iv, "
            "entry_delta, entry_gamma, entry_vega, entry_theta, "
            "entry_theta_decay_pct_per_day, entry_iv_rank, peak_bid) "
            "SELECT option_code, ticker, option_side, entry_price, contracts, lot_size, "
            "expiry, take_profit_pct, stop_loss_pct, time_stop, status, opened_at, "
            "closed_at, close_reason, exit_price, catalyst_window_end, "
            "pre_earnings_exit_date, entry_spot, entry_iv, entry_delta, "
            "entry_gamma, entry_vega, entry_theta, entry_theta_decay_pct_per_day, "
            "entry_iv_rank, peak_bid FROM positions ORDER BY rowid"
        )
        conn.execute("DROP TABLE positions")
        conn.execute("ALTER TABLE positions_new RENAME TO positions")
        conn.executescript(
            "CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);"
            "CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);"
            "CREATE INDEX IF NOT EXISTS idx_positions_option_code ON positions(option_code);"
            "CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at);"
        )

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
        entry_spot: float | None = None,
        entry_iv: float | None = None,
        entry_delta: float | None = None,
        entry_gamma: float | None = None,
        entry_vega: float | None = None,
        entry_theta: float | None = None,
        entry_theta_decay_pct_per_day: float | None = None,
        entry_iv_rank: float | None = None,
        peak_bid: float | None = None,
    ) -> int:
        """Record a new trade. Returns the assigned ``trade_id``.

        Each call inserts a fresh row -- a re-entry on the same option_code keeps
        the earlier (closed) trade intact, so equity/drawdown replays stay exact.
        """

        opened_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO positions "
                "(option_code, ticker, option_side, entry_price, contracts, lot_size, "
                "expiry, take_profit_pct, stop_loss_pct, time_stop, status, opened_at, "
                "closed_at, close_reason, exit_price, catalyst_window_end, "
                "pre_earnings_exit_date, entry_spot, entry_iv, entry_delta, "
                "entry_gamma, entry_vega, entry_theta, "
                "entry_theta_decay_pct_per_day, entry_iv_rank, peak_bid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, NULL, NULL, "
                "NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    entry_spot,
                    entry_iv,
                    entry_delta,
                    entry_gamma,
                    entry_vega,
                    entry_theta,
                    entry_theta_decay_pct_per_day,
                    entry_iv_rank,
                    peak_bid,
                ),
            )
            return int(cur.lastrowid)

    def update_peak_bid(self, option_code: str, peak_bid: float) -> int:
        """Persist the best observed exit bid for the current open trade."""

        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE positions SET peak_bid=? "
                "WHERE trade_id IN ("
                "  SELECT trade_id FROM positions "
                "  WHERE option_code=? AND status='open' "
                "  ORDER BY trade_id DESC LIMIT 1"
                ")",
                (peak_bid, option_code),
            )
            return int(cur.rowcount)

    def mark_closed(
        self,
        option_code: str,
        *,
        close_reason: str,
        exit_price: float | None,
    ) -> int:
        """Close the OPEN trade for ``option_code`` (the most recent one).

        Only the open row is touched: closed trades on the same code are kept.
        Returns the number of rows updated (0 when there was nothing to close,
        which happens when reconciliation races with itself). The caller treats
        an unknown option_code as "already not held" regardless.
        """

        closed_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE positions SET status='closed', closed_at=?, close_reason=?, "
                "exit_price=? "
                "WHERE trade_id IN ("
                "  SELECT trade_id FROM positions "
                "  WHERE option_code=? AND status='open' "
                "  ORDER BY trade_id DESC LIMIT 1"
                ")",
                (closed_at, close_reason, exit_price, option_code),
            )
            return int(cur.rowcount)

    def open_positions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='open' ORDER BY opened_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def closed_positions(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Closed rows with an exit price, in close-time order."""

        with self._connect() as conn:
            sql = (
                "SELECT * FROM positions "
                "WHERE status='closed' AND exit_price IS NOT NULL "
                "ORDER BY closed_at, trade_id"
            )
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            rows = conn.execute(sql).fetchall()
        return [dict(row) for row in rows]

    def opened_at_values(self) -> list[str]:
        """Open timestamps only, for cheap per-day entry counts."""

        with self._connect() as conn:
            rows = conn.execute("SELECT opened_at FROM positions").fetchall()
        return [str(row["opened_at"]) for row in rows]

    def all_positions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions ORDER BY opened_at, trade_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, option_code: str) -> dict[str, Any] | None:
        """The most recent trade on ``option_code`` (open preferred), or None.

        Kept for diagnostics; reconciliation uses ``mark_closed`` / ``open_positions``.
        """

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE option_code=? "
                "ORDER BY status='open' DESC, trade_id DESC LIMIT 1",
                (option_code,),
            ).fetchone()
            return dict(row) if row else None
