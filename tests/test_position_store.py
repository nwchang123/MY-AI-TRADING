import sqlite3
from datetime import date
from pathlib import Path

from trading_agent.storage.positions import PositionStore


def _open(store: PositionStore, code: str = "US.EXAMPLE260626C00005000") -> None:
    store.open_position(
        option_code=code,
        ticker="EXAMPLE",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 6, 26),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 6, 24),
    )


def test_open_and_list(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    _open(store)
    rows = store.open_positions()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "EXAMPLE"
    assert rows[0]["status"] == "open"


def test_mark_closed_removes_from_open(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    _open(store)
    store.mark_closed(
        "US.EXAMPLE260626C00005000", close_reason="take profit", exit_price=0.40
    )
    assert store.open_positions() == []
    everything = store.all_positions()
    assert len(everything) == 1
    assert everything[0]["status"] == "closed"
    assert everything[0]["close_reason"] == "take profit"
    assert everything[0]["exit_price"] == 0.40


def test_get_returns_none_for_missing(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    assert store.get("US.NONE") is None


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "p.sqlite"
    _open(PositionStore(path))
    assert len(PositionStore(path).open_positions()) == 1


def test_catalyst_window_end_is_stored_and_defaults_null(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    store.open_position(
        option_code="US.WIN260626P00005000",
        ticker="WIN",
        option_side="put",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 6, 26),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 6, 24),
        catalyst_window_end=date(2026, 6, 18),
    )
    _open(store)  # the legacy helper passes no window -> NULL
    rows = {r["option_code"]: r for r in store.open_positions()}
    assert rows["US.WIN260626P00005000"]["catalyst_window_end"] == "2026-06-18"
    assert rows["US.EXAMPLE260626C00005000"]["catalyst_window_end"] is None


def test_migration_adds_column_to_legacy_ledger(tmp_path: Path) -> None:
    # A pre-migration DB has the original schema without catalyst_window_end.
    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE positions ("
        "option_code TEXT PRIMARY KEY, ticker TEXT NOT NULL, option_side TEXT NOT NULL,"
        "entry_price REAL NOT NULL, contracts INTEGER NOT NULL, lot_size INTEGER NOT NULL,"
        "expiry TEXT NOT NULL, take_profit_pct REAL NOT NULL, stop_loss_pct REAL NOT NULL,"
        "time_stop TEXT NOT NULL, status TEXT NOT NULL, opened_at TEXT NOT NULL,"
        "closed_at TEXT, close_reason TEXT, exit_price REAL);"
    )
    conn.commit()
    conn.close()

    # Opening the store migrates the table; the new column is usable.
    store = PositionStore(path)
    _open(store)
    assert store.open_positions()[0]["catalyst_window_end"] is None
