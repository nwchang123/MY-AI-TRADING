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
    assert store.closed_positions()[0]["option_code"] == "US.EXAMPLE260626C00005000"


def test_get_returns_none_for_missing(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    assert store.get("US.NONE") is None


def test_reentry_keeps_both_trade_records(tmp_path: Path) -> None:
    # A contract closed and then re-opened must keep BOTH rows, so realized-P/L
    # replay and consecutive-loss accounting stay exact (was INSERT OR REPLACE,
    # which destroyed the earlier trade).
    store = PositionStore(tmp_path / "p.sqlite")
    _open(store)
    store.mark_closed(
        "US.EXAMPLE260626C00005000", close_reason="take profit", exit_price=0.40
    )
    # Re-enter the SAME code at a different price after the close.
    _open(store)
    rows = store.all_positions()
    assert len(rows) == 2
    closed, reopened = rows
    assert closed["status"] == "closed"
    assert closed["exit_price"] == 0.40
    assert reopened["status"] == "open"
    assert reopened["exit_price"] is None
    # Distinct trade ids.
    assert closed["trade_id"] != reopened["trade_id"]
    # Exactly one open position (the re-entry), not two.
    assert len(store.open_positions()) == 1


def test_mark_closed_only_touches_open_row(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    _open(store)
    store.mark_closed(
        "US.EXAMPLE260626C00005000", close_reason="stop loss", exit_price=0.10
    )
    _open(store)  # re-entry
    # Closing again must not rewrite the historical closed trade's exit_price.
    store.mark_closed(
        "US.EXAMPLE260626C00005000", close_reason="take profit", exit_price=0.50
    )
    closed_rows = [r for r in store.all_positions() if r["status"] == "closed"]
    assert len(closed_rows) == 2
    assert {r["exit_price"] for r in closed_rows} == {0.10, 0.50}
    assert {r["close_reason"] for r in closed_rows} == {"stop loss", "take profit"}


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "p.sqlite"
    _open(PositionStore(path))
    assert len(PositionStore(path).open_positions()) == 1


def test_connection_is_reused_until_closed(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    first = store._connect()
    second = store._connect()
    assert first is second

    store.close()
    reopened = store._connect()
    assert reopened is not first
    store.close()


def test_opened_at_values_are_lightweight(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    _open(store)

    values = store.opened_at_values()

    assert len(values) == 1
    assert "T" in values[0]


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


def test_entry_greeks_are_stored(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    store.open_position(
        option_code="US.GRK260626C00005000",
        ticker="GRK",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 6, 26),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 6, 24),
        entry_spot=5.0,
        entry_iv=0.5,
        entry_delta=0.51,
        entry_gamma=0.25,
        entry_vega=0.01,
        entry_theta=-0.004,
        entry_theta_decay_pct_per_day=2.0,
        entry_iv_rank=0.4,
    )

    row = store.open_positions()[0]
    assert row["entry_delta"] == 0.51
    assert row["entry_gamma"] == 0.25
    assert row["entry_theta_decay_pct_per_day"] == 2.0


def test_peak_bid_is_stored_and_updated(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "p.sqlite")
    _open(store)

    store.update_peak_bid("US.EXAMPLE260626C00005000", 0.31)

    row = store.open_positions()[0]
    assert row["peak_bid"] == 0.31


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
    assert store.open_positions()[0]["peak_bid"] is None
