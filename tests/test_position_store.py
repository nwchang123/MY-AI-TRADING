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
