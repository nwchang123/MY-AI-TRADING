from pathlib import Path

from trading_agent.storage.sqlite import SnapshotStore


def test_record_and_list_candidate(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots.sqlite")
    row_id = store.record_candidate(
        ticker="EXAMPLE",
        option_code="US.EXAMPLE260626C00005000",
        passed=True,
        reasons=[],
        payload={"spread_pct": 10.0, "dte": 24},
    )
    assert row_id == 1

    rows = store.list_candidates()
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "EXAMPLE"
    assert row["passed"] is True
    assert row["reasons"] == []
    assert row["payload"]["dte"] == 24


def test_list_is_most_recent_first_and_respects_limit(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots.sqlite")
    for i in range(3):
        store.record_candidate(
            ticker=f"T{i}",
            option_code=f"US.T{i}",
            passed=False,
            reasons=["bid-ask spread is too wide"],
            payload={"i": i},
        )

    rows = store.list_candidates(limit=2)
    assert len(rows) == 2
    assert rows[0]["ticker"] == "T2"  # newest first
    assert rows[1]["ticker"] == "T1"


def test_store_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "snapshots.sqlite"
    SnapshotStore(path).record_candidate(
        ticker="EXAMPLE", option_code="US.X", passed=True, reasons=[], payload={}
    )
    reopened = SnapshotStore(path)
    assert len(reopened.list_candidates()) == 1
