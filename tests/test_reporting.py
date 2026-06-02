from datetime import date

from trading_agent.reporting import build_daily_report

DAY = "2026-06-02"
OTHER_DAY = "2026-06-01"


def _ev(event_type: str, payload: dict, *, day: str = DAY) -> dict:
    return {"recorded_at": f"{day}T15:00:00+00:00", "event_type": event_type, "payload": payload}


def _events() -> list[dict]:
    return [
        _ev("committee_run", {"decision": "open_position"}),
        _ev("committee_run", {"decision": "reject"}),
        _ev("proposal_checked", {"decision": {"approved": True}}),
        _ev("proposal_checked", {"decision": {"approved": False}}),
        _ev("candidate_validated", {"result": {"passed": True, "spread_pct": 10.0}}),
        _ev("candidate_validated", {"result": {"passed": False, "spread_pct": 20.0}}),
        _ev("order_placed", {"side": "buy"}),
        _ev("order_placed", {"side": "sell"}),
        _ev("position_closed", {"reason": "take profit", "realized_pnl_usd": 12.5}),
        _ev("position_closed", {"reason": "stop loss", "realized_pnl_usd": -5.0}),
        _ev("cycle_step_failed", {"stage": "entry"}),
        _ev("committee_run", {"decision": "hold"}, day=OTHER_DAY),  # different day, excluded
    ]


def test_report_aggregates_single_day() -> None:
    report = build_daily_report(_events(), date(2026, 6, 2))
    assert report["proposals_processed"] == 2
    assert report["committee_decisions"] == {"open_position": 1, "hold": 0, "reject": 1}
    assert report["gate"] == {"approved": 1, "rejected": 1}
    assert report["contracts"] == {"passed": 1, "failed": 1}
    assert report["avg_spread_pct"] == 15.0
    assert report["orders"]["placed"] == 2
    assert report["positions_closed"] == 2
    assert report["close_reasons"] == {"take profit": 1, "stop loss": 1}
    assert report["realized_pnl_usd"] == 7.5
    assert report["failures"] == 1


def test_report_excludes_other_days() -> None:
    report = build_daily_report(_events(), date(2026, 6, 1))
    assert report["committee_decisions"]["hold"] == 1
    assert report["proposals_processed"] == 1


def test_empty_report() -> None:
    report = build_daily_report([], date(2026, 6, 2))
    assert report["events"] == 0
    assert report["avg_spread_pct"] is None
    assert report["realized_pnl_usd"] == 0.0
