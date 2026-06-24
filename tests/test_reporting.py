import json
from datetime import date

from trading_agent.reporting import (
    build_calibration_report,
    build_daily_report,
    build_funnel_report,
    build_portfolio_risk_report,
    read_audit_events,
)

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
        _ev(
            "llm_usage",
            {
                "calls": 5,
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 155,
            },
        ),
        _ev("llm_usage", {"calls": 5, "prompt_tokens": 50, "completion_tokens": 10}),
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
    assert report["llm_usage"] == {
        "calls": 10,
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 215,
    }


def test_report_uses_us_market_date_not_utc_date() -> None:
    event = {
        "recorded_at": "2026-06-02T01:00:00+00:00",
        "event_type": "committee_run",
        "payload": {"decision": "open_position"},
    }

    assert build_daily_report([event], date(2026, 6, 1))["proposals_processed"] == 1
    assert build_daily_report([event], date(2026, 6, 2))["proposals_processed"] == 0


def test_report_excludes_other_days() -> None:
    report = build_daily_report(_events(), date(2026, 6, 1))
    assert report["committee_decisions"]["hold"] == 1
    assert report["proposals_processed"] == 1


def test_read_audit_events_can_filter_while_streaming(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_ev("committee_run", {"decision": "open_position"})),
                json.dumps(_ev("committee_run", {"decision": "hold"}, day=OTHER_DAY)),
                "not-json",
            ]
        ),
        encoding="utf-8",
    )

    events = read_audit_events(path, date(2026, 6, 2))

    assert [event["payload"]["decision"] for event in events] == ["open_position"]


def test_empty_report() -> None:
    report = build_daily_report([], date(2026, 6, 2))
    assert report["events"] == 0
    assert report["avg_spread_pct"] is None
    assert report["realized_pnl_usd"] == 0.0
    assert report["llm_usage"]["total_tokens"] == 0


def _funnel_events() -> list[dict]:
    return [
        _ev("universe_selected", {
            "tickers": ["AAA", "BBB", "CCC"],
            "event_seeds": ["AAA"],
            "probe_fetches": 12,
            "probe_cache_hits": 40,
        }),
        _ev("no_eligible_contracts", {"ticker": "BBB"}),
        _ev("committee_run", {"decision": "open_position"}),
        _ev("committee_run", {"decision": "reject"}),
        _ev("committee_cache_hit", {"ticker": "CCC"}),
        _ev("monte_carlo_pop", {"pop": 0.1, "floor": 0.3}),  # rejected
        _ev("monte_carlo_pop", {"pop": 0.5, "floor": 0.3}),  # passed
        _ev("candidate_validated", {"result": {"passed": True}}),
        _ev("proposal_checked", {"decision": {"approved": True}}),
        _ev("order_filled", {"side": "buy"}),
        _ev("order_filled", {"side": "sell"}),  # an exit, not an entry
    ]


def test_funnel_counts_each_stage() -> None:
    funnel = build_funnel_report(_funnel_events())
    assert funnel["universe_selected"] == 3
    assert funnel["event_seeds"] == 1
    assert funnel["probe_fetches"] == 12
    assert funnel["probe_cache_hits"] == 40
    assert funnel["dropped_no_eligible_contract"] == 1
    assert funnel["reached_committee"] == 2
    assert funnel["served_from_cache"] == 1
    assert funnel["committee_decisions"] == {"open_position": 1, "hold": 0, "reject": 1}
    assert funnel["monte_carlo_checked"] == 2
    assert funnel["monte_carlo_rejected"] == 1
    assert funnel["risk_gate"] == {"approved": 1, "rejected": 0}
    assert funnel["orders_filled"] == 1  # only the buy counts as an entry


def test_funnel_scopes_to_a_date() -> None:
    events = [
        _ev("universe_selected", {"tickers": ["AAA"]}),
        _ev("universe_selected", {"tickers": ["X", "Y"]}, day=OTHER_DAY),
    ]
    assert build_funnel_report(events, date(2026, 6, 2))["universe_selected"] == 1
    assert build_funnel_report(events)["universe_selected"] == 3  # all days


def _calibration_events() -> list[dict]:
    def committee(code: str, win: float) -> dict:
        return _ev("committee_run", {
            "decision": "open_position",
            "output": {"win_probability": win, "proposal": {"option_code": code}},
        })

    return [
        committee("US.A260626C00005000", 0.58),
        committee("US.B260626C00005000", 0.72),
        committee("US.C260626C00005000", 0.80),
        _ev("position_closed", {"option_code": "US.A260626C00005000", "realized_pnl_usd": 8.0}),   # win, 0.55-0.65
        _ev("position_closed", {"option_code": "US.B260626C00005000", "realized_pnl_usd": -4.0}),  # loss, 0.65-0.75
        _ev("position_closed", {"option_code": "US.C260626C00005000", "realized_pnl_usd": 6.0}),   # win, 0.75+
    ]


def test_calibration_buckets_predicted_vs_actual() -> None:
    cal = build_calibration_report(_calibration_events())
    assert cal["proposals_with_estimate"] == 3
    assert cal["closed_trades_matched"] == 3
    by_range = {b["range"]: b for b in cal["buckets"]}
    assert by_range["0.55-0.65"]["trades"] == 1
    assert by_range["0.55-0.65"]["actual_win_rate"] == 1.0
    assert by_range["0.65-0.75"]["trades"] == 1
    assert by_range["0.65-0.75"]["actual_win_rate"] == 0.0
    # An untouched bucket reports None, not a divide-by-zero.
    assert by_range["0.00-0.55"]["actual_win_rate"] is None


def test_calibration_empty_until_trades_close() -> None:
    cal = build_calibration_report([])
    assert cal["closed_trades_matched"] == 0
    assert all(b["trades"] == 0 for b in cal["buckets"])


def test_portfolio_risk_report_sums_greeks_and_shocks() -> None:
    report = build_portfolio_risk_report(
        [
            {
                "status": "open",
                "ticker": "AAA",
                "option_code": "US.AAA260626C5000",
                "entry_price": 0.50,
                "contracts": 1,
                "lot_size": 100,
                "entry_spot": 10.0,
                "entry_iv": 0.5,
                "entry_delta": 0.5,
                "entry_gamma": 0.02,
                "entry_vega": 0.03,
                "entry_theta": -0.01,
                "entry_theta_decay_pct_per_day": 2.0,
                "entry_iv_rank": 0.4,
            },
            {
                "status": "open",
                "ticker": "BBB",
                "option_code": "US.BBB260626P5000",
                "entry_price": 0.40,
                "contracts": 1,
                "lot_size": 100,
                "entry_spot": 20.0,
                "entry_iv": 0.7,
                "entry_delta": -0.4,
                "entry_gamma": 0.01,
                "entry_vega": 0.02,
                "entry_theta": -0.02,
            },
        ],
        as_of=date(2026, 6, 2),
    )

    assert report["open_positions"] == 2
    assert report["premium_at_risk_usd"] == 90
    assert report["totals"] == {
        "delta": 10.0,
        "gamma": 3.0,
        "vega": 5.0,
        "theta_usd_per_day": -3.0,
    }
    assert report["theta_decay_usd_per_day"] == 3.0
    assert report["shock_pnl_usd"]["underlying_+2pct"] == -5.88
    assert set(report["scenario_revaluation_pnl_usd"]) == set(report["shock_pnl_usd"])
    assert report["scenario_revaluation_pnl_usd"]["underlying_+2pct"] > 0
    assert report["scenario_revaluation_skipped"] == 0
    assert report["missing_greeks"] == 0
