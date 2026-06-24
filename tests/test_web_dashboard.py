"""Tests for the web_dashboard module: state reads, audit tail, and API envelope."""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from trading_agent.web_dashboard import state
from trading_agent.web_dashboard.audit_tail import read_recent_audit_events


# --- audit_tail tests ---


def test_read_recent_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    assert read_recent_audit_events(path, limit=10) == []


def test_read_recent_returns_last_n(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    lines = []
    for i in range(20):
        lines.append(json.dumps({"event_type": f"ev_{i}", "recorded_at": f"2026-01-01T00:{i:02d}:00+00:00"}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = read_recent_audit_events(path, limit=5)
    assert len(result) == 5
    assert result[0]["event_type"] == "ev_15"
    assert result[-1]["event_type"] == "ev_19"


def test_read_recent_skips_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    lines = [
        json.dumps({"event_type": "ok", "recorded_at": "2026-01-01T00:00:00+00:00"}),
        "NOT JSON",
        json.dumps({"event_type": "ok2", "recorded_at": "2026-01-01T00:01:00+00:00"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = read_recent_audit_events(path, limit=10)
    assert len(result) == 2


# --- envelope tests ---


def test_envelope_ok() -> None:
    result = state.get_health()
    assert result["ok"] is True
    assert "generated_at" in result
    assert result["data"]["status"] == "running"


def test_status_no_runtime(tmp_path: Path) -> None:
    state._clear_cache()
    result = state.get_status(tmp_path)
    assert result["ok"] is True
    assert result["data"]["halted"] is False
    assert result["data"]["heartbeat_age_seconds"] is None


def test_status_with_halt(tmp_path: Path) -> None:
    state._clear_cache()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "HALT").write_text("test halt\n", encoding="utf-8")
    result = state.get_status(tmp_path)
    assert result["ok"] is True
    assert result["data"]["halted"] is True
    assert result["data"]["halt_reason"] == "test halt"


def test_status_with_heartbeat(tmp_path: Path) -> None:
    state._clear_cache()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    now = datetime.now(timezone.utc).isoformat()
    (runtime / "heartbeat.json").write_text(json.dumps({"at": now, "note": "cycle done"}), encoding="utf-8")
    result = state.get_status(tmp_path)
    assert result["ok"] is True
    assert result["data"]["heartbeat_age_seconds"] is not None
    assert result["data"]["heartbeat_age_seconds"] < 5
    assert result["data"]["heartbeat_stale"] is False


def test_positions_no_db(tmp_path: Path) -> None:
    state._clear_cache()
    result = state.get_positions(tmp_path)
    assert result["ok"] is True
    assert result["data"]["count"] == 0


def test_portfolio_risk_no_db(tmp_path: Path) -> None:
    state._clear_cache()
    result = state.get_portfolio_risk(tmp_path)
    assert result["ok"] is True
    assert result["data"]["open_positions"] == 0


def test_report_no_audit(tmp_path: Path) -> None:
    state._clear_cache()
    result = state.get_report_today(tmp_path)
    assert result["ok"] is True
    assert result["data"]["events"] == 0


def test_funnel_no_audit(tmp_path: Path) -> None:
    state._clear_cache()
    result = state.get_funnel_today(tmp_path)
    assert result["ok"] is True
    assert result["data"]["universe_selected"] == 0


def test_calibration_no_audit(tmp_path: Path) -> None:
    state._clear_cache()
    result = state.get_calibration_today(tmp_path)
    assert result["ok"] is True


def test_audit_recent_no_file(tmp_path: Path) -> None:
    state._clear_cache()
    result = state.get_audit_recent(tmp_path, limit=10)
    assert result["ok"] is True
    assert result["data"]["count"] == 0


def test_audit_recent_with_data(tmp_path: Path) -> None:
    state._clear_cache()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    lines = []
    for i in range(5):
        lines.append(json.dumps({"event_type": f"ev_{i}", "recorded_at": f"2026-01-01T00:{i:02d}:00+00:00"}))
    (runtime / "audit.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = state.get_audit_recent(tmp_path, limit=3)
    assert result["ok"] is True
    assert result["data"]["count"] == 3


def test_iv_no_file(tmp_path: Path) -> None:
    state._clear_cache()
    result = state.get_iv(tmp_path)
    assert result["ok"] is True
    assert result["data"] == {}


def test_mandate_load(tmp_path: Path) -> None:
    state._clear_cache()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "mandate.paper.yaml").write_text(
        "account:\n"
        "  initial_capital_usd: 1000\n"
        "  live_mode_requires_operator_flag: true\n"
        "  withdrawal_capability: forbidden\n"
        "  compounding: false\n"
        "universe:\n"
        "  market: US\n"
        "  min_underlying_price_usd: 5\n"
        "  min_market_cap_usd: 1e9\n"
        "  max_market_cap_usd: 50e9\n"
        "  min_average_daily_turnover_usd: 1e7\n"
        "  require_listed_equity: true\n"
        "  reject_otc: true\n"
        "  reject_halted: true\n"
        "options:\n"
        "  allowed_opening_actions: [buy_call]\n"
        "  allowed_closing_actions: [sell_to_close]\n"
        "  contracts_per_order: 1\n"
        "  min_dte: 14\n"
        "  max_dte: 45\n"
        "  min_option_bid_usd: 0.05\n"
        "  max_contract_cost_usd: 500\n"
        "  max_bid_ask_spread_pct: 15\n"
        "  min_open_interest: 25\n"
        "  min_daily_volume: 0\n"
        "  use_limit_orders_only: true\n"
        "  reject_auto_exercise: true\n"
        "  force_close_before_expiry_trading_days: 2\n"
        "  fee_buffer_usd: 1\n"
        "  min_estimated_win_probability: 0.55\n"
        "  min_monte_carlo_pop: 0.25\n"
        "  veto_win_prob_penalty: 0.02\n"
        "  earnings_window_min_days: 0\n"
        "  earnings_window_max_days: 0\n"
        "  pre_earnings_exit_trading_days: 2\n"
        "  max_entry_iv: 0\n"
        "  min_abs_delta: 0.20\n"
        "  max_gamma_per_contract: 2.0\n"
        "  max_theta_decay_pct_per_day: 8.0\n"
        "  max_iv_rank_for_long_premium: 0.80\n"
        "  iv_crush_exit_drop_pct: 25.0\n"
        "portfolio:\n"
        "  max_open_positions: 2\n"
        "  max_total_premium_at_risk_usd: 2000\n"
        "  max_single_position_cost_usd: 800\n"
        "  max_new_positions_per_day: 2\n"
        "  daily_loss_stop_usd: 2000\n"
        "  hard_drawdown_stop_usd: 2000\n"
        "  consecutive_loss_stop: 3\n"
        "  cooldown_after_consecutive_losses_hours: 24\n"
        "  max_vix_for_entries: 35\n"
        "execution:\n"
        "  commission_per_contract_usd: 0\n"
        "  stale_quote_seconds: 15\n"
        "  delayed_quote_max_age_seconds: 1200\n"
        "  cancel_unfilled_order_seconds: 60\n"
        "  max_limit_chase_pct: 5\n"
        "  kill_switch_file: runtime/HALT\n"
        "  no_entry_minutes_after_open: 15\n"
        "  max_daily_llm_tokens: 5000000\n",
        encoding="utf-8",
    )
    result = state.get_config_mandate(tmp_path, mode="paper")
    assert result["ok"] is True
    assert result["data"]["account"]["initial_capital_usd"] == 1000


def test_execute_halt(tmp_path: Path) -> None:
    state._clear_cache()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    result = state.execute_halt(tmp_path, "test halt")
    assert result["ok"] is True
    assert result["data"]["halted"] is True
    assert (tmp_path / "runtime" / "HALT").exists()


def test_execute_resume(tmp_path: Path) -> None:
    state._clear_cache()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "HALT").write_text("test\n", encoding="utf-8")
    result = state.execute_resume(tmp_path)
    assert result["ok"] is True
    assert result["data"]["halted"] is False
    assert not (tmp_path / "runtime" / "HALT").exists()


# --- Server handler tests ---


def test_server_routes(tmp_path: Path) -> None:
    from trading_agent.web_dashboard.server import make_handler

    handler_cls = make_handler(tmp_path, mode="paper")
    assert handler_cls.root_dir == tmp_path
    assert handler_cls.mode == "paper"


# --- Integration tests: start server, hit all API endpoints ---


def test_server_starts_and_serves_api(tmp_path: Path) -> None:
    """Start the server on a random port and hit every /api/* endpoint."""
    import socket
    import threading
    import time
    from urllib.request import urlopen
    from urllib.error import URLError

    from trading_agent.web_dashboard.server import serve

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # Start server in background
    server_thread = threading.Thread(
        target=serve, args=(tmp_path,), kwargs={"port": port}, daemon=True
    )
    server_thread.start()
    time.sleep(0.5)

    base = f"http://127.0.0.1:{port}"

    # Test all API endpoints
    endpoints = [
        "/",
        "/api/health",
        "/api/status",
        "/api/positions",
        "/api/portfolio-risk",
        "/api/report/today",
        "/api/funnel/today",
        "/api/calibration/today",
        "/api/audit/recent?limit=5",
        "/api/audit/24h?limit=5",
        "/api/iv",
        "/api/config/mandate",
    ]

    for ep in endpoints:
        try:
            with urlopen(f"{base}{ep}", timeout=5) as resp:
                assert resp.status == 200, f"{ep} returned {resp.status}"
                if ep.startswith("/api/"):
                    data = json.loads(resp.read())
                    assert "ok" in data, f"{ep} missing 'ok' field"
        except URLError as exc:
            raise AssertionError(f"Failed to reach {ep}: {exc}")


def test_server_handles_missing_runtime(tmp_path: Path) -> None:
    """Server should not crash when runtime files don't exist."""
    import socket
    import threading
    import time
    from urllib.request import urlopen

    from trading_agent.web_dashboard.server import serve

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server_thread = threading.Thread(
        target=serve, args=(tmp_path,), kwargs={"port": port}, daemon=True
    )
    server_thread.start()
    time.sleep(0.5)

    base = f"http://127.0.0.1:{port}"

    # All endpoints should return 200 with ok:true even with no runtime data
    for ep in ["/api/status", "/api/positions", "/api/portfolio-risk", "/api/audit/recent"]:
        with urlopen(f"{base}{ep}", timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["ok"] is True


def test_server_halt_resume_via_api(tmp_path: Path) -> None:
    """Test HALT/resume write operations via the API."""
    import socket
    import threading
    import time
    import urllib.error
    from urllib.request import urlopen, Request

    from trading_agent.web_dashboard.server import serve

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server_thread = threading.Thread(
        target=serve, args=(tmp_path,), kwargs={"port": port}, daemon=True
    )
    server_thread.start()
    time.sleep(0.5)

    base = f"http://127.0.0.1:{port}"

    # Halt without confirmation should return 400
    req = Request(f"{base}/api/halt", method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        urlopen(req, data=json.dumps({}).encode(), timeout=5)
        raise AssertionError("Expected 400 for unconfirmed halt")
    except urllib.error.HTTPError as e:
        assert e.code == 400

    # Halt with confirmation
    req = Request(f"{base}/api/halt", method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, data=json.dumps({"confirm": True, "reason": "test"}).encode(), timeout=5) as resp:
        data = json.loads(resp.read())
        assert data["ok"] is True

    # Status should show halted
    with urlopen(f"{base}/api/status", timeout=5) as resp:
        data = json.loads(resp.read())
        assert data["data"]["halted"] is True

    # Resume
    req = Request(f"{base}/api/resume", method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, data=json.dumps({"confirm": True}).encode(), timeout=5) as resp:
        data = json.loads(resp.read())
        assert data["ok"] is True

    # Status should show not halted
    with urlopen(f"{base}/api/status", timeout=5) as resp:
        data = json.loads(resp.read())
        assert data["data"]["halted"] is False


def test_server_requires_web_token_when_configured(tmp_path: Path) -> None:
    import socket
    import threading
    import time
    import urllib.error
    from urllib.request import urlopen, Request

    from trading_agent.web_dashboard.server import serve

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server_thread = threading.Thread(
        target=serve,
        args=(tmp_path,),
        kwargs={"port": port, "web_token": "secret-token"},
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)

    base = f"http://127.0.0.1:{port}"

    try:
        urlopen(f"{base}/api/status", timeout=5)
        raise AssertionError("Expected 401 without token")
    except urllib.error.HTTPError as e:
        assert e.code == 401

    req = Request(f"{base}/api/status")
    req.add_header("X-Trading-Agent-Web-Token", "secret-token")
    with urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        assert data["ok"] is True

    req = Request(f"{base}/api/halt", method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Trading-Agent-Web-Token", "secret-token")
    with urlopen(
        req,
        data=json.dumps({"confirm": True, "reason": "token test"}).encode(),
        timeout=5,
    ) as resp:
        data = json.loads(resp.read())
        assert data["ok"] is True
