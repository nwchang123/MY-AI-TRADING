import json
import sys
from datetime import date
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import pytest

from trading_agent.cli import (
    _adversary_client,
    _build_committee,
    _format_session_summary,
    _update_dashboard,
    main,
)
from trading_agent.settings import Settings
from trading_agent.storage.positions import PositionStore

ROOT = Path(__file__).parents[1]


def test_format_session_summary_folds_a_session_into_one_line() -> None:
    line = _format_session_summary(
        ran=113,
        totals={"entries": 0, "exits": 2, "errors": 1, "crashes": 9},
        rejected_by_stage={"pre_screen": 206, "llm_budget": 1977, "exit_quote": 6},
        used_tokens=5_027_787,
        cap_tokens=5_000_000,
    )
    assert "113 cycles" in line
    assert "entries 0 | exits 2" in line
    # stages sorted most-frequent-first; tokens show used/cap and over-budget %.
    assert "llm_budget:1977, pre_screen:206, exit_quote:6" in line
    assert "5,027,787/5,000,000 (101%)" in line
    assert "crashes 9" in line


def test_format_session_summary_handles_no_rejections_and_missing_tokens() -> None:
    line = _format_session_summary(
        ran=0,
        totals={"entries": 0, "exits": 0, "errors": 0, "crashes": 0},
        rejected_by_stage={},
        used_tokens=None,
        cap_tokens=0,
    )
    assert "rejected{none}" in line
    assert "tokens n/a" in line


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        root_dir=tmp_path,
        mode="paper",
        moomoo_host="127.0.0.1",
        moomoo_port=11111,
        security_firm="FUTUMY",
        account_id=None,
        live_acknowledgment="",
        llm_api_key="primary-key",
        llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-v4-flash",
        llm_model_pro="deepseek-v4-pro",
    )
    base.update(overrides)
    return Settings(**base)


def test_adversary_client_is_none_without_model(tmp_path: Path) -> None:
    assert _adversary_client(_settings(tmp_path)) is None


def test_adversary_client_built_when_model_set(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        llm_adversary_model="gemini-2.0-flash",
        llm_adversary_base_url="https://adv.example/v1",
        llm_adversary_api_key="adv-key",
    )
    client = _adversary_client(settings)
    assert client is not None
    assert client.model == "gemini-2.0-flash"


def test_build_committee_wires_distinct_adversary(tmp_path: Path) -> None:
    # _build_committee reads the mandate (win-probability floor), so the
    # settings must point at a root that has config/mandate.paper.yaml.
    committee = _build_committee(
        _settings(tmp_path, root_dir=ROOT, llm_adversary_model="gemini-2.0-flash")
    )
    assert committee.adversary_client is not committee.client
    assert committee.adversary_client.model == "gemini-2.0-flash"


def test_build_committee_falls_back_without_adversary(tmp_path: Path) -> None:
    committee = _build_committee(_settings(tmp_path, root_dir=ROOT))
    assert committee.adversary_client is committee.client
    # The mandate's win-probability floor is threaded into the committee.
    assert committee.min_win_probability == 0.55
    # The soft-veto penalty is threaded through too (paper mandate sets 0.02).
    assert committee.veto_win_prob_penalty == 0.02


def test_update_dashboard_accepts_dict_position_rows(
    tmp_path: Path, capsys
) -> None:
    runtime = tmp_path / "runtime"
    store = PositionStore(runtime / "positions.paper.sqlite")
    store.open_position(
        option_code="US.EXAMPLE260626C5000",
        ticker="EXAMPLE",
        option_side="call",
        entry_price=0.2,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 6, 26),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 6, 24),
    )
    result = SimpleNamespace(
        exits=[{"option_code": "US.OLD260626P5000", "reason": "take profit"}],
        entries=[{"ticker": "EXAMPLE", "option_code": "US.EXAMPLE260626C5000"}],
    )

    _update_dashboard(_settings(tmp_path), result)

    assert "dashboard update failed" not in capsys.readouterr().err
    html = (runtime / "dashboard.html").read_text(encoding="utf-8")
    assert "US.EXAMPLE260626C5000" in html


def test_notify_web_dashboard_command_sends_link(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    sent: list[str] = []

    class FakeNotifier:
        def send(self, text: str) -> bool:
            sent.append(text)
            return True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("trading_agent.cli._notifier", lambda _settings: FakeNotifier())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trading-agent",
            "notify-web-dashboard",
            "--url",
            "http://127.0.0.1:8765/",
        ],
    )

    main()

    assert sent and "http://127.0.0.1:8765/" in sent[0]
    assert "web dashboard link sent" in capsys.readouterr().out
    record = json.loads(
        (tmp_path / "runtime" / "audit.jsonl").read_text(encoding="utf-8")
    )
    assert record["event_type"] == "web_dashboard_link_sent"
    assert record["payload"]["sent"] is True


def _offline_input(tmp_path: Path, *, bid: float = 0.19) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    copyfile(ROOT / "config" / "mandate.paper.yaml", config_dir / "mandate.paper.yaml")
    payload = json.loads(
        (ROOT / "examples" / "proposal-check.approved.json").read_text(encoding="utf-8")
    )
    payload["quote"]["bid"] = bid
    input_path = tmp_path / "proposal.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    return input_path


def test_halt_and_resume_are_audited(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["trading-agent", "halt", "--reason", "  smoke   test  "]
    )

    main()

    halt_path = tmp_path / "runtime" / "HALT"
    assert halt_path.read_text(encoding="utf-8") == "smoke test\n"
    assert capsys.readouterr().out == "Kill switch active: runtime/HALT\n"

    monkeypatch.setattr(sys, "argv", ["trading-agent", "resume"])
    main()

    assert halt_path.exists() is False
    records = [
        json.loads(line)
        for line in (tmp_path / "runtime" / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["event_type"] == "kill_switch_activated"
    assert records[0]["payload"] == {"reason": "smoke test"}
    assert records[1]["event_type"] == "kill_switch_cleared"
    assert records[1]["payload"] == {"was_active": True}


def test_proposal_check_approves_valid_offline_input(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    input_path = _offline_input(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_AGENT_MODE", "paper")
    monkeypatch.setattr(
        sys, "argv", ["trading-agent", "proposal-check", "--input", str(input_path)]
    )

    main()

    assert json.loads(capsys.readouterr().out)["approved"] is True
    record = json.loads(
        (tmp_path / "runtime" / "audit.jsonl").read_text(encoding="utf-8")
    )
    assert record["event_type"] == "proposal_checked"
    assert record["payload"]["decision"]["approved"] is True


def test_proposal_check_returns_nonzero_for_rejection(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    input_path = _offline_input(tmp_path, bid=0.01)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_AGENT_MODE", "paper")
    monkeypatch.setattr(
        sys, "argv", ["trading-agent", "proposal-check", "--input", str(input_path)]
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["approved"] is False
