import json
import sys
from pathlib import Path
from shutil import copyfile

import pytest

from trading_agent.cli import main

ROOT = Path(__file__).parents[1]


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
