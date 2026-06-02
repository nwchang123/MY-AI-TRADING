from pathlib import Path

import pytest

from trading_agent.execution.live import live_position_cap
from trading_agent.settings import LIVE_ACKNOWLEDGMENT, Settings


def test_ramp_caps_to_one_until_ten_trades() -> None:
    assert live_position_cap(2, 0) == 1
    assert live_position_cap(2, 9) == 1
    assert live_position_cap(2, 10) == 2
    # Never exceeds the mandate maximum.
    assert live_position_cap(1, 50) == 1


def _live_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "root_dir": tmp_path,
        "mode": "live",
        "moomoo_host": "127.0.0.1",
        "moomoo_port": 11111,
        "security_firm": "FUTUMY",
        "account_id": 555,
        "live_acknowledgment": LIVE_ACKNOWLEDGMENT,
        "live_account_allowlist": (555,),
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_live_requires_allowlist(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path, live_account_allowlist=())
    with pytest.raises(RuntimeError, match="allowlist"):
        settings.assert_live_startup_allowed()


def test_live_rejects_account_not_in_allowlist(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path, account_id=999, live_account_allowlist=(555,))
    with pytest.raises(RuntimeError, match="not in the live allowlist"):
        settings.assert_live_startup_allowed()


def test_live_allows_pinned_allowlisted_account(tmp_path: Path) -> None:
    _live_settings(tmp_path).assert_live_startup_allowed()  # no raise


def test_allowlist_parsed_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRADING_AGENT_LIVE_ACCOUNT_ALLOWLIST", "111, 222 ,333")
    settings = Settings.from_env(tmp_path)
    assert settings.live_account_allowlist == (111, 222, 333)
