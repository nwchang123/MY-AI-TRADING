from pathlib import Path

import pytest

from trading_agent.settings import LIVE_ACKNOWLEDGMENT, Settings


def test_live_mode_requires_explicit_acknowledgment(tmp_path: Path) -> None:
    settings = Settings(
        root_dir=tmp_path,
        mode="live",
        moomoo_host="127.0.0.1",
        moomoo_port=11111,
        security_firm="FUTUMY",
        account_id=123,
        live_acknowledgment="",
    )

    with pytest.raises(RuntimeError, match="Live mode is disabled"):
        settings.assert_live_startup_allowed()


def test_live_mode_requires_pinned_account(tmp_path: Path) -> None:
    settings = Settings(
        root_dir=tmp_path,
        mode="live",
        moomoo_host="127.0.0.1",
        moomoo_port=11111,
        security_firm="FUTUMY",
        account_id=None,
        live_acknowledgment=LIVE_ACKNOWLEDGMENT,
    )

    with pytest.raises(RuntimeError, match="ACCOUNT_ID"):
        settings.assert_live_startup_allowed()


def test_adversary_llm_config_parsed_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADING_AGENT_LLM_ADVERSARY_API_KEY", "advkey")
    monkeypatch.setenv("TRADING_AGENT_LLM_ADVERSARY_BASE_URL", "https://adv.example/v1")
    monkeypatch.setenv("TRADING_AGENT_LLM_ADVERSARY_MODEL", "gemini-2.0-flash")

    settings = Settings.from_env(tmp_path)

    assert settings.llm_adversary_api_key == "advkey"
    assert settings.llm_adversary_base_url == "https://adv.example/v1"
    assert settings.llm_adversary_model == "gemini-2.0-flash"


def test_adversary_llm_defaults_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "TRADING_AGENT_LLM_ADVERSARY_API_KEY",
        "TRADING_AGENT_LLM_ADVERSARY_BASE_URL",
        "TRADING_AGENT_LLM_ADVERSARY_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings.from_env(tmp_path)

    assert settings.llm_adversary_model == ""


def test_telegram_readonly_chat_ids_parsed_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADING_AGENT_TELEGRAM_READONLY_CHAT_IDS", "100, 200,,300")

    settings = Settings.from_env(tmp_path)

    assert settings.telegram_readonly_chat_ids == ("100", "200", "300")
