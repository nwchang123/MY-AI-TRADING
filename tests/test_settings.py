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

