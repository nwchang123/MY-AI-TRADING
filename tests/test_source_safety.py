from pathlib import Path


def _all_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in Path("src").rglob("*.py")
    )


def test_source_does_not_contain_sdk_trade_unlock_calls() -> None:
    # The operator unlocks live trading manually in the OpenD GUI; the agent
    # must never call unlock_trade() or store the trading password.
    assert "unlock_trade(" not in _all_source()


def test_real_trd_env_is_confined_to_the_broker_adapter() -> None:
    # Live (REAL) execution may only exist in the single broker module, mapped
    # from a "REAL"/"SIMULATE" string that callers must pass explicitly. This
    # keeps REAL from leaking into other modules where it could bypass the
    # operator startup gate (settings.assert_live_startup_allowed).
    offenders = [
        path.as_posix()
        for path in Path("src").rglob("*.py")
        if "TrdEnv.REAL" in path.read_text(encoding="utf-8")
        and path.name != "moomoo.py"
    ]
    assert offenders == []
