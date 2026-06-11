import pytest

from trading_agent.execution.orchestrator import CycleResult
from trading_agent.notify import TelegramNotifier, format_cycle_alert


def test_notifier_requires_token_and_chat() -> None:
    with pytest.raises(ValueError):
        TelegramNotifier("", "123")
    with pytest.raises(ValueError):
        TelegramNotifier("tok", "")


def test_send_posts_chat_id_and_text() -> None:
    calls: list[tuple[str, dict]] = []

    def fetch(url: str, payload: dict) -> dict:
        calls.append((url, payload))
        return {"ok": True}

    notifier = TelegramNotifier("tok", "42", fetch_fn=fetch)
    assert notifier.send("hello") is True
    url, payload = calls[0]
    assert "bottok" in url
    assert payload == {"chat_id": "42", "text": "hello"}


def test_send_swallows_failures() -> None:
    def fetch(url: str, payload: dict) -> dict:
        raise RuntimeError("network down")

    notifier = TelegramNotifier("tok", "42", fetch_fn=fetch)
    assert notifier.send("hello") is False  # never raises


def test_send_reports_api_rejection_as_false() -> None:
    notifier = TelegramNotifier("tok", "42", fetch_fn=lambda u, p: {"ok": False})
    assert notifier.send("hello") is False


def test_quiet_cycle_produces_no_alert() -> None:
    result = CycleResult()
    result.rejected.append({"ticker": "AAA", "stage": "no_eligible_contracts"})
    assert format_cycle_alert(result, mode="paper") is None


def test_actionable_cycle_is_formatted() -> None:
    result = CycleResult()
    result.entries.append({"ticker": "BULL", "option_code": "US.BULL260717C5000"})
    result.exits.append({"option_code": "US.KEEL260717C2500", "reason": "take profit"})
    result.errors.append({"stage": "exit_quote", "ref": "US.X", "error": "feed down"})
    text = format_cycle_alert(result, mode="paper")
    assert text is not None
    assert "[paper]" in text
    assert "OPENED US.BULL260717C5000 (BULL)" in text
    assert "CLOSED US.KEEL260717C2500: take profit" in text
    assert "error [exit_quote]" in text


def test_circuit_breaker_leads_the_alert() -> None:
    result = CycleResult(circuit_breaker="hard drawdown stop")
    text = format_cycle_alert(result, mode="paper")
    assert text is not None
    assert "CIRCUIT BREAKER: hard drawdown stop" in text
