import pytest

from trading_agent.research.llm import LLMError, MockLLMClient


def test_mock_returns_scripted_responses_in_order() -> None:
    client = MockLLMClient(["first", "second"])
    assert client.complete(system="s", user="u") == "first"
    assert client.complete(system="s", user="u") == "second"
    assert len(client.calls) == 2


def test_mock_raises_when_exhausted() -> None:
    client = MockLLMClient(["only"])
    client.complete(system="s", user="u")
    with pytest.raises(LLMError):
        client.complete(system="s", user="u")


def test_mock_handler_sees_json_mode() -> None:
    client = MockLLMClient(handler=lambda system, user, json_mode: "json" if json_mode else "text")
    assert client.complete(system="s", user="u") == "text"
    assert client.complete(system="s", user="u", json_mode=True) == "json"


def test_mock_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        MockLLMClient()
    with pytest.raises(ValueError):
        MockLLMClient(["x"], handler=lambda s, u, j: "y")
