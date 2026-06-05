import pytest

from trading_agent.research.llm import LLMError, MockLLMClient, OpenAICompatibleClient


class _Resp:
    class _Usage:
        prompt_tokens = 11
        completion_tokens = 7

    class _Choice:
        class _Msg:
            content = "ok"

        message = _Msg()

    usage = _Usage()
    choices = [_Choice()]


class _FakeOpenAI:
    """Stand-in for the openai client: fails ``fail_times`` then returns a reply."""

    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.attempts = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError("transient 503")
        return _Resp()


def _client(fail_times: int = 0, sleeps: list | None = None) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(
        api_key="k",
        base_url="https://example/v1",
        model="m",
        sleep_fn=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )
    client._client = _FakeOpenAI(fail_times)
    return client


def test_records_token_usage_on_success() -> None:
    client = _client()
    assert client.complete(system="s", user="u") == "ok"
    assert client.usage.calls == 1
    assert client.usage.prompt_tokens == 11
    assert client.usage.completion_tokens == 7
    assert client.usage.total_tokens == 18


def test_retries_transient_failure_then_succeeds() -> None:
    sleeps: list = []
    client = _client(fail_times=2, sleeps=sleeps)  # max_retries=2 -> 3 attempts
    assert client.complete(system="s", user="u") == "ok"
    assert client._client.attempts == 3
    assert len(sleeps) == 2  # backed off twice
    assert client.usage.calls == 1


def test_raises_after_exhausting_retries() -> None:
    client = _client(fail_times=5, sleeps=[])
    with pytest.raises(LLMError, match="after 3 attempt"):
        client.complete(system="s", user="u")
    assert client.usage.calls == 0


def test_mock_tracks_usage_calls() -> None:
    client = MockLLMClient(["a", "b"])
    client.complete(system="s", user="u")
    client.complete(system="s", user="u")
    assert client.usage.calls == 2


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
