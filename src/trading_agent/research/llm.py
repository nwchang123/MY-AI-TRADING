from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Raised when the LLM provider rejects a request or returns no content."""


class LLMClient(Protocol):
    """Minimal chat interface the committee depends on.

    Implementations must be provider-agnostic so the committee can run against a
    real OpenAI-compatible endpoint or a deterministic mock in tests.
    """

    def complete(self, *, system: str, user: str, json_mode: bool = False) -> str: ...


class OpenAICompatibleClient:
    """Chat client for any OpenAI-compatible endpoint (DeepSeek, OpenAI, ...).

    The SDK import is lazy so the package and its tests do not require the
    ``openai`` dependency unless a live model call is actually made.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise LLMError("LLM API key is not configured")
        self.model = model
        self.temperature = temperature
        self._client = self._build_client(api_key, base_url, timeout)

    @staticmethod
    def _build_client(api_key: str, base_url: str, timeout: float) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without openai
            raise LLMError(
                "The 'openai' package is required for live LLM calls. "
                "Install it with: python -m pip install -e ."
            ) from exc
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize all provider errors
            raise LLMError(f"LLM request failed: {exc}") from exc
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError("LLM returned empty content")
        return content


class MockLLMClient:
    """Deterministic client for tests and offline pipeline verification.

    Pass either an iterable of canned responses (returned in order) or a callable
    that receives ``(system, user, json_mode)`` and returns the response string.
    """

    def __init__(
        self,
        responses: Iterable[str] | None = None,
        *,
        handler: Callable[[str, str, bool], str] | None = None,
    ):
        if (responses is None) == (handler is None):
            raise ValueError("Provide exactly one of 'responses' or 'handler'")
        self._queue = list(responses) if responses is not None else []
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if self._handler is not None:
            return self._handler(system, user, json_mode)
        if not self._queue:
            raise LLMError("MockLLMClient ran out of scripted responses")
        return self._queue.pop(0)
