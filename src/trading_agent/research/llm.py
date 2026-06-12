from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class LLMError(RuntimeError):
    """Raised when the LLM provider rejects a request or returns no content."""


class LLMUsage(BaseModel):
    """Running tally of LLM calls and token spend for one or more clients."""

    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "LLMUsage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


def usage_delta(before: LLMUsage, after: LLMUsage) -> LLMUsage:
    """Tokens/calls accrued between two usage snapshots (e.g. around one run)."""

    return LLMUsage(
        calls=after.calls - before.calls,
        prompt_tokens=after.prompt_tokens - before.prompt_tokens,
        completion_tokens=after.completion_tokens - before.completion_tokens,
    )


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
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        if not api_key:
            raise LLMError("LLM API key is not configured")
        self.model = model
        self.temperature = temperature
        self.max_retries = max(0, max_retries)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self.sleep_fn = sleep_fn or time.sleep
        self.usage = LLMUsage()
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

        # Retry transient failures (rate limits, 5xx, timeouts) with exponential
        # backoff so a brief provider hiccup does not abort the whole cycle.
        attempt = 0
        while True:
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - normalize all provider errors
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"LLM request failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                self.sleep_fn(self.retry_base_delay * (2**attempt))
                attempt += 1

        self.usage.calls += 1
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            self.usage.prompt_tokens += int(getattr(raw_usage, "prompt_tokens", 0) or 0)
            self.usage.completion_tokens += int(
                getattr(raw_usage, "completion_tokens", 0) or 0
            )

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
        self.usage = LLMUsage()

    def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        self.usage.calls += 1
        if self._handler is not None:
            return self._handler(system, user, json_mode)
        if not self._queue:
            raise LLMError("MockLLMClient ran out of scripted responses")
        return self._queue.pop(0)
