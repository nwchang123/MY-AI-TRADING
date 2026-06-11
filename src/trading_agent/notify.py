from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Best-effort operator alerts through a Telegram bot.

    Every failure is swallowed and reported as ``False`` on purpose: alerting
    is an observer, and a dead notifier must never break, delay, or retry-loop
    a trading cycle. Security note: the bot token grants control of the bot
    only (not the brokerage); it lives in the local .env.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        timeout: float = 10.0,
        fetch_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ):
        if not token or not chat_id:
            raise ValueError("TelegramNotifier requires a bot token and chat id")
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self._fetch = fetch_fn or self._http_post

    def _http_post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def send(self, text: str) -> bool:
        """Send a plain-text message; returns False instead of raising."""

        try:
            result = self._fetch(
                _API_URL.format(token=self.token),
                {"chat_id": self.chat_id, "text": text},
            )
            return bool(result.get("ok"))
        except Exception:  # noqa: BLE001 - alerting must never break trading
            return False


def format_cycle_alert(result: Any, *, mode: str) -> str | None:
    """Human-readable alert for a finished cycle, or None when nothing notable.

    Notable means money moved or something needs the operator: entries, exits,
    a tripped circuit breaker, errors, or an unexpected HALT. Routine quiet
    cycles (only rejects/no candidates) return None so the channel stays
    readable across a 20-session run.
    """

    lines: list[str] = []
    if getattr(result, "circuit_breaker", None):
        lines.append(f"CIRCUIT BREAKER: {result.circuit_breaker} -> HALT written")
    if getattr(result, "halted", False):
        lines.append("cycle skipped: HALT kill switch is active")
    for entry in getattr(result, "entries", []) or []:
        lines.append(f"OPENED {entry.get('option_code')} ({entry.get('ticker')})")
    for exit_ in getattr(result, "exits", []) or []:
        lines.append(f"CLOSED {exit_.get('option_code')}: {exit_.get('reason')}")
    for error in getattr(result, "errors", []) or []:
        lines.append(f"error [{error.get('stage')}] {error.get('ref')}: {error.get('error')}")
    if not lines:
        return None
    return f"[{mode}] trading agent\n" + "\n".join(lines)
