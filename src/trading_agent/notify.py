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


# Operator-facing alert text is Chinese by operator preference; audit logs and
# ledger strings stay English (they are data, matched by code and tests).
_MODE_LABELS = {"paper": "模拟盘", "live": "实盘"}

_BREAKER_LABELS = {
    "daily loss stop": "触发单日亏损上限",
    "hard drawdown stop": "触发硬回撤上限",
}

_EXIT_REASON_LABELS = {
    "take profit": "止盈",
    "stop loss": "止损",
    "time stop reached": "到达时间止损",
    "forced close before expiry": "临近到期强制平仓",
}


def format_cycle_alert(result: Any, *, mode: str) -> str | None:
    """Operator alert (Chinese) for a finished cycle; None when nothing notable.

    Notable means money moved or something needs the operator: entries, exits,
    a tripped circuit breaker, errors, or an unexpected HALT. Routine quiet
    cycles (only rejects/no candidates) return None so the channel stays
    readable across a 20-session run.
    """

    lines: list[str] = []
    breaker = getattr(result, "circuit_breaker", None)
    if breaker:
        label = _BREAKER_LABELS.get(breaker, breaker)
        lines.append(f"⛔ 熔断：{label}，已写入 HALT，交易暂停待人工复查")
    if getattr(result, "halted", False):
        lines.append("本周期跳过：HALT 停机开关处于激活状态")
    for entry in getattr(result, "entries", []) or []:
        lines.append(f"📈 开仓 {entry.get('option_code')}（{entry.get('ticker')}）")
    for exit_ in getattr(result, "exits", []) or []:
        reason = _EXIT_REASON_LABELS.get(str(exit_.get("reason")), exit_.get("reason"))
        lines.append(f"📉 平仓 {exit_.get('option_code')}：{reason}")
    for error in getattr(result, "errors", []) or []:
        lines.append(
            f"⚠️ 错误 [{error.get('stage')}] {error.get('ref')}：{error.get('error')}"
        )
    if not lines:
        return None
    mode_label = _MODE_LABELS.get(mode, mode)
    return f"【{mode_label}】交易代理\n" + "\n".join(lines)
