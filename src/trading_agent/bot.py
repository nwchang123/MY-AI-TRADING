from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trading_agent.domain.calendar import is_market_hours, market_date
from trading_agent.domain.risk import Mandate
from trading_agent.reporting import build_daily_report, read_audit_events
from trading_agent.settings import Settings
from trading_agent.storage.audit import AuditWriter
from trading_agent.storage.budget import DailyTokenBudget
from trading_agent.storage.positions import PositionStore

# Operator control bot. Design rule, mirroring the trading side's "LLM
# proposes, code disposes": ACTIONS happen only through exact deterministic
# commands; the AI layer is READ-ONLY Q&A over system state and can never
# execute anything. Only the operator's chat id is answered.

_API = "https://api.telegram.org/bot{token}/{method}"

# Daily token ceiling for bot Q&A, separate from the trading committee budget.
BOT_LLM_DAILY_TOKENS = 50_000

_HELP = """可用命令（动作只认命令，AI 无执行权）：
/status — 系统状态（心跳/HALT/持仓/权益）
/run — 立即启动今天的交易循环
/halt 原因 — 紧急停机（写入 HALT 开关）
/resume — 解除停机
/positions — 当前持仓明细
/report — 今日交易报告
/help — 本帮助
其他消息会由 AI 助手根据系统状态回答（只读）。"""

_AI_SYSTEM = (
    "你是一个交易代理系统的运维助手。仅根据提供的系统状态上下文回答操作者的问题，"
    "用华语，简短直接（最多5句）。你没有任何执行能力：如果操作者想执行动作，"
    "提示对应命令（/run /halt /resume /status /positions /report）。"
    "不知道的事情就说不知道，不要编造。"
)


class TelegramBot:
    def __init__(
        self,
        settings: Settings,
        *,
        llm_client: Any | None = None,
        http_fn: Callable[[str, dict[str, Any] | None], dict[str, Any]] | None = None,
        spawn_fn: Callable[[], None] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            raise ValueError("bot requires TRADING_AGENT_TELEGRAM_BOT_TOKEN/_CHAT_ID")
        self.settings = settings
        self.llm = llm_client
        self._http = http_fn or self._http_request
        self._spawn = spawn_fn or self._spawn_loop
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.offset = 0
        self.llm_budget = DailyTokenBudget(
            settings.root_dir / "runtime" / "llm_budget.bot.json",
            BOT_LLM_DAILY_TOKENS,
        )

    # --- plumbing --------------------------------------------------------
    def _http_request(self, method: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        url = _API.format(token=self.settings.telegram_bot_token, method=method)
        data = json.dumps(payload or {}).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=70) as response:
            return json.load(response)

    def send(self, text: str) -> None:
        try:
            self._http(
                "sendMessage",
                {"chat_id": self.settings.telegram_chat_id, "text": text},
            )
        except Exception:  # noqa: BLE001 - replies are best-effort
            pass

    # --- state helpers ----------------------------------------------------
    def _runtime(self) -> Path:
        return self.settings.root_dir / "runtime"

    def _halt_path(self) -> Path:
        mandate = Mandate.load(self.settings.mandate_path)
        return self.settings.root_dir / mandate.execution.kill_switch_file

    def _store(self) -> PositionStore:
        return PositionStore(self._runtime() / f"positions.{self.settings.mode}.sqlite")

    def _audit(self) -> AuditWriter:
        return AuditWriter(self._runtime() / "audit.jsonl")

    def _status_text(self) -> str:
        now = self._now()
        lines = [f"【{'模拟盘' if self.settings.mode == 'paper' else '实盘'}】系统状态"]

        halt = self._halt_path()
        if halt.exists():
            reason = halt.read_text(encoding="utf-8").strip() or "(无原因)"
            lines.append(f"⛔ HALT 停机中：{reason}")
        else:
            lines.append("🟢 HALT 未激活")

        heartbeat = self._runtime() / "heartbeat.json"
        if heartbeat.exists():
            try:
                beat = json.loads(heartbeat.read_text(encoding="utf-8"))
                at = datetime.fromisoformat(beat["at"])
                age_min = (now - at).total_seconds() / 60.0
                lines.append(f"心跳：{age_min:.0f} 分钟前（{beat.get('note', '')}）")
            except (ValueError, KeyError):
                lines.append("心跳文件无法解析")
        else:
            lines.append("心跳：从未运行")

        lines.append("市场：" + ("开盘中" if is_market_hours(now) else "闭市"))

        rows = self._store().all_positions()
        open_rows = [r for r in rows if r["status"] == "open"]
        realized = sum(
            (r["exit_price"] - r["entry_price"]) * r["contracts"] * r["lot_size"]
            for r in rows
            if r["status"] == "closed" and r["exit_price"] is not None
        )
        mandate = Mandate.load(self.settings.mandate_path)
        equity = mandate.account.initial_capital_usd + realized
        lines.append(
            f"持仓 {len(open_rows)} 个 | 已实现盈亏 ${realized:+.2f} | 权益 ${equity:.2f}"
        )
        return "\n".join(lines)

    def _positions_text(self) -> str:
        open_rows = self._store().open_positions()
        if not open_rows:
            return "当前没有持仓。"
        lines = ["当前持仓："]
        for r in open_rows:
            lines.append(
                f"{r['option_code']}（{r['ticker']} {r['option_side']}）"
                f" 入价 {r['entry_price']} × {r['contracts']} 张"
                f"，止盈 +{r['take_profit_pct']:.0f}% 止损 -{r['stop_loss_pct']:.0f}%"
                f"，时间止损 {r['time_stop']}"
            )
        return "\n".join(lines)

    def _report_text(self) -> str:
        events = read_audit_events(self._runtime() / "audit.jsonl")
        report = build_daily_report(events, market_date(self._now()))
        return "今日报告：\n" + json.dumps(report, ensure_ascii=False, indent=1)[:3500]

    def _loop_running(self) -> bool:
        heartbeat = self._runtime() / "heartbeat.json"
        try:
            beat = json.loads(heartbeat.read_text(encoding="utf-8"))
            at = datetime.fromisoformat(beat["at"])
        except (FileNotFoundError, ValueError, KeyError):
            return False
        return (self._now() - at).total_seconds() < 45 * 60

    def _spawn_loop(self) -> None:
        script = self.settings.root_dir / "scripts" / "run_paper_loop.ps1"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
            ],
            cwd=str(self.settings.root_dir),
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # --- command handling ---------------------------------------------------
    def handle_text(self, text: str) -> str:
        text = (text or "").strip()
        lowered = text.lower()

        if lowered in {"/start", "/help"}:
            return _HELP
        if lowered == "/status":
            return self._status_text()
        if lowered == "/positions":
            return self._positions_text()
        if lowered == "/report":
            return self._report_text()
        if lowered.startswith("/halt"):
            reason = text[5:].strip() or "operator halt via telegram"
            halt = self._halt_path()
            halt.parent.mkdir(parents=True, exist_ok=True)
            halt.write_text(f"{reason}\n", encoding="utf-8")
            self._audit().append(
                "kill_switch_activated", {"reason": reason, "via": "telegram"}
            )
            return f"⛔ 已写入 HALT：{reason}\n所有周期将在开始前中止。用 /resume 解除。"
        if lowered == "/resume":
            halt = self._halt_path()
            was = halt.exists()
            halt.unlink(missing_ok=True)
            self._audit().append(
                "kill_switch_cleared", {"was_active": was, "via": "telegram"}
            )
            return "🟢 HALT 已解除，交易恢复。" if was else "HALT 本来就没有激活。"
        if lowered == "/run":
            if self._loop_running():
                return "循环已在运行（心跳很新鲜），不重复启动。"
            try:
                self._spawn()
            except Exception as exc:  # noqa: BLE001
                return f"启动失败：{exc}"
            note = "" if is_market_hours(self._now()) else "\n（当前闭市，循环会等到开盘才交易）"
            return "🟢 交易循环已启动。" + note
        if text.startswith("/"):
            return "不认识这个命令。\n" + _HELP

        return self._ai_answer(text)

    def _ai_answer(self, question: str) -> str:
        if self.llm is None:
            return "AI 助手未配置。动作请用命令：\n" + _HELP
        today = market_date(self._now())
        if self.llm_budget.remaining(today) <= 0:
            return "今天 AI 问答的 token 预算用完了，明天恢复。命令仍然可用。"
        context = self._status_text() + "\n\n" + self._positions_text()
        try:
            before = getattr(self.llm, "usage", None)
            before_total = before.total_tokens if before is not None else 0
            answer = self.llm.complete(
                system=_AI_SYSTEM,
                user=f"系统状态：\n{context}\n\n操作者的问题：{question}",
            )
            after = getattr(self.llm, "usage", None)
            if after is not None:
                self.llm_budget.add(after.total_tokens - before_total, today)
            return answer.strip()[:3500]
        except Exception as exc:  # noqa: BLE001
            return f"AI 回答失败：{exc}\n命令仍然可用（/help）。"

    # --- polling loop ---------------------------------------------------------
    def poll_once(self) -> int:
        """One long-poll pass; returns how many messages were handled."""

        response = self._http(
            "getUpdates", {"offset": self.offset, "timeout": 50}
        )
        handled = 0
        for update in response.get("result") or []:
            self.offset = max(self.offset, int(update.get("update_id", 0)) + 1)
            message = update.get("message") or {}
            chat_id = str((message.get("chat") or {}).get("id") or "")
            text = message.get("text") or ""
            if chat_id != str(self.settings.telegram_chat_id):
                continue  # only the operator may talk to this system
            if not text:
                continue
            self.send(self.handle_text(text))
            handled += 1
        return handled


def run_bot(settings: Settings) -> None:
    """Long-polling listener; restarts its poll on transient failures."""

    from trading_agent.research.llm import OpenAICompatibleClient

    llm = None
    if settings.llm_api_key:
        llm = OpenAICompatibleClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=45,
        )
    bot = TelegramBot(settings, llm_client=llm)
    bot.send("🤖 控制台已上线。发送 /help 查看命令；其他消息由 AI 回答（只读）。")
    import time as _time

    while True:
        try:
            bot.poll_once()
        except KeyboardInterrupt:
            bot.send("🤖 控制台已下线。")
            raise
        except Exception as exc:  # noqa: BLE001 - keep listening
            print(f"bot poll error: {exc}", file=sys.stderr)
            _time.sleep(5)
