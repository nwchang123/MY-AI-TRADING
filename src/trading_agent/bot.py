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

# Dead-man's switch: the run-loop touches heartbeat.json every tick (~30 min),
# even on skipped cycles. If the market is OPEN and the heartbeat is older than
# this, the loop is dead or never started -- and since the loop announces its
# own start/stop, a loop that never starts produces NO message at all. This is
# the alarm for that silence. Re-alerts at most once per hour.
DEADMAN_STALE_SECONDS = 45 * 60
DEADMAN_REALERT_SECONDS = 60 * 60

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
    "你是一个自主期权交易代理系统的分析助手，用华语回答操作者的问题。"
    "你能看到：系统状态、当前持仓、最近的交易记录（进出价格、盈亏、平仓原因），"
    "以及审计日志摘要——里面有 AI 委员会每次决策的理由、双 AI 的胜率估计、"
    "蒙特卡洛基线概率、被拒绝的提案和原因。"
    "回答『交易了什么』『为什么买』『为什么亏』这类问题时，引用这些真实数据："
    "比如用入场论点+平仓原因+价格变化解释一笔亏损。数据里没有的就直说不知道，"
    "绝不编造。保持简短（最多8句）。"
    "你没有任何执行能力：操作者想执行动作时，提示对应命令"
    "（/run /halt /resume /status /positions /report）。"
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
        # Short in-memory chat history so follow-up questions ("那为什么亏?")
        # keep their referent. Lost on restart by design.
        self.history: list[tuple[str, str]] = []
        self._last_deadman_alert: datetime | None = None

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

    def _trades_text(self, limit: int = 15) -> str:
        rows = self._store().all_positions()
        closed = [
            r
            for r in rows
            if r["status"] == "closed" and r["exit_price"] is not None
        ]
        closed.sort(key=lambda r: str(r.get("closed_at") or ""))
        reconciled = sum(
            1 for r in rows if r["status"] == "closed" and r["exit_price"] is None
        )
        if not closed and not reconciled:
            return "最近交易记录：还没有任何已平仓的交易。"
        lines = ["最近交易记录（已平仓，按时间）："]
        for r in closed[-limit:]:
            pnl = (r["exit_price"] - r["entry_price"]) * r["contracts"] * r["lot_size"]
            lines.append(
                f"{str(r.get('closed_at') or '')[:10]} {r['option_code']}"
                f"（{r['ticker']} {r['option_side']}）"
                f" 入 {r['entry_price']} → 出 {r['exit_price']}"
                f" 盈亏 ${pnl:+.2f} 平仓原因: {r.get('close_reason')}"
            )
        if reconciled:
            lines.append(f"另有 {reconciled} 笔对账平仓（无成交价记录）。")
        return "\n".join(lines)

    def _audit_digest(self, limit: int = 50) -> str:
        """Compact recent-decision trail for the Q&A context.

        Committee rationales and probability estimates are what let the AI
        answer "why did it buy/lose" with the system's own reasoning instead
        of guessing.
        """

        events = read_audit_events(self._runtime() / "audit.jsonl")
        lines: list[str] = []
        for event in events[-400:]:
            etype = event.get("event_type", "")
            payload = event.get("payload", {}) or {}
            at = str(event.get("recorded_at", ""))[:16]
            if etype in {"committee_run", "committee_cache_hit"}:
                output = payload.get("output", {}) or {}
                rationale = str(output.get("rationale", ""))[:180]
                wp = output.get("win_probability")
                tag = "委员会" if etype == "committee_run" else "委员会(缓存)"
                lines.append(
                    f"{at} {tag}[{payload.get('ticker')}] {payload.get('decision')}"
                    + (f" 胜率估计={wp}" if wp is not None else "")
                    + (f" 理由: {rationale}" if rationale else "")
                )
            elif etype == "monte_carlo_pop":
                lines.append(
                    f"{at} 蒙特卡洛[{payload.get('option_code')}]"
                    f" 基线概率={payload.get('pop')}（门槛 {payload.get('floor')}）"
                )
            elif etype == "order_filled":
                lines.append(
                    f"{at} 成交 {payload.get('option_code')}"
                    f" {payload.get('side')} @ {payload.get('price')}"
                )
            elif etype == "position_closed":
                lines.append(
                    f"{at} 平仓 {payload.get('option_code')}"
                    f" 原因: {payload.get('reason')}"
                    f" 盈亏 ${payload.get('realized_pnl_usd')}"
                )
            elif etype in {
                "universe_selected",
                "no_eligible_contracts",
                "entries_skipped",
                "proposal_rejected",
                "order_cancelled",
                "position_adopted",
                "circuit_breaker_tripped",
                "kill_switch_activated",
                "kill_switch_cleared",
                "cycle_crashed",
                "llm_budget_exhausted",
                "risk_caps_scaled",
            }:
                detail = json.dumps(payload, ensure_ascii=False)[:140]
                lines.append(f"{at} {etype}: {detail}")
        if not lines:
            return "审计日志：还没有任何决策记录。"
        return "审计日志摘要（最近的决策轨迹）：\n" + "\n".join(lines[-limit:])

    def _qa_context(self) -> str:
        return "\n\n".join(
            [
                self._status_text(),
                self._positions_text(),
                self._trades_text(),
                self._audit_digest(),
            ]
        )

    def _report_text(self) -> str:
        events = read_audit_events(self._runtime() / "audit.jsonl")
        report = build_daily_report(events, market_date(self._now()))
        return "今日报告：\n" + json.dumps(report, ensure_ascii=False, indent=1)[:3500]

    def _heartbeat_age_seconds(self) -> float | None:
        """Seconds since the loop last touched heartbeat.json; None if absent."""

        heartbeat = self._runtime() / "heartbeat.json"
        try:
            beat = json.loads(heartbeat.read_text(encoding="utf-8"))
            at = datetime.fromisoformat(beat["at"])
        except (FileNotFoundError, ValueError, KeyError):
            return None
        return (self._now() - at).total_seconds()

    def _loop_running(self) -> bool:
        age = self._heartbeat_age_seconds()
        return age is not None and age < DEADMAN_STALE_SECONDS

    def maybe_alert_dead_loop(self) -> str | None:
        """Dead-man's switch: alert when the market is open but the loop is silent.

        Returns the alert text when one was sent (for tests), else None. The
        HALT switch does not suppress this -- a halted loop still heartbeats on
        every skipped tick, so a stale heartbeat always means a dead process.
        """

        now = self._now()
        if not is_market_hours(now):
            return None
        age = self._heartbeat_age_seconds()
        if age is not None and age < DEADMAN_STALE_SECONDS:
            return None
        if (
            self._last_deadman_alert is not None
            and (now - self._last_deadman_alert).total_seconds() < DEADMAN_REALERT_SECONDS
        ):
            return None
        self._last_deadman_alert = now
        detail = "心跳文件不存在" if age is None else f"心跳已 {age / 60:.0f} 分钟没有更新"
        mode_label = "模拟盘" if self.settings.mode == "paper" else "实盘"
        text = (
            f"【{mode_label}】🚨 死人开关报警：美股开市中，但交易循环{detail}。\n"
            "循环可能没有启动或已挂死。用 /run 启动，/status 查看详情。"
        )
        self._audit().append(
            "deadman_alert", {"heartbeat_age_seconds": age, "at": now.isoformat()}
        )
        self.send(text)
        return text

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
        context = self._qa_context()
        history = "\n".join(
            f"操作者: {q}\n助手: {a}" for q, a in self.history[-4:]
        )
        history_block = f"\n\n[之前的对话]\n{history}" if history else ""
        try:
            before = getattr(self.llm, "usage", None)
            before_total = before.total_tokens if before is not None else 0
            answer = self.llm.complete(
                system=_AI_SYSTEM,
                user=(
                    f"[系统数据]\n{context}{history_block}\n\n"
                    f"操作者的问题：{question}"
                ),
            )
            after = getattr(self.llm, "usage", None)
            if after is not None:
                self.llm_budget.add(after.total_tokens - before_total, today)
            answer = answer.strip()[:3500]
            self.history.append((question, answer))
            del self.history[:-8]
            return answer
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
            # Dead-man's switch rides the poll cadence (one long-poll ~50s):
            # market open + stale heartbeat -> alert the operator.
            bot.maybe_alert_dead_loop()
        except KeyboardInterrupt:
            bot.send("🤖 控制台已下线。")
            raise
        except Exception as exc:  # noqa: BLE001 - keep listening
            print(f"bot poll error: {exc}", file=sys.stderr)
            _time.sleep(5)
