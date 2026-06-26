from __future__ import annotations

import json
import re
import secrets
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trading_agent.domain.calendar import is_market_hours, market_date
from trading_agent.domain.risk import Mandate
from trading_agent.execution.lock import CycleLockError, single_instance_lock
from trading_agent.reporting import (
    build_calibration_report,
    build_daily_report,
    build_funnel_report,
    build_portfolio_risk_report,
    read_audit_events,
)
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
/run — 准备启动交易循环（需确认）
/halt 原因 — 紧急停机（直接写入 HALT 开关）
/resume — 准备解除停机（需确认）
/positions — 当前持仓明细
/risk — 当前组合风险（Greeks/情景损益）
/funnel — 今日筛选到开仓漏斗
/calibration — 胜率估计校准
/report — 今日交易报告
/menu — 显示按钮菜单
/help — 本帮助
其他消息会由 AI 助手根据系统状态回答（只读）。"""

_AI_SYSTEM = (
    "你是一个自主期权交易代理系统的分析助手，用华语回答操作者的问题。"
    "你能看到：系统状态、当前持仓、组合风险、今日漏斗/校准、最近的交易记录"
    "（进出价格、盈亏、平仓原因），以及审计日志摘要——里面有 AI 委员会每次"
    "决策的理由、双 AI 的胜率估计、蒙特卡洛基线概率、被拒绝的提案和原因。"
    "回答『交易了什么』『为什么买』『为什么亏』这类问题时，引用这些真实数据："
    "比如用入场论点+平仓原因+价格变化解释一笔亏损。数据里没有的就直说不知道，"
    "绝不编造。保持简短（最多8句）。"
    "你没有任何执行能力：操作者想执行动作时，提示对应命令"
    "（/run /halt /resume /status /positions /risk /funnel /calibration /report）。"
)

_ACTION_CONFIRM_SECONDS = 5 * 60
_OPERATOR_ALERT_REALERT_SECONDS = 60 * 60
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
_COMMON_UPPER_WORDS = {
    "AI",
    "API",
    "DTE",
    "ETF",
    "HALT",
    "IV",
    "LLM",
    "MC",
    "P",
    "PL",
    "POP",
    "Q",
    "US",
}


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
        self.memory = self._load_memory()
        self.pending_actions: dict[str, dict[str, Any]] = {}
        self._next_reply_markup: dict[str, Any] | None = None
        self._current_update_meta: dict[str, Any] | None = None
        self._last_deadman_alert: datetime | None = None
        self._operator_alerts_sent: dict[str, datetime] = {}
        self._last_mandate: Mandate | None = None
        self._last_mandate_error: str | None = None

    # --- plumbing --------------------------------------------------------
    def _http_request(self, method: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        url = _API.format(token=self.settings.telegram_bot_token, method=method)
        data = json.dumps(payload or {}).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=70) as response:
            return json.load(response)

    def send(
        self,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        *,
        chat_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id or self.settings.telegram_chat_id,
            "text": text,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            self._http("sendMessage", payload)
        except Exception:  # noqa: BLE001 - replies are best-effort
            pass

    def _answer_callback(self, callback_id: str) -> None:
        if not callback_id:
            return
        try:
            self._http("answerCallbackQuery", {"callback_query_id": callback_id})
        except Exception:  # noqa: BLE001 - Telegram ack is best-effort
            pass

    # --- state helpers ----------------------------------------------------
    def _runtime(self) -> Path:
        return self.settings.root_dir / "runtime"

    def _mandate(self) -> Mandate:
        """Load the mandate, serving the last good one on a parse error.

        The bot reloads the mandate live (the kill-switch path rides every poll),
        so a config edit that adds a field the running bot's older code does not
        know yet raises pydantic ``extra_forbidden`` (the models are
        ``extra="forbid"``). Without a fallback that broke every poll -- 677x on
        2026-06-26 after a ``news_cache_ttl_hours`` edit. Keep serving the last
        successfully-loaded mandate so a deploy-time config/code skew degrades to
        "slightly stale config" instead of a dead listener; restarting the bot
        still adopts the new schema. A genuine bad config at startup (no last-good
        yet) still raises, so it fails loudly when there is nothing to fall back on.
        """
        try:
            mandate = Mandate.load(self.settings.mandate_path)
        except Exception as exc:  # noqa: BLE001 - keep the listener alive
            if self._last_mandate is None:
                raise
            message = str(exc)
            if message != self._last_mandate_error:
                self._last_mandate_error = message
                print(
                    "mandate reload failed; serving last-good config until the "
                    f"bot is restarted: {message}",
                    file=sys.stderr,
                )
            return self._last_mandate
        self._last_mandate = mandate
        self._last_mandate_error = None
        return mandate

    def _halt_path(self) -> Path:
        return self.settings.root_dir / self._mandate().execution.kill_switch_file

    def _store(self) -> PositionStore:
        return PositionStore(self._runtime() / f"positions.{self.settings.mode}.sqlite")

    def _audit(self) -> AuditWriter:
        return AuditWriter(self._runtime() / "audit.jsonl")

    def _audit_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = dict(payload)
        if self._current_update_meta:
            out["telegram"] = dict(self._current_update_meta)
        return out

    def _chat_role(self, chat_id: str) -> str | None:
        if chat_id == str(self.settings.telegram_chat_id):
            return "operator"
        if chat_id in {str(c) for c in self.settings.telegram_readonly_chat_ids}:
            return "readonly"
        return None

    def _is_operator_context(self) -> bool:
        if not self._current_update_meta:
            return True
        return self._current_update_meta.get("role") == "operator"

    def _require_operator(self) -> str | None:
        if self._is_operator_context():
            return None
        return "这个 Telegram 聊天只有只读权限，不能执行控制动作。"

    def _memory_path(self) -> Path:
        return self._runtime() / "bot_memory.json"

    def _load_memory(self) -> dict[str, Any]:
        try:
            data = json.loads(self._memory_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            data = {}
        if not isinstance(data, dict):
            return {}
        return data

    def _save_memory(self) -> None:
        path = self._memory_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.memory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _tickers_from_text(self, text: str) -> list[str]:
        tickers: list[str] = []
        for match in _TICKER_RE.findall(text.upper()):
            if match in _COMMON_UPPER_WORDS:
                continue
            if match not in tickers:
                tickers.append(match)
        return tickers[:8]

    def _remember_question(self, question: str) -> None:
        tickers = self._tickers_from_text(question)
        if tickers:
            existing = [
                str(t).upper()
                for t in self.memory.get("recent_tickers", [])
                if isinstance(t, str)
            ]
            merged = tickers + [t for t in existing if t not in tickers]
            self.memory["recent_tickers"] = merged[:12]
        topic = self._question_topic(question)
        if topic:
            self.memory["last_topic"] = topic
        self.memory["preferred_language"] = "zh"
        self._save_memory()

    def _memory_text(self) -> str:
        tickers = self.memory.get("recent_tickers") or []
        topic = self.memory.get("last_topic")
        parts = ["机器人记忆（非敏感）：偏好中文短答"]
        if tickers:
            parts.append("最近关注 ticker: " + ", ".join(map(str, tickers[:8])))
        if topic:
            parts.append(f"最近问题主题: {topic}")
        return "；".join(parts)

    def _set_next_markup(self, markup: dict[str, Any] | None) -> None:
        self._next_reply_markup = markup

    def _pop_next_markup(self) -> dict[str, Any] | None:
        markup = self._next_reply_markup
        self._next_reply_markup = None
        return markup

    def _menu_markup(self) -> dict[str, Any]:
        rows = [
            [
                {"text": "状态", "callback_data": "cmd:status"},
                {"text": "持仓", "callback_data": "cmd:positions"},
                {"text": "风险", "callback_data": "cmd:risk"},
            ],
            [
                {"text": "漏斗", "callback_data": "cmd:funnel"},
                {"text": "校准", "callback_data": "cmd:calibration"},
                {"text": "报告", "callback_data": "cmd:report"},
            ],
        ]
        if self._is_operator_context():
            rows.append(
                [
                    {"text": "启动循环", "callback_data": "prepare:run"},
                    {"text": "解除 HALT", "callback_data": "prepare:resume"},
                ]
            )
            rows.append(
                [
                    {"text": "停机说明", "callback_data": "cmd:halt_help"},
                    {"text": "帮助", "callback_data": "cmd:help"},
                ]
            )
        else:
            rows.append([{"text": "帮助", "callback_data": "cmd:help"}])
        return {"inline_keyboard": rows}

    def _confirmation_markup(self, action: str, code: str) -> dict[str, Any]:
        return {
            "inline_keyboard": [
                [
                    {
                        "text": "确认执行",
                        "callback_data": f"do:{action}:{code}",
                    },
                    {"text": "取消", "callback_data": f"cancel:{code}"},
                ]
            ]
        }

    def _confirmation_snapshot(self, action: str, reason: str = "") -> str:
        halt = self._halt_path()
        mode_label = "模拟盘" if self.settings.mode == "paper" else "实盘"
        market = "开盘中" if is_market_hours(self._now()) else "闭市"
        lines = [
            f"动作：{action}",
            f"模式：{mode_label}",
            f"账户：{self.settings.account_id or '(未设置)'}",
            f"市场：{market}",
            "HALT：" + ("已激活" if halt.exists() else "未激活"),
        ]
        if reason:
            lines.append(f"原因：{reason}")
        if self.settings.mode == "live":
            lines.append("实盘确认：这是 real-money 控制动作。")
        return "\n".join(lines)

    def _prepare_action(self, action: str, reason: str = "") -> str:
        if action == "run" and self._loop_running():
            return "循环已在运行（心跳很新鲜），不重复启动。"
        code = secrets.token_hex(3)
        self.pending_actions[code] = {
            "action": action,
            "reason": reason,
            "created_at": self._now().isoformat(),
        }
        self._set_next_markup(self._confirmation_markup(action, code))
        return (
            "请确认执行下面的 Telegram 控制动作：\n"
            f"{self._confirmation_snapshot(action, reason)}\n\n"
            f"确认码：{code}\n"
            f"点击按钮，或输入 /confirm {code}。5 分钟后失效。"
        )

    def _confirmed_action(self, code: str, expected_action: str | None = None) -> str:
        pending = self.pending_actions.pop(code, None)
        if pending is None:
            return "确认码不存在或已经用过。请重新发起动作。"
        try:
            created_at = datetime.fromisoformat(str(pending.get("created_at")))
        except ValueError:
            created_at = self._now()
        if (self._now() - created_at).total_seconds() > _ACTION_CONFIRM_SECONDS:
            return "确认码已过期。请重新发起动作。"

        action = str(pending.get("action") or "")
        if expected_action is not None and action != expected_action:
            return "确认动作不匹配。请重新发起动作。"
        if action == "run":
            return self._execute_run()
        if action == "resume":
            return self._execute_resume()
        if action == "halt":
            return self._execute_halt(str(pending.get("reason") or "operator halt"))
        return "未知确认动作。"

    def _execute_halt(self, reason: str) -> str:
        halt = self._halt_path()
        halt.parent.mkdir(parents=True, exist_ok=True)
        halt.write_text(f"{reason}\n", encoding="utf-8")
        self._audit().append(
            "kill_switch_activated",
            self._audit_payload({"reason": reason, "via": "telegram"}),
        )
        return f"⛔ 已写入 HALT：{reason}\n所有周期将在开始前中止。用 /resume 解除。"

    def _execute_resume(self) -> str:
        halt = self._halt_path()
        was = halt.exists()
        halt.unlink(missing_ok=True)
        self._audit().append(
            "kill_switch_cleared",
            self._audit_payload({"was_active": was, "via": "telegram"}),
        )
        return "🟢 HALT 已解除，交易恢复。" if was else "HALT 本来就没有激活。"

    def _execute_run(self) -> str:
        if self._loop_running():
            return "循环已在运行（心跳很新鲜），不重复启动。"
        try:
            self._spawn()
        except Exception as exc:  # noqa: BLE001
            return f"启动失败：{exc}"
        self._audit().append(
            "telegram_run_requested",
            self._audit_payload({"via": "telegram", "mode": self.settings.mode}),
        )
        note = "" if is_market_hours(self._now()) else "\n（当前闭市，循环会等到开盘才交易）"
        return "🟢 交易循环已启动。" + note

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
        mandate = self._mandate()
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

    def _risk_text(self) -> str:
        report = build_portfolio_risk_report(
            self._store().open_positions(), as_of=market_date(self._now())
        )
        totals = report.get("totals") or {}
        shock = report.get("scenario_revaluation_pnl_usd") or report.get(
            "shock_pnl_usd"
        ) or {}
        lines = [
            "组合风险：",
            f"持仓 {report.get('open_positions', 0)} 个 | "
            f"权利金风险 ${float(report.get('premium_at_risk_usd') or 0):.2f}",
            "Greeks: "
            f"Delta {float(totals.get('delta') or 0):+.2f}, "
            f"Gamma {float(totals.get('gamma') or 0):+.2f}, "
            f"Vega {float(totals.get('vega') or 0):+.2f}, "
            f"Theta ${float(totals.get('theta_usd_per_day') or 0):+.2f}/日",
            f"Theta 损耗 ${float(report.get('theta_decay_usd_per_day') or 0):.2f}/日",
        ]
        if shock:
            lines.append(
                "情景损益: "
                f"-5% ${float(shock.get('underlying_-5pct') or 0):+.2f}, "
                f"-2% ${float(shock.get('underlying_-2pct') or 0):+.2f}, "
                f"+2% ${float(shock.get('underlying_+2pct') or 0):+.2f}, "
                f"+5% ${float(shock.get('underlying_+5pct') or 0):+.2f}"
            )
        missing = int(report.get("missing_greeks") or 0)
        skipped = int(report.get("scenario_revaluation_skipped") or 0)
        if missing or skipped:
            lines.append(f"数据缺口: Greeks 缺失 {missing} 个，重估跳过 {skipped} 个。")
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

    def _funnel_text(self) -> str:
        events = read_audit_events(self._runtime() / "audit.jsonl")
        report = build_funnel_report(events, market_date(self._now()))
        decisions = report.get("committee_decisions") or {}
        liquidity = report.get("liquidity_at_entry") or {}
        gate = report.get("risk_gate") or {}
        return "\n".join(
            [
                f"今日漏斗（{report.get('scope')}）：",
                f"扫描入选 {report.get('universe_selected', 0)} | "
                f"事件种子 {report.get('event_seeds', 0)} | "
                f"探测缓存 {report.get('probe_cache_hits', 0)}",
                f"无合格合约 {report.get('dropped_no_eligible_contract', 0)} | "
                f"进入委员会 {report.get('reached_committee', 0)} | "
                f"缓存决策 {report.get('served_from_cache', 0)}",
                "委员会: "
                f"开仓 {decisions.get('open_position', 0)}, "
                f"观望 {decisions.get('hold', 0)}, "
                f"拒绝 {decisions.get('reject', 0)}",
                f"MC 检查 {report.get('monte_carlo_checked', 0)} | "
                f"MC 拒绝 {report.get('monte_carlo_rejected', 0)}",
                "流动性: "
                f"通过 {liquidity.get('passed', 0)}, "
                f"失败 {liquidity.get('failed', 0)} | "
                f"风控通过 {gate.get('approved', 0)}, "
                f"拒绝 {gate.get('rejected', 0)}",
                f"买入成交 {report.get('orders_filled', 0)}",
            ]
        )

    def _calibration_text(self) -> str:
        events = read_audit_events(self._runtime() / "audit.jsonl")
        report = build_calibration_report(events, market_date(self._now()))
        lines = [
            f"今日胜率校准（{report.get('scope')}）：",
            f"有胜率估计的提案 {report.get('proposals_with_estimate', 0)} | "
            f"已闭环样本 {report.get('closed_trades_matched', 0)}",
        ]
        buckets = report.get("buckets") or []
        if not buckets:
            lines.append("暂无分桶数据。")
            return "\n".join(lines)
        for bucket in buckets:
            actual = bucket.get("actual_win_rate")
            predicted = bucket.get("predicted_avg")
            actual_s = "n/a" if actual is None else f"{float(actual):.0%}"
            predicted_s = "n/a" if predicted is None else f"{float(predicted):.0%}"
            lines.append(
                f"{bucket.get('range')}: 样本 {bucket.get('trades', 0)}, "
                f"预测 {predicted_s}, 实际 {actual_s}"
            )
        return "\n".join(lines)

    def _question_topic(self, question: str) -> str:
        q = question.lower()
        if any(k in question for k in ("风险", "希腊", "损耗", "敞口")) or any(
            k in q for k in ("greek", "delta", "gamma", "vega", "theta")
        ):
            return "risk"
        if any(k in question for k in ("漏斗", "筛选", "没开仓", "没有开仓", "拒绝")):
            return "funnel"
        if any(k in question for k in ("校准", "胜率", "命中率", "预测")):
            return "calibration"
        if any(k in question for k in ("亏", "赚", "交易", "买", "平仓")):
            return "trade"
        return "general"

    def _question_context_sections(self, question: str) -> list[str]:
        topic = self._question_topic(question)
        sections = [self._status_text(), self._memory_text()]
        if topic in {"risk", "trade", "general"}:
            sections.append(self._positions_text())
        if topic in {"risk", "general"}:
            sections.append(self._risk_text())
        if topic in {"funnel", "general"}:
            sections.append(self._funnel_text())
        if topic in {"calibration", "general"}:
            sections.append(self._calibration_text())
        if topic in {"trade", "general"}:
            sections.append(self._trades_text())
        sections.append(self._audit_digest(question=question))
        return sections

    def _audit_digest(self, limit: int = 50, question: str = "") -> str:
        """Compact recent-decision trail for the Q&A context.

        Committee rationales and probability estimates are what let the AI
        answer "why did it buy/lose" with the system's own reasoning instead
        of guessing.
        """

        return self._audit_digest_for(limit=limit, question=question)

    def _audit_digest_for(self, limit: int = 50, question: str = "") -> str:
        events = read_audit_events(self._runtime() / "audit.jsonl")
        tickers = set(self._tickers_from_text(question))
        topic = self._question_topic(question) if question else "general"
        wanted_by_topic = {
            "funnel": {
                "universe_selected",
                "no_eligible_contracts",
                "entries_skipped",
                "proposal_rejected",
                "candidate_validated",
                "proposal_checked",
                "monte_carlo_pop",
            },
            "risk": {
                "risk_caps_scaled",
                "circuit_breaker_tripped",
                "position_closed",
                "order_filled",
                "kill_switch_activated",
                "kill_switch_cleared",
            },
            "calibration": {"committee_run", "position_closed"},
            "trade": {
                "committee_run",
                "committee_cache_hit",
                "monte_carlo_pop",
                "order_filled",
                "position_closed",
                "proposal_rejected",
            },
        }
        wanted = wanted_by_topic.get(topic)
        lines: list[str] = []
        for event in events[-400:]:
            etype = event.get("event_type", "")
            payload = event.get("payload", {}) or {}
            if wanted is not None and etype not in wanted:
                continue
            if tickers and not self._event_mentions_ticker(event, tickers):
                continue
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
                "candidate_validated",
                "proposal_checked",
            }:
                detail = json.dumps(payload, ensure_ascii=False)[:140]
                lines.append(f"{at} {etype}: {detail}")
        if not lines:
            return "审计日志：还没有任何决策记录。"
        return "审计日志摘要（最近的决策轨迹）：\n" + "\n".join(lines[-limit:])

    def _event_mentions_ticker(
        self, event: dict[str, Any], tickers: set[str]
    ) -> bool:
        payload = event.get("payload", {}) or {}
        candidates = {
            str(payload.get("ticker") or "").upper(),
            str(payload.get("symbol") or "").upper(),
        }
        option_code = str(payload.get("option_code") or "").upper()
        candidates.update(self._tickers_from_text(option_code))
        output = payload.get("output") or {}
        if isinstance(output, dict):
            proposal = output.get("proposal") or {}
            if isinstance(proposal, dict):
                candidates.add(str(proposal.get("ticker") or "").upper())
                candidates.update(
                    self._tickers_from_text(str(proposal.get("option_code") or ""))
                )
        return bool(tickers & {c for c in candidates if c})

    def _qa_context(self, question: str = "") -> str:
        return "\n\n".join(self._question_context_sections(question))

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

    def _alert_allowed(self, key: str) -> bool:
        now = self._now()
        last = self._operator_alerts_sent.get(key)
        if (
            last is not None
            and (now - last).total_seconds() < _OPERATOR_ALERT_REALERT_SECONDS
        ):
            return False
        self._operator_alerts_sent[key] = now
        return True

    def _parse_date(self, value: Any) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None

    def _operator_risk_candidates(self) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        now_day = market_date(self._now())
        mandate = self._mandate()
        rows = self._store().open_positions()
        risk = build_portfolio_risk_report(rows, as_of=now_day)
        premium = float(risk.get("premium_at_risk_usd") or 0)
        premium_cap = mandate.portfolio.max_total_premium_at_risk_usd
        if premium_cap > 0 and premium >= premium_cap * 0.9:
            candidates.append(
                (
                    "premium_cap",
                    f"⚠️ 权利金风险接近上限：${premium:.2f} / ${premium_cap:.2f}",
                )
            )

        events = read_audit_events(self._runtime() / "audit.jsonl")
        daily = build_daily_report(events, now_day)
        pnl = float(daily.get("realized_pnl_usd") or 0)
        loss_stop = mandate.portfolio.daily_loss_stop_usd
        if loss_stop > 0 and pnl <= -loss_stop * 0.8:
            candidates.append(
                (
                    "daily_loss_near_stop",
                    f"⚠️ 今日已实现盈亏 ${pnl:+.2f}，接近单日亏损停机线 ${loss_stop:.2f}",
                )
            )

        theta_limit = mandate.options.max_theta_decay_pct_per_day
        for row in rows:
            code = str(row.get("option_code") or "")
            time_stop = self._parse_date(row.get("time_stop"))
            pre_earnings = self._parse_date(row.get("pre_earnings_exit_date"))
            for label, due in (("时间止损", time_stop), ("财报前退出", pre_earnings)):
                if due is None:
                    continue
                days_left = (due - now_day).days
                if days_left <= 1:
                    candidates.append(
                        (
                            f"{label}:{code}",
                            f"⚠️ {code} {label}临近：{due.isoformat()}（剩 {days_left} 天）",
                        )
                    )
            theta_pct = row.get("entry_theta_decay_pct_per_day")
            if (
                theta_limit > 0
                and isinstance(theta_pct, (int, float))
                and float(theta_pct) >= theta_limit * 0.9
            ):
                candidates.append(
                    (
                        f"theta:{code}",
                        f"⚠️ {code} Theta 损耗偏高：{float(theta_pct):.1f}%/日",
                    )
                )

        remaining = self.llm_budget.remaining(now_day)
        if remaining <= BOT_LLM_DAILY_TOKENS * 0.1:
            candidates.append(
                (
                    "bot_llm_budget_low",
                    f"⚠️ Telegram AI 问答 token 预算偏低：剩余 {remaining}",
                )
            )

        recent = events[-40:]
        no_contracts = sum(1 for e in recent if e.get("event_type") == "no_eligible_contracts")
        if no_contracts >= 3:
            candidates.append(
                (
                    "no_eligible_contracts_recent",
                    f"⚠️ 最近 {no_contracts} 次候选没有合格期权合约，可能需要复查筛选/合约门槛。",
                )
            )
        crashes = [e for e in recent[-12:] if e.get("event_type") == "cycle_crashed"]
        if len(crashes) >= 2:
            last_error = (crashes[-1].get("payload") or {}).get("error")
            candidates.append(
                (
                    "cycle_crashes_recent",
                    f"⚠️ 交易循环近期多次崩溃（{len(crashes)} 次）。最近错误：{last_error}",
                )
            )
        wide_spreads = 0
        max_spread = mandate.options.max_bid_ask_spread_pct
        for event in recent:
            if event.get("event_type") != "candidate_validated":
                continue
            result = (event.get("payload") or {}).get("result") or {}
            spread = result.get("spread_pct")
            if isinstance(spread, (int, float)) and spread > max_spread:
                wide_spreads += 1
        if wide_spreads >= 3:
            candidates.append(
                (
                    "wide_spreads_recent",
                    f"⚠️ 最近 {wide_spreads} 个合约价差超过 {max_spread:.0f}% 门槛。",
                )
            )
        return candidates

    def maybe_alert_operator_risks(self) -> list[str]:
        """Send proactive operator alerts for risk/ops issues; returns sent texts."""

        sent: list[str] = []
        for key, text in self._operator_risk_candidates():
            if not self._alert_allowed(key):
                continue
            self._audit().append(
                "operator_risk_alert",
                {"key": key, "text": text, "at": self._now().isoformat()},
            )
            self.send(text)
            sent.append(text)
        return sent

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
    def _natural_language_action(self, text: str) -> str | None:
        if any(q in text for q in ("?", "？", "吗")):
            return None
        lowered = text.lower()
        if any(k in text for k in ("暂停", "停机", "先别交易", "不要交易", "别交易")):
            denied = self._require_operator()
            if denied:
                return denied
            reason = text[:160] or "operator halt via telegram"
            return self._prepare_action("halt", reason)
        if any(k in text for k in ("恢复交易", "解除停机", "解除 halt", "继续交易")):
            denied = self._require_operator()
            if denied:
                return denied
            return self._prepare_action("resume")
        if any(k in text for k in ("启动循环", "开始交易", "跑一轮", "运行循环")) or (
            "run" in lowered and "cycle" in lowered
        ):
            denied = self._require_operator()
            if denied:
                return denied
            return self._prepare_action("run")
        return None

    def handle_callback(self, data: str) -> str:
        data = (data or "").strip()
        if data == "cmd:status":
            return self._status_text()
        if data == "cmd:positions":
            return self._positions_text()
        if data == "cmd:risk":
            return self._risk_text()
        if data == "cmd:funnel":
            return self._funnel_text()
        if data == "cmd:calibration":
            return self._calibration_text()
        if data == "cmd:report":
            return self._report_text()
        if data == "cmd:help":
            self._set_next_markup(self._menu_markup())
            return _HELP
        if data == "cmd:halt_help":
            return "急停请发送：/halt 原因\n自然语言暂停也会先给你确认码。"
        if data == "prepare:run":
            denied = self._require_operator()
            if denied:
                return denied
            return self._prepare_action("run")
        if data == "prepare:resume":
            denied = self._require_operator()
            if denied:
                return denied
            return self._prepare_action("resume")
        if data.startswith("do:"):
            denied = self._require_operator()
            if denied:
                return denied
            parts = data.split(":", 2)
            if len(parts) != 3:
                return "确认数据无效。"
            _, action, code = parts
            return self._confirmed_action(code, expected_action=action)
        if data.startswith("cancel:"):
            denied = self._require_operator()
            if denied:
                return denied
            code = data.split(":", 1)[1]
            existed = self.pending_actions.pop(code, None) is not None
            return "已取消该动作。" if existed else "这个确认码已不存在。"
        return "不认识这个按钮。"

    def handle_text(self, text: str) -> str:
        text = (text or "").strip()
        lowered = text.lower()

        if lowered in {"/start", "/help", "/menu"}:
            self._set_next_markup(self._menu_markup())
            return _HELP
        if lowered == "/status":
            return self._status_text()
        if lowered == "/positions":
            return self._positions_text()
        if lowered == "/risk":
            return self._risk_text()
        if lowered == "/funnel":
            return self._funnel_text()
        if lowered == "/calibration":
            return self._calibration_text()
        if lowered == "/report":
            return self._report_text()
        if lowered.startswith("/confirm"):
            denied = self._require_operator()
            if denied:
                return denied
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                return "用法：/confirm 确认码"
            return self._confirmed_action(parts[1].strip())
        if lowered.startswith("/cancel"):
            denied = self._require_operator()
            if denied:
                return denied
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                return "用法：/cancel 确认码"
            existed = self.pending_actions.pop(parts[1].strip(), None) is not None
            return "已取消该动作。" if existed else "这个确认码已不存在。"
        if lowered.startswith("/halt"):
            denied = self._require_operator()
            if denied:
                return denied
            reason = text[5:].strip() or "operator halt via telegram"
            return self._execute_halt(reason)
        if lowered == "/resume":
            denied = self._require_operator()
            if denied:
                return denied
            return self._prepare_action("resume")
        if lowered == "/run":
            denied = self._require_operator()
            if denied:
                return denied
            return self._prepare_action("run")
        if text.startswith("/"):
            return "不认识这个命令。\n" + _HELP

        action_reply = self._natural_language_action(text)
        if action_reply is not None:
            return action_reply
        return self._ai_answer(text)

    def _ai_answer(self, question: str) -> str:
        if self.llm is None:
            return "AI 助手未配置。动作请用命令：\n" + _HELP
        today = market_date(self._now())
        if self.llm_budget.remaining(today) <= 0:
            return "今天 AI 问答的 token 预算用完了，明天恢复。命令仍然可用。"
        self._remember_question(question)
        context = self._qa_context(question)
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
            update_id = int(update.get("update_id", 0))
            self.offset = max(self.offset, update_id + 1)
            callback = update.get("callback_query") or {}
            if callback:
                message = callback.get("message") or {}
                chat_id = str((message.get("chat") or {}).get("id") or "")
                role = self._chat_role(chat_id)
                if role is None:
                    continue
                self._answer_callback(str(callback.get("id") or ""))
                self._current_update_meta = {
                    "update_id": update_id,
                    "callback_id": callback.get("id"),
                    "message_id": message.get("message_id"),
                    "data": callback.get("data"),
                    "chat_id": chat_id,
                    "role": role,
                }
                try:
                    reply = self.handle_callback(str(callback.get("data") or ""))
                    self.send(reply, self._pop_next_markup(), chat_id=chat_id)
                    handled += 1
                finally:
                    self._current_update_meta = None
                continue

            message = update.get("message") or {}
            chat_id = str((message.get("chat") or {}).get("id") or "")
            text = message.get("text") or ""
            role = self._chat_role(chat_id)
            if role is None:
                continue  # only configured Telegram chats may talk to this system
            if not text:
                continue
            self._current_update_meta = {
                "update_id": update_id,
                "message_id": message.get("message_id"),
                "chat_id": chat_id,
                "role": role,
            }
            try:
                self.send(self.handle_text(text), self._pop_next_markup(), chat_id=chat_id)
                handled += 1
            finally:
                self._current_update_meta = None
        return handled


def run_bot(settings: Settings) -> None:
    """Long-polling listener; restarts its poll on transient failures."""

    try:
        with single_instance_lock(
            settings.root_dir / "runtime" / "bot.lock",
            stale_seconds=7 * 24 * 60 * 60,
        ):
            _run_bot_locked(settings)
    except CycleLockError as exc:
        print(f"bot already running: {exc}", file=sys.stderr)


def _run_bot_locked(settings: Settings) -> None:
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
            bot.maybe_alert_operator_risks()
        except KeyboardInterrupt:
            bot.send("🤖 控制台已下线。")
            raise
        except Exception as exc:  # noqa: BLE001 - keep listening
            print(f"bot poll error: {exc}", file=sys.stderr)
            _time.sleep(5)
