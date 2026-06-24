import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

from trading_agent.bot import TelegramBot
from trading_agent.settings import Settings

ROOT = Path(__file__).parents[1]
NOW = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)


def _settings(tmp_path: Path) -> Settings:
    (tmp_path / "config").mkdir(exist_ok=True)
    copyfile(
        ROOT / "config" / "mandate.paper.yaml",
        tmp_path / "config" / "mandate.paper.yaml",
    )
    return Settings(
        root_dir=tmp_path,
        mode="paper",
        moomoo_host="127.0.0.1",
        moomoo_port=11111,
        security_firm="FUTUMY",
        account_id=4958218,
        live_acknowledgment="",
        telegram_bot_token="tok",
        telegram_chat_id="42",
    )


class FakeLLM:
    model = "fake"

    def __init__(self, reply: str = "AI 的回答"):
        self.reply = reply
        self.calls: list[dict] = []
        from trading_agent.research.llm import LLMUsage

        self.usage = LLMUsage(calls=0, prompt_tokens=0, completion_tokens=0)

    def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user})
        self.usage.calls += 1
        self.usage.prompt_tokens += 100
        return self.reply


def _bot(tmp_path: Path, llm=None, spawned=None) -> TelegramBot:
    return TelegramBot(
        _settings(tmp_path),
        llm_client=llm,
        http_fn=lambda method, payload: {"ok": True, "result": []},
        spawn_fn=(lambda: spawned.append(True)) if spawned is not None else None,
        now_fn=lambda: NOW,
    )


def _write_heartbeat(tmp_path: Path, at: datetime) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "heartbeat.json").write_text(
        json.dumps({"at": at.isoformat(), "note": "cycle done"}), encoding="utf-8"
    )


def _write_audit_event(tmp_path: Path, event_type: str, payload: dict) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(exist_ok=True)
    path = runtime / "audit.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "recorded_at": NOW.isoformat(),
                    "event_type": event_type,
                    "payload": payload,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def test_deadman_alerts_on_stale_heartbeat_during_market_hours(tmp_path: Path) -> None:
    from datetime import timedelta

    bot = _bot(tmp_path)  # NOW = 2026-06-11 15:00 UTC = 11:00 ET Thursday (open)
    _write_heartbeat(tmp_path, NOW - timedelta(minutes=90))

    alert = bot.maybe_alert_dead_loop()

    assert alert is not None and "死人开关" in alert
    # Re-alert is throttled: a second check inside the hour stays silent.
    assert bot.maybe_alert_dead_loop() is None


def test_deadman_alerts_when_heartbeat_missing(tmp_path: Path) -> None:
    bot = _bot(tmp_path)
    alert = bot.maybe_alert_dead_loop()
    assert alert is not None and "心跳文件不存在" in alert


def test_deadman_quiet_with_fresh_heartbeat(tmp_path: Path) -> None:
    from datetime import timedelta

    bot = _bot(tmp_path)
    _write_heartbeat(tmp_path, NOW - timedelta(minutes=10))
    assert bot.maybe_alert_dead_loop() is None


def test_deadman_quiet_when_market_closed(tmp_path: Path) -> None:
    closed = datetime(2026, 6, 13, 15, 0, tzinfo=timezone.utc)  # Saturday
    bot = TelegramBot(
        _settings(tmp_path),
        http_fn=lambda method, payload: {"ok": True, "result": []},
        now_fn=lambda: closed,
    )
    assert bot.maybe_alert_dead_loop() is None  # no heartbeat, but market shut


def test_help_lists_commands(tmp_path: Path) -> None:
    reply = _bot(tmp_path).handle_text("/help")
    for cmd in (
        "/status",
        "/run",
        "/halt",
        "/resume",
        "/positions",
        "/risk",
        "/funnel",
        "/calibration",
        "/report",
    ):
        assert cmd in reply


def test_halt_and_resume_roundtrip(tmp_path: Path) -> None:
    bot = _bot(tmp_path)
    reply = bot.handle_text("/halt 测试急停")
    assert "HALT" in reply and "测试急停" in reply
    assert (tmp_path / "runtime" / "HALT").exists()
    # Status reflects the halt.
    assert "HALT 停机中" in bot.handle_text("/status")
    reply = bot.handle_text("/resume")
    assert "确认码" in reply
    code = next(iter(bot.pending_actions))
    reply = bot.handle_text(f"/confirm {code}")
    assert "已解除" in reply
    assert not (tmp_path / "runtime" / "HALT").exists()


def test_status_reports_equity_and_market(tmp_path: Path) -> None:
    from trading_agent.domain.risk import Mandate

    capital = Mandate.load(
        ROOT / "config" / "mandate.paper.yaml"
    ).account.initial_capital_usd
    reply = _bot(tmp_path).handle_text("/status")
    assert f"权益 ${capital:.2f}" in reply  # fresh ledger -> initial capital
    assert "市场" in reply
    assert "心跳：从未运行" in reply


def test_risk_command_reports_portfolio_greeks(tmp_path: Path) -> None:
    from datetime import date

    from trading_agent.storage.positions import PositionStore

    store = PositionStore(tmp_path / "runtime" / "positions.paper.sqlite")
    store.open_position(
        option_code="US.DEMO260717C5000",
        ticker="DEMO",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 7, 17),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 7, 10),
        entry_spot=50,
        entry_iv=0.4,
        entry_delta=0.5,
        entry_gamma=0.01,
        entry_vega=0.2,
        entry_theta=-0.03,
        entry_theta_decay_pct_per_day=5,
    )

    reply = _bot(tmp_path).handle_text("/risk")

    assert "组合风险" in reply
    assert "权利金风险 $20.00" in reply
    assert "Delta +50.00" in reply
    assert "Theta $-3.00/日" in reply


def test_funnel_and_calibration_commands(tmp_path: Path) -> None:
    _write_audit_event(
        tmp_path,
        "universe_selected",
        {"tickers": ["AAA", "BBB"], "event_seeds": ["AAA"], "probe_cache_hits": 1},
    )
    _write_audit_event(tmp_path, "no_eligible_contracts", {"ticker": "BBB"})
    _write_audit_event(
        tmp_path,
        "committee_run",
        {
            "ticker": "AAA",
            "decision": "open_position",
            "output": {
                "win_probability": 0.62,
                "proposal": {"option_code": "US.AAA260717C5000"},
            },
        },
    )
    _write_audit_event(
        tmp_path,
        "candidate_validated",
        {"result": {"passed": True, "spread_pct": 5}},
    )
    _write_audit_event(tmp_path, "proposal_checked", {"decision": {"approved": True}})
    _write_audit_event(tmp_path, "order_filled", {"side": "buy"})
    _write_audit_event(
        tmp_path,
        "position_closed",
        {"option_code": "US.AAA260717C5000", "realized_pnl_usd": 10.0},
    )

    bot = _bot(tmp_path)

    funnel = bot.handle_text("/funnel")
    assert "扫描入选 2" in funnel
    assert "无合格合约 1" in funnel
    assert "买入成交 1" in funnel

    calibration = bot.handle_text("/calibration")
    assert "已闭环样本 1" in calibration
    assert "0.55-0.65" in calibration
    assert "实际 100%" in calibration


def test_run_spawns_loop_once(tmp_path: Path) -> None:
    spawned: list[bool] = []
    bot = _bot(tmp_path, spawned=spawned)
    reply = bot.handle_text("/run")
    assert "确认码" in reply
    assert spawned == []

    code = next(iter(bot.pending_actions))
    reply = bot.handle_text(f"/confirm {code}")
    assert "已启动" in reply
    assert spawned == [True]

    # Fresh heartbeat -> refuses to double-start.
    hb = tmp_path / "runtime" / "heartbeat.json"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text(
        json.dumps({"at": NOW.isoformat(), "note": "cycle done"}), encoding="utf-8"
    )
    reply = bot.handle_text("/run")
    assert "已在运行" in reply
    assert spawned == [True]


def test_confirmation_code_is_single_use(tmp_path: Path) -> None:
    spawned: list[bool] = []
    bot = _bot(tmp_path, spawned=spawned)
    bot.handle_text("/run")
    code = next(iter(bot.pending_actions))

    assert "已启动" in bot.handle_text(f"/confirm {code}")
    assert "已经用过" in bot.handle_text(f"/confirm {code}")
    assert spawned == [True]


def test_plain_text_goes_to_ai_with_context(tmp_path: Path) -> None:
    llm = FakeLLM()
    reply = _bot(tmp_path, llm=llm).handle_text("今晚为什么没开仓？")
    assert reply == "AI 的回答"
    assert "今晚为什么没开仓" in llm.calls[0]["user"]
    assert "系统状态" in llm.calls[0]["user"]  # read-only context attached
    assert "没有任何执行能力" in llm.calls[0]["system"]


def _seed_losing_trade(tmp_path: Path) -> None:
    from datetime import date

    from trading_agent.storage.audit import AuditWriter
    from trading_agent.storage.positions import PositionStore

    store = PositionStore(tmp_path / "runtime" / "positions.paper.sqlite")
    store.open_position(
        option_code="US.DEMO260717C5000",
        ticker="DEMO",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 7, 17),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 7, 10),
    )
    store.mark_closed(
        "US.DEMO260717C5000", close_reason="stop loss", exit_price=0.10
    )
    audit = AuditWriter(tmp_path / "runtime" / "audit.jsonl")
    audit.append(
        "committee_run",
        {
            "ticker": "DEMO",
            "decision": "open_position",
            "output": {
                "rationale": "8-K披露重大供货协议，预期两周内放量",
                "win_probability": 0.62,
            },
        },
    )
    audit.append(
        "position_closed",
        {
            "option_code": "US.DEMO260717C5000",
            "reason": "stop loss",
            "realized_pnl_usd": -10.0,
        },
    )


def test_ai_context_includes_trades_and_committee_rationale(tmp_path: Path) -> None:
    # The user must be able to ask "what did it trade and why did it lose":
    # the AI's context carries the trade P/L, the close reason, and the
    # committee's own entry rationale from the audit log.
    _seed_losing_trade(tmp_path)
    llm = FakeLLM()
    _bot(tmp_path, llm=llm).handle_text("今天交易了哪只股票？为什么会亏？")

    prompt = llm.calls[0]["user"]
    assert "US.DEMO260717C5000" in prompt
    assert "$-10.00" in prompt  # trade P/L from the ledger
    assert "stop loss" in prompt  # close reason
    assert "8-K披露重大供货协议" in prompt  # committee's entry rationale
    assert "胜率估计=0.62" in prompt


def test_followup_questions_carry_conversation_history(tmp_path: Path) -> None:
    llm = FakeLLM(reply="买了 DEMO 的看涨期权")
    bot = _bot(tmp_path, llm=llm)
    bot.handle_text("今天交易了什么？")
    bot.handle_text("那为什么亏了？")

    second_prompt = llm.calls[1]["user"]
    assert "今天交易了什么" in second_prompt  # previous question
    assert "买了 DEMO 的看涨期权" in second_prompt  # previous answer


def test_ai_question_persists_non_sensitive_memory(tmp_path: Path) -> None:
    llm = FakeLLM()
    bot = _bot(tmp_path, llm=llm)

    bot.handle_text("AMD 风险怎么样？")

    memory = json.loads((tmp_path / "runtime" / "bot_memory.json").read_text(encoding="utf-8"))
    assert memory["preferred_language"] == "zh"
    assert "AMD" in memory["recent_tickers"]
    assert "机器人记忆" in llm.calls[0]["user"]
    assert "AMD" in llm.calls[0]["user"]


def test_plain_text_without_llm_points_to_commands(tmp_path: Path) -> None:
    reply = _bot(tmp_path).handle_text("随便聊聊")
    assert "AI 助手未配置" in reply


def test_natural_language_action_requires_confirmation(tmp_path: Path) -> None:
    bot = _bot(tmp_path)

    reply = bot.handle_text("今晚先别交易")

    assert "确认码" in reply
    assert not (tmp_path / "runtime" / "HALT").exists()
    code = next(iter(bot.pending_actions))
    assert "已写入 HALT" in bot.handle_text(f"/confirm {code}")
    assert (tmp_path / "runtime" / "HALT").exists()


def test_non_operator_messages_are_ignored(tmp_path: Path) -> None:
    sent: list[str] = []

    def http(method, payload):
        if method == "getUpdates":
            return {
                "ok": True,
                "result": [
                    {
                        "update_id": 7,
                        "message": {"chat": {"id": 999}, "text": "/halt hack"},
                    }
                ],
            }
        sent.append(payload["text"])
        return {"ok": True}

    bot = TelegramBot(_settings(tmp_path), http_fn=http, now_fn=lambda: NOW)
    handled = bot.poll_once()

    assert handled == 0
    assert sent == []  # stranger gets nothing
    assert not (tmp_path / "runtime" / "HALT").exists()  # and changes nothing
    assert bot.offset == 8  # but the update is consumed


def test_operator_message_is_answered(tmp_path: Path) -> None:
    sent: list[str] = []

    def http(method, payload):
        if method == "getUpdates":
            return {
                "ok": True,
                "result": [
                    {
                        "update_id": 9,
                        "message": {"chat": {"id": 42}, "text": "/status"},
                    }
                ],
            }
        sent.append(payload["text"])
        return {"ok": True}

    bot = TelegramBot(_settings(tmp_path), http_fn=http, now_fn=lambda: NOW)
    assert bot.poll_once() == 1
    assert sent and "系统状态" in sent[0]


def test_readonly_chat_can_read_but_not_control(tmp_path: Path) -> None:
    from dataclasses import replace

    sent: list[dict] = []
    settings = replace(_settings(tmp_path), telegram_readonly_chat_ids=("100",))
    updates = [
        {"update_id": 10, "message": {"chat": {"id": 100}, "text": "/status"}},
        {"update_id": 11, "message": {"chat": {"id": 100}, "text": "/halt nope"}},
    ]

    def http(method, payload):
        if method == "getUpdates":
            return {"ok": True, "result": updates}
        sent.append(payload)
        return {"ok": True}

    bot = TelegramBot(settings, http_fn=http, now_fn=lambda: NOW)

    assert bot.poll_once() == 2
    assert sent[0]["chat_id"] == "100"
    assert "系统状态" in sent[0]["text"]
    assert sent[1]["chat_id"] == "100"
    assert "只读权限" in sent[1]["text"]
    assert not (tmp_path / "runtime" / "HALT").exists()


def test_menu_message_sends_inline_keyboard(tmp_path: Path) -> None:
    sent: list[dict] = []

    def http(method, payload):
        if method == "getUpdates":
            return {
                "ok": True,
                "result": [
                    {
                        "update_id": 10,
                        "message": {"chat": {"id": 42}, "text": "/menu"},
                    }
                ],
            }
        sent.append({"method": method, "payload": payload})
        return {"ok": True}

    bot = TelegramBot(_settings(tmp_path), http_fn=http, now_fn=lambda: NOW)

    assert bot.poll_once() == 1
    payload = sent[0]["payload"]
    assert "reply_markup" in payload
    assert payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "cmd:status"


def test_callback_confirmation_can_start_run(tmp_path: Path) -> None:
    spawned: list[bool] = []
    bot = _bot(tmp_path, spawned=spawned)

    reply = bot.handle_callback("prepare:run")
    code = next(iter(bot.pending_actions))
    assert "确认码" in reply
    assert "已启动" in bot.handle_callback(f"do:run:{code}")
    assert spawned == [True]


def test_operator_risk_alerts_time_stop_once(tmp_path: Path) -> None:
    from datetime import date

    from trading_agent.storage.positions import PositionStore

    sent: list[str] = []
    settings = _settings(tmp_path)
    store = PositionStore(tmp_path / "runtime" / "positions.paper.sqlite")
    store.open_position(
        option_code="US.DEMO260717C5000",
        ticker="DEMO",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 7, 17),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=NOW.date(),
    )

    bot = TelegramBot(
        settings,
        http_fn=lambda method, payload: sent.append(payload["text"]) or {"ok": True},
        now_fn=lambda: NOW,
    )

    alerts = bot.maybe_alert_operator_risks()

    assert any("时间止损临近" in text for text in alerts)
    assert sent == alerts
    assert bot.maybe_alert_operator_risks() == []
