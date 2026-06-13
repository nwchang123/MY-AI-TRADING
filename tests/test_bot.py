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
    for cmd in ("/status", "/run", "/halt", "/resume", "/positions", "/report"):
        assert cmd in reply


def test_halt_and_resume_roundtrip(tmp_path: Path) -> None:
    bot = _bot(tmp_path)
    reply = bot.handle_text("/halt 测试急停")
    assert "HALT" in reply and "测试急停" in reply
    assert (tmp_path / "runtime" / "HALT").exists()
    # Status reflects the halt.
    assert "HALT 停机中" in bot.handle_text("/status")
    reply = bot.handle_text("/resume")
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


def test_run_spawns_loop_once(tmp_path: Path) -> None:
    spawned: list[bool] = []
    bot = _bot(tmp_path, spawned=spawned)
    reply = bot.handle_text("/run")
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


def test_plain_text_without_llm_points_to_commands(tmp_path: Path) -> None:
    reply = _bot(tmp_path).handle_text("随便聊聊")
    assert "AI 助手未配置" in reply


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
