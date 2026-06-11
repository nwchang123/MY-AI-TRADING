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
        from trading_agent.research.llm import LlmUsage

        self.usage = LlmUsage(calls=0, prompt_tokens=0, completion_tokens=0)

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
    reply = _bot(tmp_path).handle_text("/status")
    assert "权益 $100.00" in reply  # fresh ledger -> initial capital
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
