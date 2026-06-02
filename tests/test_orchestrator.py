import json
from datetime import date, datetime, timezone
from pathlib import Path

from trading_agent.domain.evidence import EvidenceItem
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.risk import Mandate, QuoteSnapshot, RiskGate
from trading_agent.execution.orchestrator import PaperTradingCycle
from trading_agent.research.committee import Committee
from trading_agent.research.llm import MockLLMClient
from trading_agent.storage.audit import AuditWriter
from trading_agent.storage.positions import PositionStore

NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)
OPTION_CODE = "US.EXAMPLE260626C00005000"

_PROPOSAL = {
    "decision": "open_position",
    "ticker": "EXAMPLE",
    "option_code": OPTION_CODE,
    "option_side": "call",
    "action": "buy_to_open",
    "contracts": 1,
    "limit_price": 0.2,
    "max_limit_price": 0.21,
    "thesis": "Named supply agreement catalyst.",
    "evidence_ids": ["e1"],
    "confidence": 0.7,
    "expected_catalyst_window": "2026-06-10/2026-06-20",
    "exit_plan": {"take_profit_pct": 100, "stop_loss_pct": 50, "time_stop": "2026-06-24"},
    "invalidation": ["Catalyst delayed"],
}


class FakeMarket:
    def __init__(self, bid: float = 0.19, ask: float = 0.21, listed: bool = True):
        self.bid = bid
        self.ask = ask
        self.listed = listed

    def is_listed_option(self, option_code):
        return self.listed

    def option_quote(self, *, option_code, expiry, lot_size, now=None):
        return QuoteSnapshot(
            option_code=option_code,
            bid=self.bid,
            ask=self.ask,
            open_interest=200,
            daily_volume=30,
            lot_size=lot_size,
            expiry=expiry,
            observed_at=now or NOW,
        )


class FakeBroker:
    def __init__(self, positions=None, fill_status: str = "FILLED_ALL"):
        self.positions = positions or []
        self.placed: list[tuple[str, str]] = []
        self.trd_envs: list[str] = []
        self.cancelled: list[str] = []
        self.positions_queried = False
        self.fill_status = fill_status
        self._last: dict[str, float] = {"price": 0.0, "qty": 0.0}

    def positions_query(self, account_id, trd_env="SIMULATE"):
        self.positions_queried = True
        return self.positions

    def place_limit_order(
        self, *, account_id, option_code, contracts, limit_price, side, trd_env="SIMULATE"
    ):
        self.placed.append((side, option_code))
        self.trd_envs.append(trd_env)
        self._last = {"price": limit_price, "qty": contracts}
        return [{"order_id": f"order-{len(self.placed)}"}]

    def order_status(self, account_id, order_id, trd_env="SIMULATE"):
        return {
            "order_id": order_id,
            "order_status": self.fill_status,
            "dealt_qty": self._last["qty"],
            "dealt_avg_price": self._last["price"],
        }

    def cancel_order(self, account_id, order_id, trd_env="SIMULATE"):
        self.cancelled.append(order_id)
        return [{"order_id": order_id, "order_status": "CANCELLED_ALL"}]


class FakeSec:
    def fetch_evidence(self, ticker, *, forms=None, limit=20):
        return [
            EvidenceItem(
                evidence_id="e1",
                ticker=ticker.upper(),
                source_type="sec_8k",
                source_url="https://sec.gov/e1",
                published_at=NOW,
                observed_fact="Named multi-year supply agreement filed in an 8-K.",
                retrieved_at=NOW,
            )
        ]


def _mandate() -> Mandate:
    return Mandate.load(Path("config/mandate.paper.yaml"))


def _committee(responses: list[str]) -> Committee:
    return Committee(MockLLMClient(responses))


def _cycle(
    tmp_path: Path,
    *,
    broker,
    market,
    committee,
    store: PositionStore | None = None,
    trd_env: str = "SIMULATE",
    max_open_positions_override: int | None = None,
) -> PaperTradingCycle:
    mandate = _mandate()
    return PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=market,
        broker=broker,
        sec_client=FakeSec(),
        committee=committee,
        gate=RiskGate(mandate, tmp_path),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=store or PositionStore(tmp_path / "positions.sqlite"),
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        now_fn=lambda: NOW,
        trd_env=trd_env,
        max_open_positions_override=max_open_positions_override,
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _seconds: None,
    )


def _seed_open(store: PositionStore, code: str = OPTION_CODE) -> None:
    store.open_position(
        option_code=code,
        ticker="EXAMPLE",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 6, 26),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 6, 24),
    )


def _open_responses() -> list[str]:
    return ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]


def test_halt_aborts_before_any_broker_call(tmp_path: Path) -> None:
    halt = tmp_path / "runtime" / "HALT"
    halt.parent.mkdir(parents=True)
    halt.write_text("halt\n", encoding="utf-8")
    broker = FakeBroker()
    cycle = _cycle(tmp_path, broker=broker, market=FakeMarket(), committee=_committee([]))

    result = cycle.run_once(["EXAMPLE"])

    assert result.halted is True
    assert broker.positions_queried is False
    assert broker.placed == []


def test_entry_routes_through_gate_and_places_order(tmp_path: Path) -> None:
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path, broker=broker, market=FakeMarket(), committee=_committee(_open_responses())
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == [("buy", OPTION_CODE)]
    assert result.entries == [{"ticker": "EXAMPLE", "option_code": OPTION_CODE}]
    assert len(cycle.position_store.open_positions()) == 1


def test_gate_rejection_blocks_order_no_bypass(tmp_path: Path) -> None:
    # Broker already holds the proposed code, so the gate flags a duplicate order.
    broker = FakeBroker(positions=[{"code": OPTION_CODE, "qty": 1}])
    store = PositionStore(tmp_path / "positions.sqlite")
    store.open_position(
        option_code=OPTION_CODE,
        ticker="EXAMPLE",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 6, 26),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 6, 24),
    )
    mandate = _mandate()
    cycle = PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=FakeMarket(),
        broker=broker,
        sec_client=FakeSec(),
        committee=_committee(_open_responses()),
        gate=RiskGate(mandate, tmp_path),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=store,
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        now_fn=lambda: NOW,
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _seconds: None,
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []  # no order placed -> gate was not bypassed
    assert any(r["stage"] == "risk_gate" for r in result.rejected)


def test_take_profit_exit_closes_position(tmp_path: Path) -> None:
    broker = FakeBroker(positions=[{"code": OPTION_CODE, "qty": 1}])
    store = PositionStore(tmp_path / "positions.sqlite")
    store.open_position(
        option_code=OPTION_CODE,
        ticker="EXAMPLE",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 6, 26),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 6, 24),
    )
    mandate = _mandate()
    cycle = PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=FakeMarket(bid=0.40, ask=0.42),  # +100% -> take profit
        broker=broker,
        sec_client=FakeSec(),
        committee=_committee([]),  # no entries attempted (capacity reached anyway)
        gate=RiskGate(mandate, tmp_path),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=store,
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        now_fn=lambda: NOW,
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _seconds: None,
    )

    result = cycle.run_once([])

    assert ("sell", OPTION_CODE) in broker.placed
    assert result.exits and result.exits[0]["reason"] == "take profit"
    assert store.open_positions() == []


def test_reconcile_closes_ledger_position_not_held(tmp_path: Path) -> None:
    broker = FakeBroker(positions=[])  # broker holds nothing
    store = PositionStore(tmp_path / "positions.sqlite")
    store.open_position(
        option_code=OPTION_CODE,
        ticker="EXAMPLE",
        option_side="call",
        entry_price=0.20,
        contracts=1,
        lot_size=100,
        expiry=date(2026, 6, 26),
        take_profit_pct=100,
        stop_loss_pct=50,
        time_stop=date(2026, 6, 24),
    )
    mandate = _mandate()
    cycle = PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=FakeMarket(),
        broker=broker,
        sec_client=FakeSec(),
        committee=_committee([]),
        gate=RiskGate(mandate, tmp_path),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=store,
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        now_fn=lambda: NOW,
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _seconds: None,
    )

    result = cycle.run_once([])

    assert result.reconciled_closed == [OPTION_CODE]
    assert store.open_positions() == []
    assert broker.placed == []


def test_live_trd_env_passed_through_to_broker(tmp_path: Path) -> None:
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(),
        committee=_committee(_open_responses()),
        trd_env="REAL",
    )

    cycle.run_once(["EXAMPLE"])

    assert broker.placed == [("buy", OPTION_CODE)]
    assert broker.trd_envs == ["REAL"]


def test_unfilled_entry_is_cancelled_and_not_recorded(tmp_path: Path) -> None:
    broker = FakeBroker(fill_status="SUBMITTED")  # never fills
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(),
        committee=_committee(_open_responses()),
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == [("buy", OPTION_CODE)]
    assert broker.cancelled  # order was cancelled after the timeout
    assert cycle.position_store.open_positions() == []  # no phantom position
    assert any(r["stage"] == "unfilled" for r in result.rejected)


def test_hallucinated_contract_is_rejected(tmp_path: Path) -> None:
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(listed=False),  # contract is not a real listed option
        committee=_committee(_open_responses()),
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []  # never reached the broker
    assert any(r["stage"] == "not_listed" for r in result.rejected)


def test_position_cap_override_blocks_new_entries(tmp_path: Path) -> None:
    # Already holding one; override cap of 1 means no new entry is attempted.
    broker = FakeBroker(positions=[{"code": OPTION_CODE, "qty": 1}])
    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_open(store)
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(),
        committee=_committee([]),  # would raise if a committee run were attempted
        store=store,
        max_open_positions_override=1,
    )

    result = cycle.run_once(["OTHERCO"])

    assert result.entries == []
    assert broker.placed == []  # only reconciliation, no order


def test_cooldown_blocks_entries_after_consecutive_losses(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "positions.sqlite")
    # Three small losses (-$3 each): below the $10 daily stop, but they hit the
    # 3-loss consecutive stop, so the cooldown blocks new entries.
    for code in (
        "US.AAA260626C00005000",
        "US.BBB260626C00005000",
        "US.CCC260626C00005000",
    ):
        _seed_open(store, code)
        store.mark_closed(code, close_reason="stop loss", exit_price=0.17)
    broker = FakeBroker()
    mandate = _mandate()
    cycle = PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=FakeMarket(),
        broker=broker,
        sec_client=FakeSec(),
        committee=_committee([]),  # would raise if an entry were attempted
        gate=RiskGate(mandate, tmp_path),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=store,
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        now_fn=lambda: datetime.now(timezone.utc),
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _seconds: None,
    )

    result = cycle.run_once(["EXAMPLE"])

    assert result.cooldown is True
    assert result.circuit_breaker is None  # daily loss not breached
    assert result.entries == []
    assert broker.placed == []


def test_circuit_breaker_trips_halt_and_blocks_entries(tmp_path: Path) -> None:
    store = PositionStore(tmp_path / "positions.sqlite")
    # Two realized losses of -$15 each, closed today => daily P/L -$30 trips the
    # $10 daily loss stop (checked before the $25 drawdown stop).
    for code in ("US.AAA260626C00005000", "US.BBB260626C00005000"):
        _seed_open(store, code)
        store.mark_closed(code, close_reason="stop loss", exit_price=0.05)
    broker = FakeBroker()
    mandate = _mandate()
    cycle = PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=FakeMarket(),
        broker=broker,
        sec_client=FakeSec(),
        committee=_committee([]),
        gate=RiskGate(mandate, tmp_path),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=store,
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        now_fn=lambda: datetime.now(timezone.utc),  # match same-day closed_at
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _seconds: None,
    )

    result = cycle.run_once(["EXAMPLE"])

    assert result.circuit_breaker == "daily loss stop"
    assert (tmp_path / "runtime" / "HALT").exists()
    assert result.entries == []
    assert broker.placed == []
