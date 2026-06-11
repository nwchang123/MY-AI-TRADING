import json
from datetime import date, datetime, timezone
from pathlib import Path

from trading_agent.data.moomoo_market import parse_us_option_code
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
    def __init__(
        self,
        bid: float = 0.19,
        ask: float = 0.21,
        chain_codes: list[str] | None = None,
        spot: float = 0.0,
        iv: float = 0.5,
    ):
        self.bid = bid
        self.ask = ask
        # Contracts the chain offers; defaults to the proposed code so it is an
        # eligible candidate. Set to a different code to simulate the committee
        # naming a contract that is not actually tradeable.
        self.chain_codes = [OPTION_CODE] if chain_codes is None else chain_codes
        # spot=0 means "no underlying snapshot": the Monte Carlo check is
        # skipped, matching markets where the price feed is unavailable.
        self.spot = spot
        self.iv = iv

    def underlying_snapshot(self, underlying):
        return {"price": self.spot} if self.spot else {}

    def option_chain(self, underlying, start, end, option_type="ALL"):
        rows = []
        for code in self.chain_codes:
            _, expiry, side, strike = parse_us_option_code(code)
            rows.append(
                {
                    "code": code,
                    "side": side,
                    "strike": strike,
                    "expiry": expiry,
                    "bid": self.bid,
                    "ask": self.ask,
                    "open_interest": 200,
                    "daily_volume": 30,
                    "iv": self.iv,
                }
            )
        return rows

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
            is_delayed=True,
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


def test_contract_not_in_chain_blocks_order(tmp_path: Path) -> None:
    # The chain offers a different contract than the committee names, so the
    # committee's candidate check rejects the code -> no order reaches the broker.
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(chain_codes=["US.OTHER260626C00005000"]),
        committee=_committee(_open_responses()),  # PM names OPTION_CODE
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []
    assert cycle.position_store.open_positions() == []


class FakeNews:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def fetch_evidence(self, ticker):
        if self.fail:
            raise RuntimeError("feed down")
        return [
            EvidenceItem(
                evidence_id="news-1",
                ticker=ticker.upper(),
                source_type="news_rss",
                source_url="https://example.com/n1",
                published_at=NOW,
                observed_fact="Partnership headline from a news feed.",
                retrieved_at=NOW,
            )
        ]


def test_news_evidence_merges_with_sec(tmp_path: Path) -> None:
    cycle = _cycle(
        tmp_path, broker=FakeBroker(), market=FakeMarket(), committee=_committee([])
    )
    cycle.news_client = FakeNews()

    from trading_agent.execution.orchestrator import CycleResult

    evidence = cycle._gather_evidence("EXAMPLE", CycleResult())
    assert {e.evidence_id for e in evidence} == {"e1", "news-1"}


def test_news_failure_is_nonfatal_entry_still_proceeds(tmp_path: Path) -> None:
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path, broker=broker, market=FakeMarket(), committee=_committee(_open_responses())
    )
    cycle.news_client = FakeNews(fail=True)

    result = cycle.run_once(["EXAMPLE"])

    # The dead feed is recorded but the SEC-backed entry still goes through.
    assert any(e["stage"] == "news_fetch" for e in result.errors)
    assert broker.placed == [("buy", OPTION_CODE)]


def test_monte_carlo_floor_blocks_hopeless_ticket(tmp_path: Path) -> None:
    # Spot $2 vs strike $5 with iv 0.5 and 24 DTE: doubling the option before
    # halving is ~impossible, so the deterministic baseline (floor 0.2 in the
    # paper mandate) rejects it even though the committee approved.
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(spot=2.0),
        committee=_committee(_open_responses()),
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []
    assert any(r["stage"] == "monte_carlo_pop" for r in result.rejected)


def test_monte_carlo_passes_underpriced_atm_ticket(tmp_path: Path) -> None:
    # ATM, high vol, entry far below model value: baseline POP is high, the
    # trade flows through to the broker.
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(spot=5.0, iv=1.0),
        committee=_committee(_open_responses()),
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == [("buy", OPTION_CODE)]
    assert result.entries


def test_missing_spot_skips_monte_carlo_check(tmp_path: Path) -> None:
    # No underlying snapshot (spot=0): the MC gate must skip, not block, so a
    # data outage cannot freeze trading. The default FakeMarket has spot=0 and
    # this is the same path every pre-existing entry test exercises.
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(),
        committee=_committee(_open_responses()),
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == [("buy", OPTION_CODE)]


def test_no_eligible_contracts_skips_committee(tmp_path: Path) -> None:
    # An empty chain means nothing is tradeable: the committee is never called
    # (saving tokens) and the ticker is recorded as having no eligible contract.
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(chain_codes=[]),
        committee=_committee([]),  # would raise if the committee were run
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []
    assert any(r["stage"] == "no_eligible_contracts" for r in result.rejected)


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


def _seed_closed(store: PositionStore, code: str, entry: float, exit_price: float) -> None:
    _seed_open(store, code)
    store.mark_closed(code, close_reason="test", exit_price=exit_price)
    # _seed_open uses entry 0.20; adjust by reopening with the wanted entry.
    if entry != 0.20:
        with store._connect() as conn:  # noqa: SLF001 - test seeding
            conn.execute(
                "UPDATE positions SET entry_price=? WHERE option_code=?", (entry, code)
            )


def test_compounding_win_unlocks_bigger_contracts(tmp_path: Path) -> None:
    # A realized +$100 win doubles equity; the $25 contract cap scales to $50,
    # so a $0.40-ask contract (cost ~$41) becomes tradeable. The paper mandate
    # has compounding: true.
    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_closed(store, "US.WIN260626C00005000", entry=0.20, exit_price=1.20)  # +100

    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(bid=0.38, ask=0.40),
        committee=_committee(_open_responses()),
        store=store,
    )
    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == [("buy", OPTION_CODE)]
    assert result.entries
    assert cycle.active_mandate.options.max_contract_cost_usd == 50.0


def test_fixed_mandate_rejects_what_compounding_allows(tmp_path: Path) -> None:
    # Same +$100 win and same $0.40-ask contract, but compounding off: the
    # static $25 cap rejects it at the liquidity stage.
    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_closed(store, "US.WIN260626C00005000", entry=0.20, exit_price=1.20)

    mandate = _mandate()
    account = mandate.account.model_copy(update={"compounding": False})
    mandate = mandate.model_copy(update={"account": account})
    broker = FakeBroker()
    cycle = PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=FakeMarket(bid=0.38, ask=0.40),
        broker=broker,
        sec_client=FakeSec(),
        committee=_committee([]),  # never reached: no candidate passes the cap
        gate=RiskGate(mandate, tmp_path),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=store,
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        now_fn=lambda: NOW,
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _seconds: None,
    )
    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []
    assert any(r["stage"] == "no_eligible_contracts" for r in result.rejected)


def test_compounding_drawdown_measured_from_peak(tmp_path: Path) -> None:
    # Win +$100 (peak 200) then lose $60 back (equity 140). Versus initial
    # capital that is a GAIN, but versus the peak it is a $60 drawdown, beyond
    # the scaled stop (25 x 2 = 50): the circuit breaker must halt. Both closes
    # happened "yesterday" relative to the cycle clock, so the daily stop stays
    # quiet and the drawdown logic is what trips.
    from datetime import timedelta

    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_closed(store, "US.WIN260626C00005000", entry=0.20, exit_price=1.20)  # +100
    _seed_closed(store, "US.LOSS260626C00005000", entry=0.80, exit_price=0.20)  # -60

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
        now_fn=lambda: datetime.now(timezone.utc) + timedelta(days=1),
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _seconds: None,
    )
    result = cycle.run_once(["EXAMPLE"])

    assert result.circuit_breaker == "hard drawdown stop"
    assert (tmp_path / "runtime" / "HALT").exists()
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
