import json
from datetime import date, datetime, timezone
from pathlib import Path

from trading_agent.data.moomoo_market import build_us_option_code, parse_us_option_code
from trading_agent.domain.evidence import EvidenceItem
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.risk import Mandate, QuoteSnapshot, RiskGate
from trading_agent.execution.orchestrator import (
    CycleResult,
    PaperTradingCycle,
    _parse_window_end,
)
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
    def fetch_evidence(
        self, ticker, *, forms=None, limit=20, fetch_bodies=False,
        body_limit=2, body_chars=1500,
    ):
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

    def company_name(self, ticker):
        return f"{ticker.title()} Inc."


def _mandate() -> Mandate:
    # Pin the canonical $100 risk profile so the compounding/cap assertions stay
    # stable when the live paper mandate's capital/caps change (e.g. $500 base).
    mandate = Mandate.load(Path("config/mandate.paper.yaml"))
    account = mandate.account.model_copy(update={"initial_capital_usd": 100.0})
    options = mandate.options.model_copy(update={"max_contract_cost_usd": 65.0})
    portfolio = mandate.portfolio.model_copy(
        update={
            "max_total_premium_at_risk_usd": 100.0,
            "daily_loss_stop_usd": 35.0,
            "hard_drawdown_stop_usd": 50.0,
        }
    )
    return mandate.model_copy(
        update={"account": account, "options": options, "portfolio": portfolio}
    )


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
    mandate: "Mandate | None" = None,
    earnings_calendar=None,
) -> PaperTradingCycle:
    mandate = mandate or _mandate()
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
        earnings_calendar=earnings_calendar,
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


class _FixedEarningsCalendar:
    def __init__(self, day: date):
        self.day = day

    def next_earnings_date(self, ticker: str):
        return self.day


def test_eligible_candidates_drops_contracts_expiring_before_earnings(
    tmp_path: Path,
) -> None:
    # Pre-earnings (IV-ramp) play: a contract must OUTLIVE the print to carry its
    # event vega, so the one expiring before the earnings date is dropped while
    # the one expiring after it survives.
    from trading_agent.execution.orchestrator import CycleResult

    base = _mandate()
    opts = base.options.model_copy(update={"pre_earnings_exit_trading_days": 2})
    mandate = base.model_copy(update={"options": opts})

    before = "US.EXAMPLE260622C00005000"  # expires 06-22, BEFORE earnings 06-26
    after = "US.EXAMPLE260710C00005000"  # expires 07-10, AFTER earnings
    cycle = _cycle(
        tmp_path,
        broker=FakeBroker(),
        market=FakeMarket(chain_codes=[before, after]),
        committee=_committee([]),
        mandate=mandate,
        earnings_calendar=_FixedEarningsCalendar(date(2026, 6, 26)),
    )
    codes = {
        c.option_code
        for c in cycle._eligible_candidates(NOW, "EXAMPLE", CycleResult())
    }
    assert before not in codes
    assert after in codes


def test_eligible_candidates_keeps_all_when_pre_earnings_disabled(
    tmp_path: Path,
) -> None:
    # Strategy off (default mandate) -> the earnings-DTE filter is inert.
    from trading_agent.execution.orchestrator import CycleResult

    before = "US.EXAMPLE260622C00005000"
    after = "US.EXAMPLE260710C00005000"
    cycle = _cycle(
        tmp_path,
        broker=FakeBroker(),
        market=FakeMarket(chain_codes=[before, after]),
        committee=_committee([]),
    )
    codes = {
        c.option_code
        for c in cycle._eligible_candidates(NOW, "EXAMPLE", CycleResult())
    }
    assert codes == {before, after}


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


def test_entry_persists_catalyst_window_end(tmp_path: Path) -> None:
    cycle = _cycle(
        tmp_path, broker=FakeBroker(), market=FakeMarket(),
        committee=_committee(_open_responses()),
    )
    cycle.run_once(["EXAMPLE"])
    row = cycle.position_store.open_positions()[0]
    # _PROPOSAL's window is "2026-06-10/2026-06-20".
    assert row["catalyst_window_end"] == "2026-06-20"


def test_parse_window_end_variants() -> None:
    assert _parse_window_end("2026-06-10/2026-06-20") == date(2026, 6, 20)
    assert _parse_window_end("2026-06-20") == date(2026, 6, 20)  # single date
    assert _parse_window_end("not a date") is None
    assert _parse_window_end("") is None
    assert _parse_window_end(None) is None


def test_eligible_candidates_keeps_ten_per_side(tmp_path: Path) -> None:
    expiry = date(2026, 6, 26)
    codes = [build_us_option_code("EXAMPLE", expiry, "call", 3.0 + 0.5 * i) for i in range(12)]
    codes += [build_us_option_code("EXAMPLE", expiry, "put", 3.0 + 0.5 * i) for i in range(12)]
    market = FakeMarket(chain_codes=codes)
    cycle = _cycle(tmp_path, broker=FakeBroker(), market=market, committee=_committee([]))

    cands = cycle._eligible_candidates(NOW, "EXAMPLE", CycleResult(), spot=0.0)

    calls = [c for c in cands if c.option_side == "call"]
    puts = [c for c in cands if c.option_side == "put"]
    assert len(calls) == 10  # capped per side, not 12
    assert len(puts) == 10  # the put side is never starved out


def test_eligible_candidates_annotate_delta_and_breakeven(tmp_path: Path) -> None:
    code = build_us_option_code("EXAMPLE", date(2026, 6, 26), "call", 5.0)
    market = FakeMarket(chain_codes=[code], spot=5.0, iv=0.5, bid=0.19, ask=0.21)
    cycle = _cycle(tmp_path, broker=FakeBroker(), market=market, committee=_committee([]))

    cand = cycle._eligible_candidates(NOW, "EXAMPLE", CycleResult(), spot=5.0)[0]

    assert cand.delta is not None and 0 < cand.delta < 1  # ATM call ~0.5
    # Call breakeven = strike + ask = 5.21, a +4.2% move from spot 5.0.
    assert cand.breakeven_move_pct == 4.2


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


def test_opening_window_blocks_new_entries(tmp_path: Path) -> None:
    # 13:35 UTC = 9:35 ET, inside the paper mandate's 15-minute no-entry
    # window. The committee (empty mock) must never be invoked.
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
        position_store=PositionStore(tmp_path / "positions.sqlite"),
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        now_fn=lambda: datetime(2026, 6, 2, 13, 35, tzinfo=timezone.utc),
        order_poll_interval_seconds=0.1,
        sleep_fn=lambda _s: None,
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []
    regime = [r for r in result.rejected if r["stage"] == "regime"]
    assert regime and "open" in regime[0]["reasons"][0]


def test_vix_above_cap_blocks_new_entries(tmp_path: Path) -> None:
    # FakeMarket reports its spot for every symbol including _VIX; spot=40
    # exceeds the mandate's 35 entry cap, so all entries are skipped.
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(spot=40.0),
        committee=_committee([]),
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []
    regime = [r for r in result.rejected if r["stage"] == "regime"]
    assert regime and "VIX" in regime[0]["reasons"][0]


def test_exhausted_llm_budget_skips_committee(tmp_path: Path) -> None:
    from trading_agent.domain.calendar import market_date as _md
    from trading_agent.storage.budget import DailyTokenBudget

    budget = DailyTokenBudget(tmp_path / "budget.json", limit_tokens=100)
    budget.add(100, _md(NOW))  # today's budget already spent
    broker = FakeBroker()
    cycle = _cycle(
        tmp_path, broker=broker, market=FakeMarket(), committee=_committee([])
    )
    cycle.llm_budget = budget

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []
    assert any(r["stage"] == "llm_budget" for r in result.rejected)


def test_decision_cache_skips_committee_on_unchanged_inputs(tmp_path: Path) -> None:
    from trading_agent.storage.decisions import DecisionCache

    cache = DecisionCache(tmp_path / "decisions.json")
    # First cycle: committee runs (consumes the 5 mock responses), decision
    # ("reject" via hold) is cached.
    cycle1 = _cycle(
        tmp_path,
        broker=FakeBroker(),
        market=FakeMarket(),
        committee=_committee(
            ["c", "o", "fine", "fine", json.dumps({"decision": "hold", "rationale": "wait"})]
        ),
    )
    cycle1.decision_cache = cache
    cycle1.run_once(["EXAMPLE"])

    # Second cycle, same evidence + same candidates: an empty MockLLMClient
    # would raise on any call, proving the committee was never invoked.
    cycle2 = _cycle(
        tmp_path, broker=FakeBroker(), market=FakeMarket(), committee=_committee([])
    )
    cycle2.decision_cache = cache
    result = cycle2.run_once(["EXAMPLE"])

    assert result.errors == []  # no committee call -> no mock exhaustion error


def test_same_underlying_second_strike_is_rejected(tmp_path: Path) -> None:
    # Already long EXAMPLE at strike 5; the committee proposes ANOTHER EXAMPLE
    # strike. The concentration rule must block it even though the option code
    # differs (no duplicate), keeping the 2-position book on distinct names.
    other_strike = "US.EXAMPLE260626C00007000"
    broker = FakeBroker(positions=[{"code": other_strike, "qty": 1}])
    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_open(store, other_strike)
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(),
        committee=_committee(_open_responses()),  # proposes OPTION_CODE (strike 5)
        store=store,
    )

    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == []
    rejected = [r for r in result.rejected if r["stage"] == "risk_gate"]
    assert rejected and any(
        "underlying" in reason for reason in rejected[0]["reasons"]
    )


def test_orphan_broker_position_is_adopted(tmp_path: Path) -> None:
    # Broker holds an option the ledger has no record of (crash between fill
    # and ledger write): reconcile must adopt it so the exit engine manages it.
    broker = FakeBroker(
        positions=[{"code": OPTION_CODE, "qty": 1, "cost_price": 0.25}]
    )
    store = PositionStore(tmp_path / "positions.sqlite")
    cycle = _cycle(
        tmp_path, broker=broker, market=FakeMarket(), committee=_committee([]), store=store
    )

    result = cycle.run_once([])

    assert result.adopted == [OPTION_CODE]
    adopted = store.get(OPTION_CODE)
    assert adopted is not None and adopted["status"] == "open"
    assert adopted["entry_price"] == 0.25
    assert adopted["take_profit_pct"] == 100.0 and adopted["stop_loss_pct"] == 50.0


def test_stock_holdings_are_not_adopted(tmp_path: Path) -> None:
    broker = FakeBroker(positions=[{"code": "US.AAPL", "qty": 10, "cost_price": 200}])
    store = PositionStore(tmp_path / "positions.sqlite")
    cycle = _cycle(
        tmp_path, broker=broker, market=FakeMarket(), committee=_committee([]), store=store
    )

    result = cycle.run_once([])

    assert result.adopted == []
    assert store.get("US.AAPL") is None


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

    def fetch_evidence(self, ticker, query_name=None):
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
    # Three small losses (-$3 each): below the $35 daily stop, but they hit the
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


def test_commission_reduces_realized_ledger_pnl(tmp_path: Path) -> None:
    # Live sets a per-side commission; equity replay must be NET of fees: a
    # +$100 gross win at $0.95/side nets 100 - 1.90 = 98.10.
    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_closed(store, "US.WIN260626C00005000", entry=0.20, exit_price=1.20)

    mandate = _mandate()
    execution = mandate.execution.model_copy(
        update={"commission_per_contract_usd": 0.95}
    )
    mandate = mandate.model_copy(update={"execution": execution})
    cycle = PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=FakeMarket(),
        broker=FakeBroker(),
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

    equity, peak = cycle._equity_and_peak()
    assert equity == 198.1  # 100 initial + 100 gross - 1.90 round-trip fees
    assert peak == 198.1


def test_paper_mandate_keeps_zero_commission(tmp_path: Path) -> None:
    # Paper must mirror the broker sim (which charges nothing); the yaml pins 0.
    assert _mandate().execution.commission_per_contract_usd == 0


def test_compounding_win_unlocks_bigger_contracts(tmp_path: Path) -> None:
    # A realized +$100 win doubles equity; the $65 contract cap scales to $130,
    # so a $0.80-ask contract (cost ~$81, rejected at the $65 base) becomes
    # tradeable. The paper mandate has compounding: true.
    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_closed(store, "US.WIN260626C00005000", entry=0.20, exit_price=1.20)  # +100

    broker = FakeBroker()
    cycle = _cycle(
        tmp_path,
        broker=broker,
        market=FakeMarket(bid=0.78, ask=0.80),
        committee=_committee(_open_responses()),
        store=store,
    )
    result = cycle.run_once(["EXAMPLE"])

    assert broker.placed == [("buy", OPTION_CODE)]
    assert result.entries
    assert cycle.active_mandate.options.max_contract_cost_usd == 130.0


def test_fixed_mandate_rejects_what_compounding_allows(tmp_path: Path) -> None:
    # Same +$100 win and same $0.80-ask contract, but compounding off: the
    # static $65 cap rejects it at the liquidity stage.
    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_closed(store, "US.WIN260626C00005000", entry=0.20, exit_price=1.20)

    mandate = _mandate()
    account = mandate.account.model_copy(update={"compounding": False})
    mandate = mandate.model_copy(update={"account": account})
    broker = FakeBroker()
    cycle = PaperTradingCycle(
        mandate=mandate,
        account_id=123,
        market=FakeMarket(bid=0.78, ask=0.80),
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
    # Win +$100 (peak 200) then lose $120 back (equity 80). Versus initial
    # capital that is still a GAIN... no, equity 80 < 100; the point is the stop
    # is measured from the PEAK: a $120 drawdown is beyond the scaled stop
    # (50 x 2 = 100), so the circuit breaker must halt. Both closes happened
    # "yesterday" relative to the cycle clock, so the daily stop stays quiet and
    # the drawdown logic is what trips.
    from datetime import timedelta

    store = PositionStore(tmp_path / "positions.sqlite")
    _seed_closed(store, "US.WIN260626C00005000", entry=0.20, exit_price=1.20)  # +100
    _seed_closed(store, "US.LOSS260626C00005000", entry=1.40, exit_price=0.20)  # -120

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
    # Three realized losses of -$15 each, closed today => daily P/L -$45 trips
    # the $35 daily loss stop (checked before the $50 drawdown stop).
    for code in (
        "US.AAA260626C00005000",
        "US.BBB260626C00005000",
        "US.CCC260626C00005000",
    ):
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
