from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from trading_agent.brokers.moomoo import MoomooConnection
from trading_agent.data.moomoo_market import (
    MoomooMarket,
    best_bid_ask,
    build_universe_filters,
    parse_filter_rows,
)
from trading_agent.domain.risk import Mandate

NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)


class FakeFrame:
    def __init__(self, records: list[dict[str, object]]):
        self.records = records

    def to_dict(self, *, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return self.records


class FakeSimpleFilter:
    def __init__(self) -> None:
        self.stock_field = None
        self.is_no_filter = None
        self.filter_min = None
        self.filter_max = None


def _stock_field() -> SimpleNamespace:
    return SimpleNamespace(CUR_PRICE="CUR_PRICE", MARKET_VAL="MARKET_VAL", TURNOVER="TURNOVER")


def _universe() -> "Mandate":
    return Mandate.load(Path("config/mandate.paper.yaml")).universe


def test_build_universe_filters_maps_mandate() -> None:
    sdk = SimpleNamespace(SimpleFilter=FakeSimpleFilter, StockField=_stock_field())
    filters = build_universe_filters(_universe(), sdk)

    by_field = {f.stock_field: f for f in filters}
    assert by_field["CUR_PRICE"].filter_min == 2
    assert by_field["MARKET_VAL"].filter_min == 100000000
    assert by_field["MARKET_VAL"].filter_max == 5000000000
    assert by_field["TURNOVER"].filter_min == 5000000
    assert all(f.is_no_filter is False for f in filters)


def test_parse_filter_rows_extracts_fields() -> None:
    rows = [
        SimpleNamespace(
            stock_code="US.AAA", stock_name="Alpha", cur_price=5.0, market_val=2e8, turnover=9e6
        )
    ]
    assert parse_filter_rows(rows) == [
        {"code": "US.AAA", "name": "Alpha", "cur_price": 5.0, "market_val": 2e8, "turnover": 9e6}
    ]


def test_best_bid_ask_reads_top_of_book() -> None:
    book = {"Bid": [(0.19, 10), (0.18, 5)], "Ask": [(0.21, 8), (0.22, 3)]}
    assert best_bid_ask(book) == (0.19, 0.21)


def test_best_bid_ask_handles_empty_side() -> None:
    assert best_bid_ask({"Bid": [], "Ask": []}) == (0.0, 0.0)


class FakeQuoteContext:
    def __init__(self) -> None:
        self.closed = False

    def get_stock_filter(self, **kwargs):
        rows = [
            SimpleNamespace(
                stock_code="US.AAA", stock_name="Alpha", cur_price=5.0, market_val=2e8, turnover=9e6
            )
        ]
        return 0, (True, 1, rows)

    def subscribe(self, *args, **kwargs):
        return 0, None

    def get_order_book(self, code):
        return 0, {"Bid": [(0.19, 10)], "Ask": [(0.21, 8)]}

    def get_market_snapshot(self, codes):
        return 0, FakeFrame([{"option_open_interest": 200, "volume": 30}])

    def close(self) -> None:
        self.closed = True


def _fake_sdk(ctx: FakeQuoteContext) -> SimpleNamespace:
    return SimpleNamespace(
        RET_OK=0,
        OpenQuoteContext=lambda **kwargs: ctx,
        Market=SimpleNamespace(US="US"),
        SimpleFilter=FakeSimpleFilter,
        StockField=_stock_field(),
        SubType=SimpleNamespace(QUOTE="QUOTE", ORDER_BOOK="ORDER_BOOK"),
    )


def test_scan_small_caps_parses_and_closes(monkeypatch) -> None:
    ctx = FakeQuoteContext()
    monkeypatch.setattr(MoomooMarket, "_sdk", staticmethod(lambda: _fake_sdk(ctx)))
    market = MoomooMarket(MoomooConnection(host="127.0.0.1", port=11111))

    candidates = market.scan_small_caps(_universe())

    assert candidates == [
        {"code": "US.AAA", "name": "Alpha", "cur_price": 5.0, "market_val": 2e8, "turnover": 9e6}
    ]
    assert ctx.closed is True


def test_option_quote_builds_snapshot_and_closes(monkeypatch) -> None:
    ctx = FakeQuoteContext()
    monkeypatch.setattr(MoomooMarket, "_sdk", staticmethod(lambda: _fake_sdk(ctx)))
    market = MoomooMarket(MoomooConnection(host="127.0.0.1", port=11111))

    quote = market.option_quote(
        option_code="US.EXAMPLE260626C00005000",
        expiry=date(2026, 6, 26),
        lot_size=100,
        now=NOW,
    )

    assert quote.bid == 0.19
    assert quote.ask == 0.21
    assert quote.open_interest == 200
    assert quote.daily_volume == 30
    assert quote.lot_size == 100
    assert quote.expiry == date(2026, 6, 26)
    assert ctx.closed is True
