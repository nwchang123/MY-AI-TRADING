from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from trading_agent.brokers.moomoo import MoomooConnection
from trading_agent.data.moomoo_market import (
    MoomooMarket,
    best_bid_ask,
    build_universe_filters,
    build_us_option_code,
    occ_to_moomoo_code,
    parse_filter_rows,
    parse_us_option_code,
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
        self.sort = None


class FakeAccumulateFilter(FakeSimpleFilter):
    def __init__(self) -> None:
        super().__init__()
        self.days = None


def _stock_field() -> SimpleNamespace:
    return SimpleNamespace(
        CUR_PRICE="CUR_PRICE",
        MARKET_VAL="MARKET_VAL",
        TURNOVER="TURNOVER",
        VOLUME_RATIO="VOLUME_RATIO",
        CHANGE_RATE="CHANGE_RATE",
    )


def _sort_dir() -> SimpleNamespace:
    return SimpleNamespace(ASCEND="ASCEND", DESCEND="DESCEND", NONE="NONE")


def _universe() -> "Mandate":
    return Mandate.load(Path("config/mandate.paper.yaml")).universe


def test_build_universe_filters_maps_mandate() -> None:
    sdk = SimpleNamespace(
        SimpleFilter=FakeSimpleFilter,
        AccumulateFilter=FakeAccumulateFilter,
        StockField=_stock_field(),
        SortDir=_sort_dir(),
    )
    filters = build_universe_filters(_universe(), sdk)

    by_field = {f.stock_field: f for f in filters}
    assert by_field["CUR_PRICE"].filter_min == 2
    assert by_field["MARKET_VAL"].filter_min == 100000000
    assert by_field["MARKET_VAL"].filter_max == 5000000000
    # TURNOVER is accumulate-class: SimpleFilter is rejected by OpenD.
    assert isinstance(by_field["TURNOVER"], FakeAccumulateFilter)
    assert by_field["TURNOVER"].filter_min == 5000000
    assert by_field["TURNOVER"].days == 1
    # VOLUME_RATIO is retrieve-and-sort only: the server orders the FULL match
    # set by it, so pagination sees the most abnormal names first.
    assert not isinstance(by_field["VOLUME_RATIO"], FakeAccumulateFilter)
    assert by_field["VOLUME_RATIO"].is_no_filter is True
    assert by_field["VOLUME_RATIO"].sort == "DESCEND"
    assert by_field["VOLUME_RATIO"].filter_min is None
    # CHANGE_RATE rides along unconstrained for context.
    assert isinstance(by_field["CHANGE_RATE"], FakeAccumulateFilter)
    assert by_field["CHANGE_RATE"].is_no_filter is True
    assert by_field["CHANGE_RATE"].days == 1
    constrained = [by_field["CUR_PRICE"], by_field["MARKET_VAL"], by_field["TURNOVER"]]
    assert all(f.is_no_filter is False for f in constrained)


def test_parse_filter_rows_extracts_fields() -> None:
    rows = [
        SimpleNamespace(
            stock_code="US.AAA",
            stock_name="Alpha",
            cur_price=5.0,
            market_val=2e8,
            turnover=9e6,
            volume_ratio=3.2,
        )
    ]
    assert parse_filter_rows(rows) == [
        {
            "code": "US.AAA",
            "name": "Alpha",
            "cur_price": 5.0,
            "market_val": 2e8,
            "turnover": 9e6,
            "volume_ratio": 3.2,
            "change_rate": None,
        }
    ]


def test_parse_filter_rows_reads_tuple_keyed_accumulate_fields() -> None:
    # Real FilterStockData stores accumulate fields under a (field, days) tuple
    # key, not a plain attribute (verified live: row.turnover raises).
    row = SimpleNamespace(
        stock_code="US.TRLV", stock_name="Trulieve", cur_price=11.885, market_val=2.27e9
    )
    row.__dict__[("turnover", 1)] = 6569147.589
    row.__dict__[("change_rate", 1)] = -2.4
    parsed = parse_filter_rows([row])[0]
    assert parsed["turnover"] == 6569147.589
    assert parsed["change_rate"] == -2.4


def test_build_us_option_code_emits_no_pad_strike() -> None:
    # Verified live against OpenD: the broker accepts the no-pad strike form and
    # rejects the OCC 8-digit-padded form.
    code = build_us_option_code("AAPL", date(2026, 7, 17), "call", 310.0)
    assert code == "US.AAPL260717C310000"
    assert build_us_option_code("US.NVDA", date(2026, 6, 26), "put", 5.0) == (
        "US.NVDA260626P5000"
    )


def test_occ_to_moomoo_strips_strike_padding() -> None:
    # CBOE/Tradier 'option' field is OCC with an 8-digit padded strike.
    assert occ_to_moomoo_code("AAPL260717C00310000") == "US.AAPL260717C310000"
    assert occ_to_moomoo_code("nvda260626p00005000") == "US.NVDA260626P5000"


def test_option_code_round_trips_through_parse() -> None:
    code = build_us_option_code("AAPL", date(2026, 7, 17), "call", 310.0)
    root, expiry, side, strike = parse_us_option_code(code)
    assert (root, expiry, side, strike) == ("AAPL", date(2026, 7, 17), "call", 310.0)
    # Parser still tolerates a zero-padded (OCC-style) strike.
    assert parse_us_option_code("US.EXAMPLE260626C00005000")[3] == 5.0


def test_best_bid_ask_reads_top_of_book() -> None:
    book = {"Bid": [(0.19, 10), (0.18, 5)], "Ask": [(0.21, 8), (0.22, 3)]}
    assert best_bid_ask(book) == (0.19, 0.21)


def test_best_bid_ask_handles_empty_side() -> None:
    assert best_bid_ask({"Bid": [], "Ask": []}) == (0.0, 0.0)


def _scan_row(code: str) -> SimpleNamespace:
    return SimpleNamespace(
        stock_code=code, stock_name=code, cur_price=5.0, market_val=2e8, turnover=9e6
    )


class FakeQuoteContext:
    def __init__(self, pages: list[tuple[bool, list]] | None = None) -> None:
        self.closed = False
        self.filter_calls: list[tuple[int, int]] = []
        self.pages = pages if pages is not None else [(True, [_scan_row("US.AAA")])]

    def get_stock_filter(self, *, market, filter_list, begin, num):
        self.filter_calls.append((begin, num))
        last_page, rows = self.pages[len(self.filter_calls) - 1]
        return 0, (last_page, sum(len(r) for _, r in self.pages), rows)

    def get_owner_plate(self, code_list):
        return 0, FakeFrame(
            [
                {"code": "US.AAA", "plate_type": "INDUSTRY", "plate_name": "Gold"},
                {"code": "US.AAA", "plate_type": "CONCEPT", "plate_name": "Meme"},
                {"code": "US.BBB", "plate_type": "REGION", "plate_name": "Nevada"},
            ]
        )

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
        AccumulateFilter=FakeAccumulateFilter,
        StockField=_stock_field(),
        SortDir=_sort_dir(),
        SubType=SimpleNamespace(QUOTE="QUOTE", ORDER_BOOK="ORDER_BOOK"),
    )


def test_scan_small_caps_parses_and_closes(monkeypatch) -> None:
    ctx = FakeQuoteContext()
    monkeypatch.setattr(MoomooMarket, "_sdk", staticmethod(lambda: _fake_sdk(ctx)))
    market = MoomooMarket(MoomooConnection(host="127.0.0.1", port=11111))

    candidates = market.scan_small_caps(_universe())

    assert candidates == [
        {
            "code": "US.AAA",
            "name": "US.AAA",
            "cur_price": 5.0,
            "market_val": 2e8,
            "turnover": 9e6,
            "volume_ratio": None,
            "change_rate": None,
        }
    ]
    assert ctx.closed is True


def test_scan_small_caps_paginates_until_last_page(monkeypatch) -> None:
    pages = [
        (False, [_scan_row("US.AAA"), _scan_row("US.BBB")]),
        (True, [_scan_row("US.CCC")]),
    ]
    ctx = FakeQuoteContext(pages=pages)
    monkeypatch.setattr(MoomooMarket, "_sdk", staticmethod(lambda: _fake_sdk(ctx)))
    market = MoomooMarket(MoomooConnection(host="127.0.0.1", port=11111))

    candidates = market.scan_small_caps(_universe(), page_size=2)

    assert [c["code"] for c in candidates] == ["US.AAA", "US.BBB", "US.CCC"]
    # begin advances by rows received, not by page size assumptions.
    assert ctx.filter_calls == [(0, 2), (2, 2)]
    assert ctx.closed is True


def test_scan_small_caps_respects_max_rows(monkeypatch) -> None:
    pages = [
        (False, [_scan_row("US.AAA"), _scan_row("US.BBB")]),
        (True, [_scan_row("US.CCC")]),
    ]
    ctx = FakeQuoteContext(pages=pages)
    monkeypatch.setattr(MoomooMarket, "_sdk", staticmethod(lambda: _fake_sdk(ctx)))
    market = MoomooMarket(MoomooConnection(host="127.0.0.1", port=11111))

    candidates = market.scan_small_caps(_universe(), page_size=2, max_rows=2)

    assert [c["code"] for c in candidates] == ["US.AAA", "US.BBB"]
    assert ctx.filter_calls == [(0, 2)]  # cap reached: no second request


def test_industry_plates_keeps_only_industry_rows(monkeypatch) -> None:
    ctx = FakeQuoteContext()
    monkeypatch.setattr(MoomooMarket, "_sdk", staticmethod(lambda: _fake_sdk(ctx)))
    market = MoomooMarket(MoomooConnection(host="127.0.0.1", port=11111))

    plates = market.industry_plates(["US.AAA", "US.BBB"])

    assert plates == {"US.AAA": "Gold"}
    assert ctx.closed is True


def test_industry_plates_empty_input_skips_the_call() -> None:
    market = MoomooMarket(MoomooConnection(host="127.0.0.1", port=11111))
    assert market.industry_plates([]) == {}


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
