from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from trading_agent.brokers.moomoo import MoomooConnection
from trading_agent.domain.risk import QuoteSnapshot, UniverseMandate

# US equity options trade in lots of 100 shares.
US_OPTION_LOT_SIZE = 100

# Moomoo US option code: US.{ROOT}{YYMMDD}{C|P}{strike*1000}. Unlike the OCC
# symbol, Moomoo does NOT zero-pad the strike to 8 digits -- place_order rejects
# the padded form with "Cannot find ... in US Stocks". Verified live 2026-06-11:
# US.AAPL260717C310000 is accepted and echoed back by position_list_query;
# US.AAPL260717C00310000 is rejected. The parser accepts any strike width (\d+)
# so an OCC-padded code still reads, but the builders below always emit the
# canonical no-pad form that the broker accepts.
_OPTION_CODE_RE = re.compile(r"^US\.([A-Z]+)(\d{6})([CP])(\d+)$")

# OCC symbol (e.g. CBOE 'option' field 'AAPL260717C00310000'): strike IS
# zero-padded to 8 digits. Used to convert free-data-source chains to Moomoo.
_OCC_CODE_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


class MoomooMarketError(RuntimeError):
    """Raised when OpenD or the quote SDK rejects a market-data request."""


def parse_us_option_code(option_code: str) -> tuple[str, date, str, float]:
    """Parse a Moomoo US option code, e.g. 'US.NVDA260626C5000'.

    Returns (underlying, expiry, side, strike). Lets the agent derive expiry
    without an extra chain call. Tolerates a zero-padded strike (OCC form) too.
    Raises ValueError on an unrecognized code.
    """

    match = _OPTION_CODE_RE.match(option_code)
    if match is None:
        raise ValueError(f"unrecognized US option code: {option_code}")
    root, ymd, call_put, strike = match.groups()
    expiry = date(2000 + int(ymd[0:2]), int(ymd[2:4]), int(ymd[4:6]))
    side = "call" if call_put == "C" else "put"
    return root, expiry, side, int(strike) / 1000


def build_us_option_code(
    underlying: str, expiry: date, side: str, strike: float
) -> str:
    """Build the canonical Moomoo US option code the broker accepts.

    Strike is encoded as ``int(strike * 1000)`` with NO zero-padding. The inverse
    of :func:`parse_us_option_code`.
    """

    if side not in {"call", "put"}:
        raise ValueError("side must be 'call' or 'put'")
    call_put = "C" if side == "call" else "P"
    strike_milli = int(round(strike * 1000))
    root = underlying.upper().removeprefix("US.")
    return f"US.{root}{expiry:%y%m%d}{call_put}{strike_milli}"


def occ_to_moomoo_code(occ_symbol: str) -> str:
    """Convert an OCC option symbol to a Moomoo code.

    CBOE/Tradier emit OCC symbols with an 8-digit zero-padded strike
    (``AAPL260717C00310000``); Moomoo wants the strike without leading zeros
    (``US.AAPL260717C310000``). Raises ValueError on an unrecognized symbol.
    """

    match = _OCC_CODE_RE.match(occ_symbol.strip().upper())
    if match is None:
        raise ValueError(f"unrecognized OCC option symbol: {occ_symbol}")
    root, ymd, call_put, strike8 = match.groups()
    return f"US.{root}{ymd}{call_put}{int(strike8)}"


def build_universe_filters(universe: UniverseMandate, sdk: Any) -> list[Any]:
    """Translate the universe mandate into Moomoo filter objects.

    Pure with respect to the SDK module, so it can be tested with a fake ``sdk``.
    TURNOVER is an accumulate-class field: passing it via SimpleFilter makes
    get_stock_filter fail with "This filter field is not supported" (verified
    live 2026-06-11), so it goes through AccumulateFilter with a 1-day window.

    VOLUME_RATIO is included unconstrained (is_no_filter) purely to retrieve the
    value and to have OpenD sort the FULL match set by it server-side, so the
    first page already holds the names with the most abnormal activity today --
    the catalyst signal this strategy trades, unlike raw turnover whose ranking
    barely changes day to day. CHANGE_RATE rides along the same way for context.
    Verified live 2026-06-12: OpenD accepts both no-filter fields, returns rows
    sorted by volume_ratio descending, and CHANGE_RATE days=1 populates.
    """

    simple_filter = sdk.SimpleFilter
    field = sdk.StockField

    def make(stock_field: Any, fmin: float | None = None, fmax: float | None = None) -> Any:
        item = simple_filter()
        item.stock_field = stock_field
        item.is_no_filter = False
        if fmin is not None:
            item.filter_min = fmin
        if fmax is not None:
            item.filter_max = fmax
        return item

    turnover = sdk.AccumulateFilter()
    turnover.stock_field = field.TURNOVER
    turnover.is_no_filter = False
    turnover.filter_min = universe.min_average_daily_turnover_usd
    turnover.days = 1

    volume_ratio = simple_filter()
    volume_ratio.stock_field = field.VOLUME_RATIO
    volume_ratio.is_no_filter = True
    volume_ratio.sort = sdk.SortDir.DESCEND

    change_rate = sdk.AccumulateFilter()
    change_rate.stock_field = field.CHANGE_RATE
    change_rate.is_no_filter = True
    change_rate.days = 1

    return [
        make(field.CUR_PRICE, fmin=universe.min_underlying_price_usd),
        make(
            field.MARKET_VAL,
            fmin=universe.min_market_cap_usd,
            fmax=universe.max_market_cap_usd,
        ),
        turnover,
        volume_ratio,
        change_rate,
    ]


def _accumulate_value(row: Any, field_name: str) -> Any:
    """Read an accumulate-filter value off a FilterStockData row.

    Verified live 2026-06-11: simple-filter fields are plain attributes, but
    accumulate fields are stored under a ``(field, days)`` tuple key in the
    row's ``__dict__`` (e.g. ``('turnover', 1)``), so ``row.turnover`` raises
    AttributeError. Falls back to the plain attribute for fake rows in tests.
    """

    data = getattr(row, "__dict__", None) or {}
    for key, value in data.items():
        if isinstance(key, tuple) and key and key[0] == field_name:
            return value
    return getattr(row, field_name, None)


def parse_filter_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Normalize FilterStockData rows into plain dicts."""

    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "code": getattr(row, "stock_code", None),
                "name": getattr(row, "stock_name", None),
                "cur_price": getattr(row, "cur_price", None),
                "market_val": getattr(row, "market_val", None),
                "turnover": _accumulate_value(row, "turnover"),
                "volume_ratio": getattr(row, "volume_ratio", None),
                "change_rate": _accumulate_value(row, "change_rate"),
            }
        )
    return parsed


def best_bid_ask(order_book: dict[str, Any]) -> tuple[float, float]:
    """Best bid/ask from a Moomoo order book ('Bid'/'Ask' price-volume tuples)."""

    bids = order_book.get("Bid") or []
    asks = order_book.get("Ask") or []
    bid = float(bids[0][0]) if bids else 0.0
    ask = float(asks[0][0]) if asks else 0.0
    return bid, ask


def to_quote_snapshot(
    *,
    option_code: str,
    bid: float,
    ask: float,
    open_interest: int,
    daily_volume: int,
    lot_size: int,
    expiry: date,
    observed_at: datetime,
) -> QuoteSnapshot:
    return QuoteSnapshot(
        option_code=option_code,
        bid=bid,
        ask=ask,
        open_interest=open_interest,
        daily_volume=daily_volume,
        lot_size=lot_size,
        expiry=expiry,
        observed_at=observed_at,
    )


class MoomooMarket:
    """Quote-side adapter: small-cap screener, option chain, and live quotes.

    Live methods reuse one ``OpenQuoteContext`` per adapter instance.
    Deterministic parsing lives in the module-level helpers above so it can be
    unit tested without OpenD.
    """

    def __init__(self, connection: MoomooConnection):
        self.connection = connection
        self._quote_ctx: Any | None = None

    def scan_small_caps(
        self, universe: UniverseMandate, *, page_size: int = 200, max_rows: int = 1000
    ) -> list[dict[str, Any]]:
        """All mandate-passing rows, paginated past OpenD's 200-row page limit.

        The US small-cap mandate matches far more than one page, so stopping at
        the first page would rank within an arbitrary slice of the universe.
        ``max_rows`` bounds the walk (get_stock_filter is rate-limited to 10
        requests per 30s; 1000 rows = 5 calls) -- combined with the server-side
        VOLUME_RATIO sort the cap drops only the quietest tail.
        """

        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        parsed: list[dict[str, Any]] = []
        begin = 0
        filters = build_universe_filters(universe, sdk)
        while len(parsed) < max_rows:
            ret, data = quote_ctx.get_stock_filter(
                market=sdk.Market.US,
                filter_list=filters,
                begin=begin,
                num=page_size,
            )
            self._require_ok(sdk, ret, data, "get_stock_filter")
            last_page, _all_count, rows = data
            parsed.extend(parse_filter_rows(rows))
            begin += len(rows)
            if last_page or not rows:
                break
        return parsed[:max_rows]

    def industry_plates(self, codes: list[str]) -> dict[str, str]:
        """Industry plate name per stock code, from one batched get_owner_plate.

        Used by universe selection to keep one hot theme from monopolizing the
        candidate list. Codes get_owner_plate cannot resolve are simply absent
        (the caller treats unknown-industry names as uncapped). The API accepts
        at most 200 codes per call, which the probe-limit-sized input respects.
        """

        if not codes:
            return {}
        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        ret, data = quote_ctx.get_owner_plate(code_list=codes)
        self._require_ok(sdk, ret, data, "get_owner_plate")
        plates: dict[str, str] = {}
        for row in data.to_dict(orient="records"):
            if str(row.get("plate_type") or "").upper() != "INDUSTRY":
                continue
            code = row.get("code")
            name = row.get("plate_name")
            if code and name and code not in plates:
                plates[str(code)] = str(name)
        return plates

    def option_expirations(self, code: str) -> list[dict[str, Any]]:
        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        ret, data = quote_ctx.get_option_expiration_date(code=code)
        self._require_ok(sdk, ret, data, "get_option_expiration_date")
        return data.to_dict(orient="records")

    def is_optionable(self, code: str) -> bool:
        return bool(self.option_expirations(code))

    def option_chain(
        self,
        code: str,
        start: str,
        end: str,
        option_type: str = "ALL",
    ) -> list[dict[str, Any]]:
        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        ret, data = quote_ctx.get_option_chain(
            code=code, start=start, end=end, option_type=option_type
        )
        self._require_ok(sdk, ret, data, "get_option_chain")
        return data.to_dict(orient="records")

    def is_listed_option(self, option_code: str) -> bool:
        """Confirm a proposed option code is a real, listed contract.

        Guards against an LLM hallucinating a plausible-but-nonexistent code:
        the chain for the parsed underlying and expiry must contain the code.
        """

        underlying, expiry, _, _ = parse_us_option_code(option_code)
        day = expiry.isoformat()
        rows = self.option_chain(f"US.{underlying}", day, day)
        return any(row.get("code") == option_code for row in rows)

    def option_quote(
        self,
        *,
        option_code: str,
        expiry: date,
        lot_size: int,
        now: datetime | None = None,
    ) -> QuoteSnapshot:
        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        observed_at = now or datetime.now(timezone.utc)
        ret, data = quote_ctx.subscribe(
            [option_code], [sdk.SubType.QUOTE, sdk.SubType.ORDER_BOOK]
        )
        self._require_ok(sdk, ret, data, "subscribe")

        ret, order_book = quote_ctx.get_order_book(option_code)
        self._require_ok(sdk, ret, order_book, "get_order_book")
        bid, ask = best_bid_ask(order_book)

        ret, snapshot = quote_ctx.get_market_snapshot([option_code])
        self._require_ok(sdk, ret, snapshot, "get_market_snapshot")
        row = snapshot.to_dict(orient="records")[0]

        return to_quote_snapshot(
            option_code=option_code,
            bid=bid,
            ask=ask,
            open_interest=int(row.get("option_open_interest") or 0),
            daily_volume=int(row.get("volume") or 0),
            lot_size=lot_size,
            expiry=expiry,
            observed_at=observed_at,
        )

    def stock_snapshot(self, ticker: str) -> dict[str, Any]:
        """Get real-time stock snapshot from Moomoo OpenD.

        Returns price, volume, change, and other market data.
        This is REAL-TIME data, unlike CBOE/Yahoo which are delayed ~15 min.
        """
        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        # Add US. prefix if not present
        code = ticker.upper() if ticker.upper().startswith("US.") else f"US.{ticker.upper()}"

        # Subscribe to get real-time quotes
        ret, data = quote_ctx.subscribe([code], [sdk.SubType.QUOTE])
        self._require_ok(sdk, ret, data, "subscribe")

        # Get market snapshot
        ret, snapshot = quote_ctx.get_market_snapshot([code])
        self._require_ok(sdk, ret, snapshot, "get_market_snapshot")

        row = snapshot.to_dict(orient="records")[0]

        # Extract relevant fields
        last_price = float(row.get("last_price") or 0)
        prev_close = float(row.get("prev_close_price") or 0)
        open_price = float(row.get("open") or 0)
        high = float(row.get("high_price") or 0)
        low = float(row.get("low_price") or 0)
        volume = int(row.get("volume") or 0)
        turnover = float(row.get("turnover") or 0)

        # Calculate change
        change_pct = 0.0
        if prev_close > 0:
            change_pct = ((last_price - prev_close) / prev_close) * 100

        # Keys mirror the CBOE underlying snapshot (parse_cboe_underlying) so
        # this is a drop-in for the committee briefing, which reads
        # day_high/day_low. iv30 is not available from a stock snapshot, so
        # the committee simply omits the IV line under this source.
        return {
            "price": last_price,
            "prev_close": prev_close,
            "open": open_price,
            "day_high": high,
            "day_low": low,
            "volume": volume,
            "turnover": turnover,
            "change_pct": round(change_pct, 2),
            "source": "moomoo_realtime",
        }

    def _quote_context(self, sdk: Any) -> Any:
        if self._quote_ctx is None:
            self._quote_ctx = sdk.OpenQuoteContext(
                host=self.connection.host, port=self.connection.port
            )
        return self._quote_ctx

    def close(self) -> None:
        if self._quote_ctx is None:
            return
        try:
            self._quote_ctx.close()
        finally:
            self._quote_ctx = None

    @staticmethod
    def _sdk() -> Any:
        import moomoo

        return moomoo

    @staticmethod
    def _require_ok(sdk: Any, ret: int, data: Any, operation: str) -> None:
        if ret != sdk.RET_OK:
            raise MoomooMarketError(f"{operation} failed: {data}")
