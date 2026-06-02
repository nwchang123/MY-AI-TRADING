from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from trading_agent.brokers.moomoo import MoomooConnection
from trading_agent.domain.risk import QuoteSnapshot, UniverseMandate

# US equity options trade in lots of 100 shares.
US_OPTION_LOT_SIZE = 100

_OPTION_CODE_RE = re.compile(r"^US\.([A-Z]+)(\d{6})([CP])(\d{8})$")


class MoomooMarketError(RuntimeError):
    """Raised when OpenD or the quote SDK rejects a market-data request."""


def parse_us_option_code(option_code: str) -> tuple[str, date, str, float]:
    """Parse a Moomoo US option code, e.g. 'US.NVDA260626C00005000'.

    Returns (underlying, expiry, side, strike). Lets the agent derive expiry
    without an extra chain call. Raises ValueError on an unrecognized code.
    """

    match = _OPTION_CODE_RE.match(option_code)
    if match is None:
        raise ValueError(f"unrecognized US option code: {option_code}")
    root, ymd, call_put, strike = match.groups()
    expiry = date(2000 + int(ymd[0:2]), int(ymd[2:4]), int(ymd[4:6]))
    side = "call" if call_put == "C" else "put"
    return root, expiry, side, int(strike) / 1000


def build_universe_filters(universe: UniverseMandate, sdk: Any) -> list[Any]:
    """Translate the universe mandate into Moomoo SimpleFilter objects.

    Pure with respect to the SDK module, so it can be tested with a fake ``sdk``.
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

    return [
        make(field.CUR_PRICE, fmin=universe.min_underlying_price_usd),
        make(
            field.MARKET_VAL,
            fmin=universe.min_market_cap_usd,
            fmax=universe.max_market_cap_usd,
        ),
        make(field.TURNOVER, fmin=universe.min_average_daily_turnover_usd),
    ]


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
                "turnover": getattr(row, "turnover", None),
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

    Live methods open an ``OpenQuoteContext`` against OpenD and always close it.
    Deterministic parsing lives in the module-level helpers above so it can be
    unit tested without OpenD.
    """

    def __init__(self, connection: MoomooConnection):
        self.connection = connection

    def scan_small_caps(
        self, universe: UniverseMandate, begin: int = 0, num: int = 200
    ) -> list[dict[str, Any]]:
        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        try:
            filters = build_universe_filters(universe, sdk)
            ret, data = quote_ctx.get_stock_filter(
                market=sdk.Market.US, filter_list=filters, begin=begin, num=num
            )
            self._require_ok(sdk, ret, data, "get_stock_filter")
            _last_page, _all_count, rows = data
            return parse_filter_rows(rows)
        finally:
            quote_ctx.close()

    def option_expirations(self, code: str) -> list[dict[str, Any]]:
        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        try:
            ret, data = quote_ctx.get_option_expiration_date(code=code)
            self._require_ok(sdk, ret, data, "get_option_expiration_date")
            return data.to_dict(orient="records")
        finally:
            quote_ctx.close()

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
        try:
            ret, data = quote_ctx.get_option_chain(
                code=code, start=start, end=end, option_type=option_type
            )
            self._require_ok(sdk, ret, data, "get_option_chain")
            return data.to_dict(orient="records")
        finally:
            quote_ctx.close()

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
        try:
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
        finally:
            quote_ctx.close()

    def _quote_context(self, sdk: Any) -> Any:
        return sdk.OpenQuoteContext(
            host=self.connection.host, port=self.connection.port
        )

    @staticmethod
    def _sdk() -> Any:
        import moomoo

        return moomoo

    @staticmethod
    def _require_ok(sdk: Any, ret: int, data: Any, operation: str) -> None:
        if ret != sdk.RET_OK:
            raise MoomooMarketError(f"{operation} failed: {data}")
