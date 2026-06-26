"""Yahoo Finance option data provider (unofficial, via yfinance).

Free, no API key, ~15-min delayed. Emits the SAME contract shape and the SAME
Moomoo option codes (``US.{ROOT}{YYMMDD}{C|P}{strike*1000}``, no zero-pad) as
the CBOE/Tradier providers, so it is a drop-in for the cycle's ``market``
dependency. The critical detail: yfinance returns OCC ``contractSymbol`` strings
(``AAPL260612C00125000`` -- no ``US.`` prefix, 8-digit padded strike), which
``parse_us_option_code`` rejects and the Moomoo broker will not place. Every row
is converted with ``occ_to_moomoo_code`` so the codes parse and trade unchanged.

Per-symbol results are cached for ``cache_ttl_seconds`` because one ticker is
touched several times per cycle (is_optionable -> probe -> eligible -> quote);
without the cache each touch re-fetched the chain, hammering Yahoo's unofficial
endpoint hard enough to get throttled in an unattended loop.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Callable

from trading_agent.data.moomoo_market import (
    US_OPTION_LOT_SIZE,
    occ_to_moomoo_code,
    parse_us_option_code,
)
from trading_agent.data.option_data import OptionDataError, underlying_symbol
from trading_agent.domain.risk import QuoteSnapshot


def _num(value: Any) -> float:
    """Coerce a yfinance cell to a finite float (NaN/None/garbage -> 0.0).

    yfinance fills missing bid/ask/OI/volume/IV with NaN; a NaN slipping into a
    QuoteSnapshot makes every downstream comparison nonsense, so it is zeroed.
    """

    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) else out


def yahoo_row_to_contract(row: Any, side: str, expiry: date) -> dict[str, Any] | None:
    """Normalize one yfinance chain row to the shared contract shape.

    Returns None when the OCC ``contractSymbol`` cannot be converted to a Moomoo
    code, so an odd symbol is skipped rather than poisoning the chain with a code
    nothing downstream can parse.
    """

    occ = str(getattr(row, "contractSymbol", "") or "")
    try:
        code = occ_to_moomoo_code(occ)
        underlying = parse_us_option_code(code)[0]
    except ValueError:
        return None
    return {
        "code": code,
        "occ": occ,
        "underlying": underlying,
        "side": side,
        "strike": _num(getattr(row, "strike", 0)),
        "expiry": expiry,
        "bid": _num(getattr(row, "bid", 0)),
        "ask": _num(getattr(row, "ask", 0)),
        "open_interest": int(_num(getattr(row, "openInterest", 0))),
        "daily_volume": int(_num(getattr(row, "volume", 0))),
        "iv": _num(getattr(row, "impliedVolatility", 0)),
    }


class YahooOptionData:
    """Option data from Yahoo Finance via the yfinance library."""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        cache_ttl_seconds: float = 30.0,
        now_fn: Callable[[], datetime] | None = None,
        ticker_factory: Callable[[str], Any] | None = None,
    ):
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        # Injectable so the parsing/format logic is unit-testable without hitting
        # Yahoo over the network.
        self._ticker_factory = ticker_factory or self._yfinance_ticker
        # symbol -> (fetched_at, expiration strings); (symbol, exp) -> (fetched_at, contracts)
        self._exp_cache: dict[str, tuple[datetime, list[str]]] = {}
        self._chain_cache: dict[tuple[str, str], tuple[datetime, list[dict[str, Any]]]] = {}

    @staticmethod
    def _yfinance_ticker(symbol: str) -> Any:
        try:
            from trading_agent.data.yf_compat import import_yfinance

            yf = import_yfinance()
        except ImportError as exc:
            raise OptionDataError(
                "yfinance not installed. Run: pip install yfinance"
            ) from exc
        return yf.Ticker(symbol)

    def _fresh(self, fetched_at: datetime) -> bool:
        return (self._now_fn() - fetched_at).total_seconds() < self.cache_ttl_seconds

    def _expiry_strings(self, symbol: str) -> list[str]:
        cached = self._exp_cache.get(symbol)
        if cached and self._fresh(cached[0]):
            return cached[1]
        options = list(self._ticker_factory(symbol).options or [])
        self._exp_cache[symbol] = (self._now_fn(), options)
        return options

    def _contracts_for_expiry(
        self, symbol: str, exp_str: str, expiry: date
    ) -> list[dict[str, Any]]:
        key = (symbol, exp_str)
        cached = self._chain_cache.get(key)
        if cached and self._fresh(cached[0]):
            return cached[1]
        chain = self._ticker_factory(symbol).option_chain(exp_str)
        rows: list[dict[str, Any]] = []
        for side, frame in (("call", chain.calls), ("put", chain.puts)):
            for _, row in frame.iterrows():
                contract = yahoo_row_to_contract(row, side, expiry)
                if contract is not None:
                    rows.append(contract)
        self._chain_cache[key] = (self._now_fn(), rows)
        return rows

    def is_optionable(self, underlying: str) -> bool:
        try:
            return bool(self._expiry_strings(underlying_symbol(underlying)))
        except Exception:  # noqa: BLE001 - optionability probe is best-effort
            return False

    def option_expirations(self, underlying: str) -> list[date]:
        try:
            out: list[date] = []
            for exp in self._expiry_strings(underlying_symbol(underlying)):
                try:
                    out.append(datetime.strptime(exp, "%Y-%m-%d").date())
                except ValueError:
                    continue
            return sorted(out)
        except OptionDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise OptionDataError(f"Yahoo request failed: {exc}") from exc

    def option_chain(
        self, underlying: str, start: date, end: date, option_type: str = "ALL"
    ) -> list[dict[str, Any]]:
        symbol = underlying_symbol(underlying)
        wanted = option_type.upper()
        try:
            rows: list[dict[str, Any]] = []
            for exp_str in self._expiry_strings(symbol):
                try:
                    expiry = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if not (start <= expiry <= end):
                    continue
                for contract in self._contracts_for_expiry(symbol, exp_str, expiry):
                    if wanted == "ALL" or contract["side"] == wanted.lower():
                        rows.append(contract)
            rows.sort(key=lambda c: (c["expiry"], c["side"], c["strike"]))
            return rows
        except OptionDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise OptionDataError(f"Yahoo request failed: {exc}") from exc

    def is_listed_option(self, option_code: str) -> bool:
        try:
            underlying, expiry, _, _ = parse_us_option_code(option_code)
        except ValueError:
            return False
        try:
            # Only the contract's own expiry is fetched, not the whole surface.
            chain = self.option_chain(underlying, expiry, expiry)
        except OptionDataError:
            return False
        return any(c["code"] == option_code for c in chain)

    def option_quote(
        self,
        *,
        option_code: str,
        expiry: date,
        lot_size: int = US_OPTION_LOT_SIZE,
        now: datetime | None = None,
    ) -> QuoteSnapshot:
        underlying, _, _, _ = parse_us_option_code(option_code)
        symbol = underlying_symbol(underlying)
        try:
            contracts = self._contracts_for_expiry(
                symbol, expiry.strftime("%Y-%m-%d"), expiry
            )
        except OptionDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise OptionDataError(f"Yahoo request failed: {exc}") from exc
        match = next((c for c in contracts if c["code"] == option_code), None)
        if match is None:
            raise OptionDataError(f"{option_code} not found in Yahoo chain")
        if match["ask"] <= 0:
            # No offer means no marketable exit/entry; QuoteSnapshot requires
            # ask > 0, so reject it here like the other providers do.
            raise OptionDataError(f"{option_code} has no ask (no liquidity)")
        return QuoteSnapshot(
            option_code=option_code,
            bid=match["bid"],
            ask=match["ask"],
            open_interest=match["open_interest"],
            daily_volume=match["daily_volume"],
            lot_size=lot_size,
            expiry=expiry,
            observed_at=now or self._now_fn(),
            is_delayed=True,
        )

    @staticmethod
    def _index_symbol(underlying: str) -> str:
        """Map a feed-neutral symbol to the yfinance form.

        The cycle's regime guard asks for the VIX as ``_VIX`` (CBOE's index
        convention, leading underscore); yfinance spells indices with a caret
        (``^VIX``). Plain tickers just lose any ``US.`` prefix.
        """

        symbol = underlying.strip().upper()
        if symbol.startswith("_"):
            return "^" + symbol[1:]
        return underlying_symbol(symbol)

    def underlying_snapshot(self, underlying: str) -> dict[str, Any]:
        """Delayed price context for an underlying or index (best-effort, {} on fail).

        Restores the VIX panic-regime guard under the Yahoo feed: the guard calls
        ``underlying_snapshot('_VIX')`` and reads ``price``. Returns the CBOE-shaped
        keys (price/prev_close/day_high/day_low/change_pct) the committee briefing
        also reads, so a Yahoo-sourced snapshot is a drop-in for the CBOE one.
        """

        symbol = self._index_symbol(underlying)
        try:
            info = getattr(self._ticker_factory(symbol), "fast_info", None)
            if not info:
                return {}
        except Exception:  # noqa: BLE001 - snapshot is best-effort, never blocks
            return {}

        def field(*names: str) -> float:
            for name in names:
                value = None
                try:
                    value = info[name]
                except (KeyError, TypeError, IndexError):
                    value = getattr(info, name, None)
                if value is not None:
                    return _num(value)
            return 0.0

        price = field("last_price", "lastPrice")
        if price <= 0:
            return {}
        prev_close = field("previous_close", "previousClose")
        change_pct = ((price - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
        return {
            "price": price,
            "prev_close": prev_close,
            "day_high": field("day_high", "dayHigh"),
            "day_low": field("day_low", "dayLow"),
            "change_pct": round(change_pct, 2),
        }


def build_yahoo_provider() -> YahooOptionData:
    """Factory for the Yahoo provider."""

    return YahooOptionData()
