"""Yahoo Finance option data provider (unofficial, via yfinance).

Free, no API key, ~15-min delayed. Emits the SAME contract shape and the SAME
Moomoo option codes (``US.{ROOT}{YYMMDD}{C|P}{strike*1000}``, no zero-pad) as
the CBOE/Tradier providers, so it is a drop-in for the cycle's ``market``
dependency. The critical detail: yfinance returns OCC ``contractSymbol`` strings
(``AAPL260612C00125000`` -- no ``US.`` prefix, 8-digit padded strike), which
``parse_us_option_code`` rejects and the Moomoo broker will not place. Every row
is converted with ``occ_to_moomoo_code`` so the codes parse and trade unchanged.
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
        now_fn: Callable[[], datetime] | None = None,
        ticker_factory: Callable[[str], Any] | None = None,
    ):
        self.timeout = timeout
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        # Injectable so the parsing/format logic is unit-testable without hitting
        # Yahoo over the network.
        self._ticker_factory = ticker_factory or self._yfinance_ticker

    @staticmethod
    def _yfinance_ticker(symbol: str) -> Any:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise OptionDataError(
                "yfinance not installed. Run: pip install yfinance"
            ) from exc
        return yf.Ticker(symbol)

    def _get_ticker(self, underlying: str) -> Any:
        return self._ticker_factory(underlying_symbol(underlying))

    def is_optionable(self, underlying: str) -> bool:
        try:
            return bool(self._get_ticker(underlying).options)
        except Exception:  # noqa: BLE001 - optionability probe is best-effort
            return False

    def option_expirations(self, underlying: str) -> list[date]:
        try:
            ticker = self._get_ticker(underlying)
            out: list[date] = []
            for exp in ticker.options or []:
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
        wanted = option_type.upper()
        try:
            ticker = self._get_ticker(underlying)
            rows: list[dict[str, Any]] = []
            for exp_str in ticker.options or []:
                try:
                    expiry = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if not (start <= expiry <= end):
                    continue
                chain = ticker.option_chain(exp_str)
                if wanted in ("ALL", "CALL"):
                    for _, row in chain.calls.iterrows():
                        contract = yahoo_row_to_contract(row, "call", expiry)
                        if contract is not None:
                            rows.append(contract)
                if wanted in ("ALL", "PUT"):
                    for _, row in chain.puts.iterrows():
                        contract = yahoo_row_to_contract(row, "put", expiry)
                        if contract is not None:
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
        chain = self.option_chain(underlying, expiry, expiry)
        match = next((c for c in chain if c["code"] == option_code), None)
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


def build_yahoo_provider() -> YahooOptionData:
    """Factory for the Yahoo provider."""

    return YahooOptionData()
