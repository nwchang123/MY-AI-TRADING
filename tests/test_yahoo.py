from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from trading_agent.data.option_data import OptionDataError
from trading_agent.data.yahoo import YahooOptionData, yahoo_row_to_contract

NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)


class FakeFrame:
    """Minimal stand-in for a pandas DataFrame's iterrows()."""

    def __init__(self, rows: list[SimpleNamespace]):
        self._rows = rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, row


class FakeChain:
    def __init__(self, calls, puts):
        self.calls = FakeFrame(calls)
        self.puts = FakeFrame(puts)


class FakeTicker:
    def __init__(self, options: list[str], chains: dict[str, FakeChain]):
        self.options = options
        self._chains = chains

    def option_chain(self, exp: str) -> FakeChain:
        return self._chains[exp]


def _row(symbol: str, strike: float, **over) -> SimpleNamespace:
    base = dict(
        contractSymbol=symbol,
        strike=strike,
        bid=0.19,
        ask=0.21,
        openInterest=200,
        volume=30,
        impliedVolatility=0.5,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _provider(ticker: FakeTicker) -> YahooOptionData:
    return YahooOptionData(now_fn=lambda: NOW, ticker_factory=lambda symbol: ticker)


# --- the core fix: OCC contractSymbol -> Moomoo code -----------------------


def test_row_converts_occ_symbol_to_moomoo_code() -> None:
    # yfinance gives the 8-digit padded OCC form; the contract must carry the
    # no-pad Moomoo code the broker and parse_us_option_code accept.
    contract = yahoo_row_to_contract(
        _row("AAPL260612C00125000", 125.0), "call", date(2026, 6, 12)
    )
    assert contract is not None
    assert contract["code"] == "US.AAPL260612C125000"
    assert contract["occ"] == "AAPL260612C00125000"
    assert contract["underlying"] == "AAPL"


def test_row_with_unparseable_symbol_is_skipped() -> None:
    assert yahoo_row_to_contract(_row("not-an-occ-symbol", 5.0), "call", date(2026, 6, 12)) is None


def test_row_coerces_nan_quote_fields_to_zero() -> None:
    nan = float("nan")
    contract = yahoo_row_to_contract(
        _row("AAPL260612P00005000", 5.0, bid=nan, ask=nan, openInterest=nan, volume=nan),
        "put",
        date(2026, 6, 12),
    )
    assert contract["bid"] == 0.0
    assert contract["ask"] == 0.0
    assert contract["open_interest"] == 0
    assert contract["daily_volume"] == 0


# --- chain / quote behaviour ----------------------------------------------


def test_option_chain_filters_window_and_emits_moomoo_codes() -> None:
    chains = {
        "2026-06-26": FakeChain(
            calls=[_row("AAPL260626C00125000", 125.0)],
            puts=[_row("AAPL260626P00120000", 120.0)],
        ),
        "2026-09-18": FakeChain(  # outside the DTE window -> dropped
            calls=[_row("AAPL260918C00125000", 125.0)], puts=[]
        ),
    }
    ticker = FakeTicker(["2026-06-26", "2026-09-18"], chains)
    rows = _provider(ticker).option_chain("US.AAPL", date(2026, 6, 16), date(2026, 7, 16))

    assert {r["code"] for r in rows} == {
        "US.AAPL260626C125000",
        "US.AAPL260626P120000",
    }
    assert all(r["code"].startswith("US.") for r in rows)


def test_option_quote_matches_by_moomoo_code() -> None:
    chains = {
        "2026-06-26": FakeChain(
            calls=[_row("AAPL260626C00125000", 125.0, bid=0.40, ask=0.45)],
            puts=[],
        )
    }
    ticker = FakeTicker(["2026-06-26"], chains)
    quote = _provider(ticker).option_quote(
        option_code="US.AAPL260626C125000", expiry=date(2026, 6, 26), now=NOW
    )
    assert quote.bid == 0.40
    assert quote.ask == 0.45
    assert quote.open_interest == 200
    assert quote.is_delayed is True


def test_option_quote_rejects_zero_ask() -> None:
    chains = {
        "2026-06-26": FakeChain(
            calls=[_row("AAPL260626C00125000", 125.0, ask=0.0)], puts=[]
        )
    }
    ticker = FakeTicker(["2026-06-26"], chains)
    with pytest.raises(OptionDataError, match="no ask"):
        _provider(ticker).option_quote(
            option_code="US.AAPL260626C125000", expiry=date(2026, 6, 26)
        )


def test_option_quote_missing_contract_raises() -> None:
    ticker = FakeTicker(["2026-06-26"], {"2026-06-26": FakeChain([], [])})
    with pytest.raises(OptionDataError, match="not found"):
        _provider(ticker).option_quote(
            option_code="US.AAPL260626C125000", expiry=date(2026, 6, 26)
        )


def test_is_optionable_reflects_expirations() -> None:
    assert _provider(FakeTicker(["2026-06-26"], {})).is_optionable("US.AAPL") is True
    assert _provider(FakeTicker([], {})).is_optionable("US.AAPL") is False


def test_is_listed_option_checks_only_the_contract_expiry() -> None:
    chains = {"2026-06-26": FakeChain(calls=[_row("AAPL260626C00125000", 125.0)], puts=[])}
    provider = _provider(FakeTicker(["2026-06-26"], chains))
    assert provider.is_listed_option("US.AAPL260626C125000") is True
    assert provider.is_listed_option("US.AAPL260626C999000") is False
    assert provider.is_listed_option("garbage") is False


def test_provider_normalizes_yfinance_errors() -> None:
    class Boom:
        @property
        def options(self):
            raise RuntimeError("network down")

    provider = YahooOptionData(ticker_factory=lambda symbol: Boom())
    # is_optionable swallows it; option_chain surfaces a normalized error.
    assert provider.is_optionable("US.AAPL") is False
    with pytest.raises(OptionDataError):
        provider.option_chain("US.AAPL", date(2026, 6, 1), date(2026, 7, 1))
