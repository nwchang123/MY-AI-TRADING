from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from trading_agent.data.option_data import (
    CboeOptionData,
    FallbackOptionProvider,
    OptionDataError,
    TradierOptionData,
    build_option_provider,
    parse_cboe_payload,
)
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.risk import Mandate, QuoteSnapshot

NOW = datetime(2026, 6, 10, 16, 47, 0, tzinfo=timezone.utc)

# Mirrors the real CBOE delayed-quotes payload shape (verified live 2026-06-10).
CBOE_PAYLOAD = {
    "timestamp": "2026-06-10 16:46:23",  # naive UTC publish time
    "symbol": "AAPL",
    "data": {
        "symbol": "AAPL",
        "current_price": 290.0,
        "price_change_percent": 1.25,
        "prev_day_close": 286.4,
        "high": 291.5,
        "low": 285.9,
        "volume": 12345678,
        "iv30": 0.31,
        "iv30_change": -0.02,
        "options": [
            {
                "option": "AAPL260717C00310000",
                "bid": 2.97,
                "ask": 3.10,
                "open_interest": 42352,
                "volume": 4972,
                "iv": 0.2367,
            },
            {  # no ask -> not tradeable
                "option": "AAPL260717C00400000",
                "bid": 0.0,
                "ask": 0.0,
                "open_interest": 10,
                "volume": 0,
                "iv": 0.5,
            },
            {  # a put on an earlier expiry
                "option": "AAPL260619P00250000",
                "bid": 1.0,
                "ask": 1.1,
                "open_interest": 500,
                "volume": 50,
                "iv": 0.3,
            },
            {"option": "NOT_A_CODE", "bid": 1, "ask": 2},  # skipped
        ],
    },
}


def _cboe(counter: list[int] | None = None) -> CboeOptionData:
    def fetch(symbol: str) -> dict:
        if counter is not None:
            counter.append(symbol)
        if symbol != "AAPL":
            raise OptionDataError(f"CBOE has no option data for {symbol}")
        return CBOE_PAYLOAD

    return CboeOptionData(now_fn=lambda: NOW, fetch_fn=fetch)


def test_parse_cboe_payload_normalizes_and_skips_bad() -> None:
    observed_at, contracts = parse_cboe_payload(CBOE_PAYLOAD, fallback_now=NOW)
    assert observed_at == datetime(2026, 6, 10, 16, 46, 23, tzinfo=timezone.utc)
    codes = {c["code"] for c in contracts}
    # The unparseable 'NOT_A_CODE' row is dropped; the rest map to Moomoo codes.
    assert codes == {
        "US.AAPL260717C310000",
        "US.AAPL260717C400000",
        "US.AAPL260619P250000",
    }
    first = next(c for c in contracts if c["code"] == "US.AAPL260717C310000")
    assert first["bid"] == 2.97 and first["ask"] == 3.10
    assert first["open_interest"] == 42352 and first["daily_volume"] == 4972
    assert first["side"] == "call" and first["strike"] == 310.0


def test_parse_cboe_timestamp_falls_back_when_unparseable() -> None:
    observed_at, _ = parse_cboe_payload({"data": {"options": []}}, fallback_now=NOW)
    assert observed_at == NOW


def test_is_optionable_true_and_false() -> None:
    cboe = _cboe()
    assert cboe.is_optionable("US.AAPL") is True
    assert cboe.is_optionable("ZZZZ") is False


def test_option_expirations_sorted_unique() -> None:
    assert _cboe().option_expirations("AAPL") == [
        date(2026, 6, 19),
        date(2026, 7, 17),
    ]


def test_option_chain_filters_by_window_and_type() -> None:
    cboe = _cboe()
    calls = cboe.option_chain("AAPL", date(2026, 6, 15), date(2026, 7, 20), "CALL")
    assert {c["code"] for c in calls} == {
        "US.AAPL260717C310000",
        "US.AAPL260717C400000",
    }
    # The earlier put is outside a tighter window.
    july = cboe.option_chain("AAPL", date(2026, 7, 1), date(2026, 7, 20), "ALL")
    assert {c["code"] for c in july} == {
        "US.AAPL260717C310000",
        "US.AAPL260717C400000",
    }
    puts = cboe.option_chain("AAPL", date(2026, 6, 1), date(2026, 7, 20), "PUT")
    assert [c["code"] for c in puts] == ["US.AAPL260619P250000"]


def test_is_listed_option_real_and_fake() -> None:
    cboe = _cboe()
    assert cboe.is_listed_option("US.AAPL260717C310000") is True
    assert cboe.is_listed_option("US.AAPL260101C100000") is False
    assert cboe.is_listed_option("garbage") is False


def test_option_quote_builds_delayed_snapshot() -> None:
    quote = _cboe().option_quote(
        option_code="US.AAPL260717C310000", expiry=date(2026, 7, 17)
    )
    assert quote.bid == 2.97 and quote.ask == 3.10
    assert quote.open_interest == 42352 and quote.daily_volume == 4972
    assert quote.lot_size == 100
    assert quote.is_delayed is True
    assert quote.observed_at == datetime(2026, 6, 10, 16, 46, 23, tzinfo=timezone.utc)


def test_option_quote_rejects_zero_ask_and_missing() -> None:
    cboe = _cboe()
    with pytest.raises(OptionDataError):
        cboe.option_quote(option_code="US.AAPL260717C400000", expiry=date(2026, 7, 17))
    with pytest.raises(OptionDataError):
        cboe.option_quote(option_code="US.AAPL260101C100000", expiry=date(2026, 1, 1))


def test_underlying_snapshot_from_cached_payload() -> None:
    calls: list[str] = []
    cboe = _cboe(calls)
    cboe.option_expirations("AAPL")
    snapshot = cboe.underlying_snapshot("US.AAPL")
    assert snapshot["price"] == 290.0
    assert snapshot["change_pct"] == 1.25
    assert snapshot["iv30"] == 0.31
    assert snapshot["volume"] == 12345678
    assert calls == ["AAPL"]  # served from the same cached fetch


def test_fallback_underlying_snapshot_skips_failing_provider() -> None:
    class NoSnapshot:
        pass

    fallback = FallbackOptionProvider([NoSnapshot(), _cboe()])
    assert fallback.underlying_snapshot("AAPL")["price"] == 290.0
    assert FallbackOptionProvider([NoSnapshot()]).underlying_snapshot("AAPL") == {}


def test_cboe_caches_within_ttl() -> None:
    calls: list[str] = []
    cboe = _cboe(calls)
    cboe.is_optionable("AAPL")
    cboe.option_expirations("AAPL")
    cboe.option_quote(option_code="US.AAPL260717C310000", expiry=date(2026, 7, 17))
    # One underlying, three lookups, one HTTP fetch.
    assert calls == ["AAPL"]


# --- Tradier fallback -------------------------------------------------------

TRADIER_EXPIRATIONS = {"expirations": {"date": ["2026-06-19", "2026-07-17"]}}
TRADIER_CHAIN = {
    "options": {
        "option": [
            {
                "symbol": "AAPL260717C00310000",
                "bid": 2.9,
                "ask": 3.0,
                "open_interest": 42000,
                "volume": 4000,
                "greeks": {"mid_iv": 0.23},
            }
        ]
    }
}


def _tradier() -> TradierOptionData:
    def fetch(path: str, params: dict) -> dict:
        if path.endswith("expirations"):
            return TRADIER_EXPIRATIONS
        if path.endswith("chains"):
            return TRADIER_CHAIN if params.get("expiration") == "2026-07-17" else {
                "options": None
            }
        raise OptionDataError(path)

    return TradierOptionData("tok", now_fn=lambda: NOW, fetch_fn=fetch)


def test_tradier_expirations_and_quote() -> None:
    tradier = _tradier()
    assert tradier.is_optionable("AAPL") is True
    assert tradier.option_expirations("AAPL") == [date(2026, 6, 19), date(2026, 7, 17)]
    quote = tradier.option_quote(
        option_code="US.AAPL260717C310000", expiry=date(2026, 7, 17)
    )
    assert quote.ask == 3.0 and quote.is_delayed is True
    assert quote.observed_at == NOW


def test_tradier_requires_token() -> None:
    with pytest.raises(ValueError):
        TradierOptionData("")


def test_fallback_falls_through_on_error() -> None:
    class Failing:
        def option_quote(self, **kwargs):
            raise OptionDataError("down")

        def is_optionable(self, underlying):
            return False

    fallback = FallbackOptionProvider([Failing(), _cboe()])
    quote = fallback.option_quote(
        option_code="US.AAPL260717C310000", expiry=date(2026, 7, 17)
    )
    assert quote.ask == 3.10
    assert fallback.is_optionable("AAPL") is True


def test_build_option_provider_variants() -> None:
    assert isinstance(build_option_provider("cboe"), CboeOptionData)
    combined = build_option_provider("cboe+tradier", tradier_token="tok")
    assert isinstance(combined, FallbackOptionProvider)
    # cboe+tradier with no token degrades to CBOE alone.
    assert isinstance(build_option_provider("cboe+tradier"), CboeOptionData)
    with pytest.raises(ValueError):
        build_option_provider("tradier")  # no token


# --- delayed staleness through the real validator ---------------------------


def _quote(observed_at: datetime, *, is_delayed: bool) -> QuoteSnapshot:
    return QuoteSnapshot(
        option_code="US.AAPL260717C310000",
        bid=2.97,
        ask=3.10,
        open_interest=42352,
        daily_volume=4972,
        lot_size=100,
        expiry=date(2026, 7, 17),
        observed_at=observed_at,
        is_delayed=is_delayed,
    )


def test_delayed_quote_passes_staleness_realtime_does_not() -> None:
    mandate = Mandate.load(Path("config/mandate.paper.yaml"))
    validator = LiquidityValidator(mandate.options, mandate.execution)
    now = datetime(2026, 6, 25, 16, 0, tzinfo=timezone.utc)
    ten_min_ago = datetime(2026, 6, 25, 15, 50, tzinfo=timezone.utc)

    delayed = validator.validate(_quote(ten_min_ago, is_delayed=True), now=now)
    assert "quote is stale" not in delayed.reasons  # 600s < 1200s delayed limit

    realtime = validator.validate(_quote(ten_min_ago, is_delayed=False), now=now)
    assert "quote is stale" in realtime.reasons  # 600s > 15s real-time limit
