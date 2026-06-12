from types import SimpleNamespace

from trading_agent.data.price_history import YahooPriceHistory, compute_price_features


def test_compute_features_full_series() -> None:
    closes = [float(x) for x in range(100, 125)]  # 25 bars, 100..124, last 124
    f = compute_price_features(closes)
    assert f["last_close"] == 124.0
    assert f["ret_5d"] == round((124 / 119 - 1) * 100, 2)   # 4.2
    assert f["ret_20d"] == round((124 / 104 - 1) * 100, 2)  # 19.23
    assert f["pct_from_20d_high"] == 0.0                    # at the high
    assert f["pct_from_20d_low"] == round((124 / 105 - 1) * 100, 2)  # 18.1
    assert f["realized_vol_20d"] is not None and f["realized_vol_20d"] > 0


def test_compute_features_short_series_nulls_long_lookbacks() -> None:
    f = compute_price_features([10.0, 10.5, 11.0])
    assert f["last_close"] == 11.0
    assert f["ret_5d"] is None and f["ret_20d"] is None  # not enough history
    assert f["pct_from_20d_high"] is not None  # range still computable


def test_compute_features_too_little_data_is_empty() -> None:
    assert compute_price_features([]) == {}
    assert compute_price_features([10.0]) == {}
    assert compute_price_features([0.0, None, -5]) == {}  # garbage filtered out


def test_negative_return_is_signed() -> None:
    closes = [100.0] * 19 + [120.0, 90.0]  # last bar drops hard
    f = compute_price_features(closes)
    assert f["ret_5d"] < 0
    assert f["pct_from_20d_high"] < 0  # below the recent high


class _FakeTicker:
    def __init__(self, closes):
        self._closes = closes

    def history(self, period):
        return {"Close": self._closes}


def test_context_strips_us_prefix_and_computes() -> None:
    seen: list[str] = []

    def factory(symbol: str):
        seen.append(symbol)
        return _FakeTicker([float(x) for x in range(100, 125)])

    ctx = YahooPriceHistory(ticker_factory=factory).context("US.AAPL")
    assert seen == ["AAPL"]
    assert ctx["last_close"] == 124.0


def test_context_empty_on_fetch_failure() -> None:
    def boom(symbol: str):
        raise RuntimeError("yahoo down")

    assert YahooPriceHistory(ticker_factory=boom).context("AAPL") == {}
