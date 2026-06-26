"""Underlying daily-bar context via yfinance (free, best-effort).

Closes two AI-layer blind spots the committee had:

1. It saw only a single day's snapshot, so it could chase a name already up
   300% in three days. Recent returns + position in the 20-day range give it
   the "how much is already priced in" context.
2. It saw a contract's IV but had no baseline for whether that IV is cheap or
   rich. The 20-day realized volatility is that baseline: a contract IV far
   above realized vol means you are buying movement the stock has not been
   delivering (an expensive lottery ticket bought near the vol top).

Strictly best-effort: a yfinance failure yields an empty dict, never an error.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Callable


def compute_price_features(closes: list[float]) -> dict[str, Any]:
    """Daily-bar features from a list of closes (oldest -> newest).

    Pure and stdlib-only so it is unit-testable from a fixture. Returns ``{}``
    when there is too little data to say anything; individual fields are None
    when their specific lookback is not covered.
    """

    series = [float(c) for c in closes if c is not None and float(c) > 0]
    if len(series) < 2:
        return {}
    last = series[-1]

    def ret(days: int) -> float | None:
        if len(series) > days:
            base = series[-(days + 1)]
            if base > 0:
                return round((last / base - 1.0) * 100.0, 2)
        return None

    window = series[-21:]  # up to 20 daily log returns
    log_returns = [
        math.log(window[i] / window[i - 1])
        for i in range(1, len(window))
        if window[i - 1] > 0
    ]
    realized_vol = None
    if len(log_returns) >= 2:
        realized_vol = round(statistics.stdev(log_returns) * math.sqrt(252) * 100, 1)

    recent = series[-20:]
    high_20d, low_20d = max(recent), min(recent)
    return {
        "last_close": round(last, 4),
        "ret_5d": ret(5),
        "ret_20d": ret(20),
        "realized_vol_20d": realized_vol,
        "pct_from_20d_high": round((last / high_20d - 1.0) * 100.0, 2),
        "pct_from_20d_low": round((last / low_20d - 1.0) * 100.0, 2),
    }


class YahooPriceHistory:
    """Underlying technical context from yfinance daily bars."""

    def __init__(
        self,
        *,
        lookback: str = "3mo",
        ticker_factory: Callable[[str], Any] | None = None,
    ):
        self.lookback = lookback
        self._ticker_factory = ticker_factory or self._yfinance_ticker

    @staticmethod
    def _yfinance_ticker(symbol: str) -> Any:
        from trading_agent.data.yf_compat import import_yfinance

        return import_yfinance().Ticker(symbol)

    def context(self, ticker: str) -> dict[str, Any]:
        from trading_agent.data.option_data import underlying_symbol

        symbol = underlying_symbol(ticker)
        try:
            history = self._ticker_factory(symbol).history(period=self.lookback)
            closes = list(history["Close"])  # pandas Series or plain list
        except Exception:  # noqa: BLE001 - best-effort context, never blocks
            return {}
        return compute_price_features(closes)
