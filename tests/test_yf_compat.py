import warnings

from trading_agent.data.yf_compat import import_yfinance


def test_import_yfinance_returns_usable_module() -> None:
    yf = import_yfinance()
    assert hasattr(yf, "Ticker")


def test_rehoists_ignore_above_a_later_prepended_filter() -> None:
    # yfinance prepends its own filters above our package-level ignore on import,
    # burying it. import_yfinance must re-hoist the ignore back to the front.
    with warnings.catch_warnings():
        warnings.resetwarnings()
        # Simulate yfinance's prepended catch-all that would otherwise SHOW it.
        warnings.filterwarnings("default", category=DeprecationWarning)
        import_yfinance()
        top = warnings.filters[0]
        assert top[0] == "ignore"
        assert top[1] is not None and top[1].pattern.startswith("Timestamp")


def test_utcnow_warning_is_actually_suppressed() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.filterwarnings("default", category=DeprecationWarning)  # competing
        import_yfinance()  # re-hoists the ignore above the competing filter
        warnings.warn(
            "Timestamp.utcnow is deprecated and will be removed in a future version.",
            DeprecationWarning,
        )
        assert not any("utcnow" in str(w.message) for w in caught)
