from __future__ import annotations

from typing import Any

from trading_agent.domain.risk import UniverseMandate

# Each optionability probe downloads a full option chain from the data feed, so
# only the most liquid scan hits are probed. Liquidity (turnover) ranking comes
# first because an optionable-but-dead name is useless to the strategy anyway.
DEFAULT_MAX_TICKERS = 10
DEFAULT_PROBE_LIMIT = 30


def select_universe(
    *,
    market: Any,
    provider: Any,
    universe: UniverseMandate,
    max_tickers: int = DEFAULT_MAX_TICKERS,
    probe_limit: int = DEFAULT_PROBE_LIMIT,
) -> list[str]:
    """Autonomously pick today's candidate tickers.

    Screens U.S. small caps through the Moomoo stock filter (price, market cap,
    turnover from the mandate), ranks by turnover descending, then keeps the
    first ``max_tickers`` names whose option chain actually exists on the option
    data feed. Returns bare ticker symbols (no ``US.`` prefix), ready for the
    trading cycle.
    """

    if max_tickers <= 0:
        return []

    rows = market.scan_small_caps(universe)
    ranked = sorted(rows, key=lambda r: float(r.get("turnover") or 0.0), reverse=True)

    selected: list[str] = []
    seen: set[str] = set()
    probed = 0
    for row in ranked:
        code = str(row.get("code") or "")
        ticker = code.removeprefix("US.").strip().upper()
        # Option roots are plain alphabetic tickers; skip warrants/units/etc.
        if not ticker or not ticker.isalpha() or ticker in seen:
            continue
        seen.add(ticker)
        if probed >= probe_limit:
            break
        probed += 1
        if provider.is_optionable(ticker):
            selected.append(ticker)
            if len(selected) >= max_tickers:
                break
    return selected
