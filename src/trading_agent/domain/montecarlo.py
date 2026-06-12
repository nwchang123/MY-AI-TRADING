from __future__ import annotations

import math
import random
import zlib

# Deterministic, math-based probability-of-profit for one long-option trade.
#
# This is the BASELINE estimate: risk-neutral GBM with a sticky implied vol and
# no catalyst edge. The LLM committee's job is precisely to claim an edge above
# this baseline, so the mandate floor for this number is set LOW (filter out
# structurally hopeless tickets), while the higher 0.55 floor applies to the
# AI conviction estimates. Stdlib-only on purpose (no numpy dependency).

# Contract IVs above this are treated as unpriceable junk (CBOE shows >300%
# on some illiquid deep-ITM strikes); callers should skip the check.
MAX_USABLE_IV = 5.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    *, side: str, spot: float, strike: float, t_years: float, iv: float,
    rate: float = 0.04,
) -> float:
    """European Black-Scholes price; intrinsic value at/after expiry."""

    if side not in {"call", "put"}:
        raise ValueError("side must be 'call' or 'put'")
    if t_years <= 0 or iv <= 0 or spot <= 0:
        if side == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    sq = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2.0) * t_years) / sq
    d2 = d1 - sq
    if side == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * t_years) * _norm_cdf(d2)
    return strike * math.exp(-rate * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def black_scholes_delta(
    *, side: str, spot: float, strike: float, t_years: float, iv: float,
    rate: float = 0.04,
) -> float:
    """Black-Scholes delta; at/after expiry collapses to the intrinsic 0/±1.

    Shown to the committee as an honest measure of how much the option actually
    tracks the underlying: a 0.10-delta lottery ticket needs a far bigger move
    than a 0.50-delta near-the-money contract to reach the same +100%.
    """

    if side not in {"call", "put"}:
        raise ValueError("side must be 'call' or 'put'")
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        if side == "call":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2.0) * t_years) / (
        iv * math.sqrt(t_years)
    )
    return _norm_cdf(d1) if side == "call" else _norm_cdf(d1) - 1.0


def stable_seed(key: str) -> int:
    """Reproducible per-contract seed so audited numbers can be re-derived."""

    return zlib.crc32(key.encode("utf-8"))


def win_probability(
    *,
    side: str,
    spot: float,
    strike: float,
    dte_days: int,
    iv: float,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    hold_days: int | None = None,
    paths: int = 2000,
    rate: float = 0.04,
    seed: int | None = None,
) -> float:
    """P(option mark reaches take-profit before stop-loss or the horizon).

    Simulates daily GBM steps of the underlying and re-prices the option with
    Black-Scholes (same IV) along each path, mirroring how the position
    monitor actually exits: the FIRST barrier touched decides. Paths that
    reach the horizon (time stop / forced close before expiry) without
    touching take-profit are losses for this purpose, matching the committee's
    WIN_PROB definition. No edge or IV change is modeled -- this is the
    no-catalyst baseline.
    """

    if spot <= 0 or strike <= 0 or iv <= 0 or entry_price <= 0:
        raise ValueError("spot, strike, iv, and entry_price must be positive")
    if dte_days <= 0 or paths <= 0:
        raise ValueError("dte_days and paths must be positive")

    # Stop simulating where the monitor force-closes (~2 trading days before
    # expiry) or at the proposal's own time stop, whichever comes first.
    horizon = max(1, dte_days - 3)
    if hold_days is not None:
        horizon = max(1, min(horizon, hold_days))

    tp_level = entry_price * (1.0 + take_profit_pct / 100.0)
    sl_level = entry_price * (1.0 - stop_loss_pct / 100.0)
    rng = random.Random(seed)
    dt = 1.0 / 365.0
    drift = (rate - 0.5 * iv * iv) * dt
    vol = iv * math.sqrt(dt)

    wins = 0
    for _ in range(paths):
        s = spot
        for day in range(1, horizon + 1):
            s *= math.exp(drift + vol * rng.gauss(0.0, 1.0))
            t_remaining = max(dte_days - day, 0) / 365.0
            mark = black_scholes_price(
                side=side, spot=s, strike=strike, t_years=t_remaining, iv=iv,
                rate=rate,
            )
            if mark >= tp_level:
                wins += 1
                break
            if mark <= sl_level:
                break
    return wins / paths
