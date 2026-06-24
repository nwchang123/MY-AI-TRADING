from __future__ import annotations

import math
import random
import zlib
from datetime import date, timedelta

from trading_agent.domain.calendar import is_trading_day

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


def black_scholes_gamma(
    *, spot: float, strike: float, t_years: float, iv: float,
    rate: float = 0.04,
) -> float:
    """Black-Scholes gamma; call and put share the same gamma.

    Gamma is highest for ATM options and increases sharply near expiry.
    A large gamma means delta is very sensitive to small price moves --
    useful for spotting elevated risk on short-dated, near-the-money contracts.
    """

    if spot <= 0 or strike <= 0 or iv <= 0:
        return 0.0
    if t_years <= 0:
        return 0.0
    sq = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2.0) * t_years) / sq
    return math.exp(-d1 * d1 / 2.0) / (spot * sq * math.sqrt(2.0 * math.pi))


def black_scholes_vega(
    *, spot: float, strike: float, t_years: float, iv: float,
    rate: float = 0.04,
) -> float:
    """Black-Scholes vega; call and put share the same vega.

    Vega measures sensitivity to a 1-unit (100%) change in IV.  In practice
    IV moves in percentage points, so multiply by 0.01 to get the dollar
    change per 1% IV move.  High vega = big exposure to IV changes.
    """

    if spot <= 0 or strike <= 0 or iv <= 0 or t_years <= 0:
        return 0.0
    sq = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2.0) * t_years) / sq
    return spot * math.exp(-d1 * d1 / 2.0) * math.sqrt(t_years) / math.sqrt(2.0 * math.pi)


def black_scholes_theta(
    *, side: str, spot: float, strike: float, t_years: float, iv: float,
    rate: float = 0.04,
) -> float:
    """Black-Scholes theta (time decay per year).

    Negative for long options (they lose value over time).  Divide by 365
    for daily decay.  Theta is largest for ATM options near expiry.
    """

    if side not in {"call", "put"}:
        raise ValueError("side must be 'call' or 'put'")
    if spot <= 0 or strike <= 0 or iv <= 0 or t_years <= 0:
        return 0.0
    sq = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2.0) * t_years) / sq
    d2 = d1 - sq
    common = -(spot * math.exp(-d1 * d1 / 2.0) * iv) / (2.0 * math.sqrt(2.0 * math.pi * t_years))
    if side == "call":
        return common - rate * strike * math.exp(-rate * t_years) * _norm_cdf(d2)
    return common + rate * strike * math.exp(-rate * t_years) * _norm_cdf(-d2)


def iv_rank(
    *,
    current_iv: float,
    iv_history: list[float] | None = None,
    iv_52w_high: float | None = None,
    iv_52w_low: float | None = None,
) -> float | None:
    """IV Rank: where current IV sits in its 52-week range (0.0 – 1.0).

    When a full history or 52-week extremes are provided the rank is exact;
    otherwise returns None so callers can skip gracefully.  Useful for the
    committee: IV Rank > 0.70 signals expensive options (don't buy premium),
    < 0.30 signals cheap options (favorable for long premium).
    """

    if current_iv <= 0:
        return None

    if iv_history and len(iv_history) >= 2:
        hi = max(iv_history)
        lo = min(iv_history)
        if hi <= lo:
            return None
        return (current_iv - lo) / (hi - lo)

    if iv_52w_high is not None and iv_52w_low is not None:
        if iv_52w_high <= iv_52w_low:
            return None
        return (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)

    return None


def kelly_criterion(
    *,
    win_prob: float,
    win_amount: float,
    lose_amount: float,
) -> float | None:
    """Kelly fraction: optimal fraction of bankroll to risk per trade.

    f* = (p * b - q) / b  where p = win_prob, q = 1-p, b = win_amount / lose_amount.
    Returns a value in [0, 1] (negative → no bet).  In practice, fractional
    Kelly (0.25–0.50) is used to smooth volatility.  Returns None on invalid
    inputs.
    """

    if win_prob <= 0 or win_prob >= 1:
        return None
    if win_amount <= 0 or lose_amount <= 0:
        return None

    p = win_prob
    q = 1.0 - p
    b = win_amount / lose_amount
    fraction = (p * b - q) / b

    return max(0.0, min(fraction, 1.0))


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
    as_of: date | None = None,
) -> float:
    """P(option mark reaches take-profit before stop-loss or the horizon).

    Simulates GBM steps of the underlying and re-prices the option with
    Black-Scholes (same IV) along each path, mirroring how the position
    monitor actually exits: the FIRST barrier touched decides. Paths that
    reach the horizon (time stop / forced close before expiry) without
    touching take-profit are losses for this purpose, matching the committee's
    WIN_PROB definition. No edge or IV change is modeled -- this is the
    no-catalyst baseline.

    ``as_of`` (recommended) steps the simulation over TRADING days starting
    from that calendar date, skipping weekends/holidays -- a step then spans
    the actual calendar days elapsed (a Friday→Monday step uses 3/365y),
    which is how the position monitor actually samples marks. Without it the
    legacy calendar-day cadence is used so existing seeded tests stay stable.
    """

    if spot <= 0 or strike <= 0 or iv <= 0 or entry_price <= 0:
        raise ValueError("spot, strike, iv, and entry_price must be positive")
    if dte_days <= 0 or paths <= 0:
        raise ValueError("dte_days and paths must be positive")

    # Stop simulating where the monitor force-closes (~2 trading days before
    # expiry) or at the proposal's own time stop, whichever comes first.
    # ``dte_days`` is in CALENDAR days (broker convention), so when stepping
    # over trading days we still cap the step count by the same horizon but
    # convert via the trading-day calendar so weekends don't burn a step.
    horizon_calendar_days = max(1, dte_days - 3)
    if hold_days is not None:
        horizon_calendar_days = max(1, min(horizon_calendar_days, hold_days))

    tp_level = entry_price * (1.0 + take_profit_pct / 100.0)
    sl_level = entry_price * (1.0 - stop_loss_pct / 100.0)
    rng = random.Random(seed)

    # Build the list of (step_calendar_days_elapsed) for each simulated step.
    # Each step's dt is the calendar days since the previous step / 365y, so
    # option pricing (always actual/365) stays correct while the cadence
    # matches real market sessions.
    if as_of is not None:
        steps = _trading_day_steps(as_of, horizon_calendar_days, dte_days)
    else:
        steps = [(1, day) for day in range(1, horizon_calendar_days + 1)]

    wins = 0
    for _ in range(paths):
        s = spot
        for step_dt_days, day_elapsed in steps:
            dt = step_dt_days / 365.0
            drift = (rate - 0.5 * iv * iv) * dt
            vol = iv * math.sqrt(dt)
            s *= math.exp(drift + vol * rng.gauss(0.0, 1.0))
            t_remaining = max(dte_days - day_elapsed, 0) / 365.0
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


def _trading_day_steps(
    start: date, horizon_calendar_days: int, dte_days: int
) -> list[tuple[int, int]]:
    """Return ``(calendar_days_in_step, total_calendar_days_elapsed)`` per step.

    Walks forward from ``start`` over trading days only, stopping once the
    cumulative calendar span reaches ``horizon_calendar_days`` or we run out
    of DTE. Each entry's first element is the gap (in calendar days) from the
    previous session -- typically 1, but 3 over a weekend and more over a
    holiday weekend -- so drift/vol scale to the real elapsed time.
    """

    steps: list[tuple[int, int]] = []
    prev = start
    cur = start + timedelta(days=1)
    guard = 0
    while guard < 4 * (horizon_calendar_days + 14):
        guard += 1
        if is_trading_day(cur):
            gap = (cur - prev).days
            elapsed = (cur - start).days
            if elapsed > horizon_calendar_days or elapsed >= dte_days:
                break
            steps.append((gap, elapsed))
            prev = cur
        cur += timedelta(days=1)
    if not steps:
        # Degenerate horizon (e.g. expiry today): one minimal step so the
        # inner loop runs at least once and pricing still works.
        steps.append((1, 1))
    return steps


# ---------------------------------------------------------------------------
# #4  Value at Risk (VaR)
# ---------------------------------------------------------------------------


def portfolio_var(
    *,
    positions: list[dict],
    confidence: float = 0.95,
    horizon_days: int = 1,
    paths: int = 5000,
    rate: float = 0.04,
    seed: int | None = None,
) -> float:
    """Historical-simulation VaR for a portfolio of long option positions.

    Each position dict: {side, spot, strike, dte_days, iv, contracts,
    lot_size, entry_price}.  Returns the dollar loss at the given confidence
    level over the horizon.  Positive = loss amount.
    """

    if not positions:
        return 0.0
    rng = random.Random(seed)
    dt = horizon_days / 365.0
    pnl_paths: list[float] = []
    for _ in range(paths):
        total_pnl = 0.0
        for pos in positions:
            s = pos["spot"]
            drift = (rate - 0.5 * pos["iv"] ** 2) * dt
            vol = pos["iv"] * math.sqrt(dt)
            s_end = s * math.exp(drift + vol * rng.gauss(0.0, 1.0))
            t_end = max(pos["dte_days"] - horizon_days, 0) / 365.0
            old_mark = black_scholes_price(
                side=pos["side"], spot=s, strike=pos["strike"],
                t_years=pos["dte_days"] / 365.0, iv=pos["iv"], rate=rate,
            )
            new_mark = black_scholes_price(
                side=pos["side"], spot=s_end, strike=pos["strike"],
                t_years=t_end, iv=pos["iv"], rate=rate,
            )
            total_pnl += (new_mark - old_mark) * pos["contracts"] * pos["lot_size"]
        pnl_paths.append(total_pnl)
    pnl_paths.sort()
    idx = int((1 - confidence) * len(pnl_paths))
    return -pnl_paths[max(idx, 0)]


# ---------------------------------------------------------------------------
# #5  IV Surface helpers
# ---------------------------------------------------------------------------


def iv_skew(
    *,
    calls_by_strike: list[dict],
    spot: float,
) -> float | None:
    """Approximate 25-delta risk-reversal skew: IV(25d put) - IV(25d call).

    ``calls_by_strike``: list of {strike, iv, delta} for ATM附近的options.
    Returns the skew (positive = put vol > call vol = bearish skew), or None
    if insufficient data.
    """

    puts = sorted(
        [r for r in calls_by_strike if r.get("delta") is not None and r["delta"] < -0.20],
        key=lambda r: abs(r["delta"] + 0.25),
    )
    calls = sorted(
        [r for r in calls_by_strike if r.get("delta") is not None and r["delta"] > 0.20],
        key=lambda r: abs(r["delta"] - 0.25),
    )
    if not puts or not calls:
        return None
    return puts[0]["iv"] - calls[0]["iv"]


def iv_term_structure(
    *,
    iv_by_expiry: list[dict],
) -> float | None:
    """Term-structure slope: IV(90d) - IV(30d).

    ``iv_by_expiry``: list of {dte_days, iv}.  Returns the slope (positive =
    contango = longer-dated more expensive), or None if insufficient data.
    """

    short = [r for r in iv_by_expiry if 20 <= r["dte_days"] <= 40]
    long = [r for r in iv_by_expiry if 80 <= r["dte_days"] <= 100]
    if not short or not long:
        return None
    return long[0]["iv"] - short[0]["iv"]


# ---------------------------------------------------------------------------
# #6  P&L Attribution
# ---------------------------------------------------------------------------


def pnl_attribution(
    *,
    side: str,
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    spot_move: float,
    iv_change: float,
    time_decay_days: int,
    rate: float = 0.04,
) -> dict[str, float]:
    """Decompose P&L into delta/gamma/vega/theta components.

    Uses second-order Taylor expansion:
      dP = delta*dS + 0.5*gamma*dS^2 + vega*dIV + theta*dt
    Returns dict with each component and total.
    """

    delta = black_scholes_delta(
        side=side, spot=spot, strike=strike, t_years=t_years, iv=iv, rate=rate,
    )
    gamma = black_scholes_gamma(
        spot=spot, strike=strike, t_years=t_years, iv=iv, rate=rate,
    )
    vega = black_scholes_vega(
        spot=spot, strike=strike, t_years=t_years, iv=iv, rate=rate,
    )
    theta = black_scholes_theta(
        side=side, spot=spot, strike=strike, t_years=t_years, iv=iv, rate=rate,
    )

    d_s = spot_move
    d_iv = iv_change
    dt = time_decay_days / 365.0

    delta_pnl = delta * d_s
    gamma_pnl = 0.5 * gamma * d_s * d_s
    vega_pnl = vega * d_iv
    theta_pnl = theta * dt
    total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

    return {
        "delta_pnl": round(delta_pnl, 4),
        "gamma_pnl": round(gamma_pnl, 4),
        "vega_pnl": round(vega_pnl, 4),
        "theta_pnl": round(theta_pnl, 4),
        "total": round(total, 4),
    }


# ---------------------------------------------------------------------------
# #7  Delta Hedging simulation
# ---------------------------------------------------------------------------


def delta_hedge_pnl(
    *,
    side: str,
    spot: float,
    strike: float,
    dte_days: int,
    iv: float,
    rebalance_days: int = 1,
    rate: float = 0.04,
    seed: int | None = None,
) -> dict[str, float]:
    """Simulate delta-hedged P&L for a long option position.

    Buys delta shares at start, rebalances every ``rebalance_days``.
    Returns dict with hedged_pnl, unhedged_pnl, hedge_cost.
    """

    rng = random.Random(seed)
    dt = 1.0 / 365.0
    s = spot
    t = dte_days / 365.0
    drift = (rate - 0.5 * iv * iv) * dt
    vol = iv * math.sqrt(dt)

    d = black_scholes_delta(
        side=side, spot=s, strike=strike, t_years=t, iv=iv, rate=rate,
    )
    shares = d * (-100 if side == "put" else 100)
    hedge_cost = s * shares

    unhedged_start = black_scholes_price(
        side=side, spot=s, strike=strike, t_years=t, iv=iv, rate=rate,
    )
    total_hedge_pnl = 0.0

    for day in range(1, dte_days + 1):
        s *= math.exp(drift + vol * rng.gauss(0.0, 1.0))
        t = max(dte_days - day, 0) / 365.0
        if t <= 0:
            break
        if day % rebalance_days == 0:
            new_d = black_scholes_delta(
                side=side, spot=s, strike=strike, t_years=t, iv=iv, rate=rate,
            )
            new_shares = new_d * (-100 if side == "put" else 100)
            total_hedge_pnl += (new_shares - shares) * s
            shares = new_shares

    option_end = black_scholes_price(
        side=side, spot=s, strike=strike, t_years=0, iv=iv, rate=rate,
    ) if t <= 0 else black_scholes_price(
        side=side, spot=s, strike=strike, t_years=t, iv=iv, rate=rate,
    )

    unhedged_pnl = (option_end - unhedged_start) * 100
    hedged_pnl = unhedged_pnl + total_hedge_pnl - hedge_cost * rate * dte_days / 365.0

    return {
        "hedged_pnl": round(hedged_pnl, 2),
        "unhedged_pnl": round(unhedged_pnl, 2),
        "hedge_cost": round(abs(total_hedge_pnl), 2),
    }
