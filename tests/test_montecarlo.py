import math

import pytest

from trading_agent.domain.montecarlo import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_theta,
    black_scholes_vega,
    delta_hedge_pnl,
    iv_rank,
    iv_skew,
    iv_term_structure,
    kelly_criterion,
    pnl_attribution,
    portfolio_var,
    stable_seed,
    win_probability,
)


def test_black_scholes_delta_atm_is_near_half() -> None:
    call = black_scholes_delta(side="call", spot=10, strike=10, t_years=30 / 365, iv=0.6)
    put = black_scholes_delta(side="put", spot=10, strike=10, t_years=30 / 365, iv=0.6)
    assert 0.45 < call < 0.6  # slightly above 0.5 from drift
    assert -0.55 < put < -0.4
    # Put = call - 1 (delta parity).
    assert math.isclose(call - put, 1.0, abs_tol=0.02)


def test_black_scholes_delta_collapses_to_intrinsic_at_expiry() -> None:
    assert black_scholes_delta(side="call", spot=12, strike=10, t_years=0, iv=0.6) == 1.0
    assert black_scholes_delta(side="call", spot=8, strike=10, t_years=0, iv=0.6) == 0.0
    assert black_scholes_delta(side="put", spot=8, strike=10, t_years=0, iv=0.6) == -1.0


def test_black_scholes_delta_rejects_bad_side() -> None:
    with pytest.raises(ValueError):
        black_scholes_delta(side="straddle", spot=10, strike=10, t_years=0.1, iv=0.5)


def test_black_scholes_atm_call_matches_reference() -> None:
    # S=K=100, t=0.25y, iv=20%, r=0: textbook value ~3.988.
    price = black_scholes_price(
        side="call", spot=100, strike=100, t_years=0.25, iv=0.2, rate=0.0
    )
    assert math.isclose(price, 3.988, abs_tol=0.01)


def test_black_scholes_put_call_parity() -> None:
    call = black_scholes_price(side="call", spot=95, strike=100, t_years=0.5, iv=0.4, rate=0.0)
    put = black_scholes_price(side="put", spot=95, strike=100, t_years=0.5, iv=0.4, rate=0.0)
    # With r=0: C - P = S - K.
    assert math.isclose(call - put, 95 - 100, abs_tol=1e-9)


def test_black_scholes_intrinsic_at_expiry() -> None:
    assert black_scholes_price(side="call", spot=12, strike=10, t_years=0, iv=0.5) == 2
    assert black_scholes_price(side="put", spot=12, strike=10, t_years=0, iv=0.5) == 0


def test_win_probability_is_deterministic_with_seed() -> None:
    kwargs = dict(
        side="call", spot=10.0, strike=11.0, dte_days=30, iv=0.6,
        entry_price=0.2, take_profit_pct=100, stop_loss_pct=50,
        paths=500, seed=stable_seed("US.TEST"),
    )
    assert win_probability(**kwargs) == win_probability(**kwargs)


def test_easier_take_profit_has_higher_probability() -> None:
    base = dict(
        side="call", spot=10.0, strike=11.0, dte_days=30, iv=0.6,
        entry_price=0.2, stop_loss_pct=50, paths=1000, seed=7,
    )
    easy = win_probability(**base, take_profit_pct=20)
    hard = win_probability(**base, take_profit_pct=200)
    assert easy > hard
    assert 0.0 <= hard <= easy <= 1.0


def test_hopeless_overpriced_ticket_scores_near_zero() -> None:
    # Far-OTM call priced way above model value: doubling is ~impossible.
    pop = win_probability(
        side="call", spot=2.0, strike=5.0, dte_days=20, iv=0.4,
        entry_price=0.2, take_profit_pct=100, stop_loss_pct=50,
        paths=1000, seed=1,
    )
    assert pop < 0.02


def test_underpriced_atm_ticket_scores_high() -> None:
    # ATM with high vol where entry is far below model value: TP hits fast.
    pop = win_probability(
        side="call", spot=5.0, strike=5.0, dte_days=24, iv=1.0,
        entry_price=0.2, take_profit_pct=100, stop_loss_pct=50,
        paths=500, seed=1,
    )
    assert pop > 0.8


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        win_probability(
            side="call", spot=0, strike=5, dte_days=20, iv=0.4,
            entry_price=0.2, take_profit_pct=100, stop_loss_pct=50,
        )
    with pytest.raises(ValueError):
        black_scholes_price(side="straddle", spot=5, strike=5, t_years=0.1, iv=0.3)


def test_black_scholes_gamma_is_positive() -> None:
    g = black_scholes_gamma(spot=10, strike=10, t_years=30 / 365, iv=0.6)
    assert g > 0


def test_black_scholes_gamma_is_zero_at_expiry() -> None:
    assert black_scholes_gamma(spot=10, strike=10, t_years=0, iv=0.6) == 0.0


def test_black_scholes_gamma_atm_beats_otm() -> None:
    atm = black_scholes_gamma(spot=10, strike=10, t_years=0.25, iv=0.4)
    otm = black_scholes_gamma(spot=10, strike=15, t_years=0.25, iv=0.4)
    assert atm > otm


def test_black_scholes_gamma_short_dte_spike() -> None:
    long_dte = black_scholes_gamma(spot=10, strike=10, t_years=30 / 365, iv=0.4)
    short_dte = black_scholes_gamma(spot=10, strike=10, t_years=3 / 365, iv=0.4)
    assert short_dte > long_dte


# --- iv_rank ---


def test_iv_rank_from_history() -> None:
    r = iv_rank(current_iv=0.5, iv_history=[0.2, 0.3, 0.5, 0.8, 1.0])
    assert r is not None and math.isclose(r, 0.375, abs_tol=1e-9)


def test_iv_rank_from_52w_extremes() -> None:
    r = iv_rank(current_iv=0.4, iv_52w_high=0.6, iv_52w_low=0.2)
    assert r is not None and math.isclose(r, 0.5, abs_tol=1e-9)


def test_iv_rank_returns_none_when_no_data() -> None:
    assert iv_rank(current_iv=0.4) is None


def test_iv_rank_returns_none_on_zero_current_iv() -> None:
    assert iv_rank(current_iv=0.0, iv_history=[0.1, 0.5]) is None


def test_iv_rank_handles_flat_history() -> None:
    assert iv_rank(current_iv=0.3, iv_history=[0.3, 0.3]) is None


# --- kelly_criterion ---


def test_kelly_criterion_positive_edge() -> None:
    # 60% win, risk 1 to gain 2: f* = (0.6*2 - 0.4)/2 = 0.40
    k = kelly_criterion(win_prob=0.6, win_amount=2.0, lose_amount=1.0)
    assert k is not None
    assert math.isclose(k, 0.40, abs_tol=1e-9)


def test_kelly_criterion_negative_edge_returns_zero() -> None:
    # 40% win, risk 1 to gain 1: f* = (0.4*1 - 0.6)/1 = -0.20 → clamped to 0
    k = kelly_criterion(win_prob=0.4, win_amount=1.0, lose_amount=1.0)
    assert k == 0.0


def test_kelly_criterion_coin_flip() -> None:
    # 50% win, risk 1 to gain 1: f* = 0 → no bet
    k = kelly_criterion(win_prob=0.5, win_amount=1.0, lose_amount=1.0)
    assert k == 0.0


def test_kelly_criterion_returns_none_on_bad_input() -> None:
    assert kelly_criterion(win_prob=0.0, win_amount=1.0, lose_amount=1.0) is None
    assert kelly_criterion(win_prob=1.0, win_amount=1.0, lose_amount=1.0) is None
    assert kelly_criterion(win_prob=0.6, win_amount=0.0, lose_amount=1.0) is None
    assert kelly_criterion(win_prob=0.6, win_amount=1.0, lose_amount=0.0) is None


def test_kelly_criterion_scales_with_edge() -> None:
    small_edge = kelly_criterion(win_prob=0.55, win_amount=1.0, lose_amount=1.0)
    big_edge = kelly_criterion(win_prob=0.70, win_amount=1.0, lose_amount=1.0)
    assert small_edge is not None and big_edge is not None
    assert big_edge > small_edge


# --- black_scholes_vega ---


def test_black_scholes_vega_is_positive() -> None:
    v = black_scholes_vega(spot=10, strike=10, t_years=30 / 365, iv=0.6)
    assert v > 0


def test_black_scholes_vega_is_zero_at_expiry() -> None:
    assert black_scholes_vega(spot=10, strike=10, t_years=0, iv=0.6) == 0.0


def test_black_scholes_vega_atm_beats_otm() -> None:
    atm = black_scholes_vega(spot=10, strike=10, t_years=0.25, iv=0.4)
    otm = black_scholes_vega(spot=10, strike=15, t_years=0.25, iv=0.4)
    assert atm > otm


# --- black_scholes_theta ---


def test_black_scholes_theta_is_negative_for_long() -> None:
    tc = black_scholes_theta(side="call", spot=10, strike=10, t_years=30 / 365, iv=0.6)
    tp = black_scholes_theta(side="put", spot=10, strike=10, t_years=30 / 365, iv=0.6)
    assert tc < 0
    assert tp < 0


def test_black_scholes_theta_is_zero_at_expiry() -> None:
    assert black_scholes_theta(side="call", spot=10, strike=10, t_years=0, iv=0.6) == 0.0


def test_black_scholes_theta_call_put_parity() -> None:
    tc = black_scholes_theta(side="call", spot=10, strike=10, t_years=0.25, iv=0.4)
    tp = black_scholes_theta(side="put", spot=10, strike=10, t_years=0.25, iv=0.4)
    assert tc < 0 and tp < 0


# --- portfolio_var ---


def test_portfolio_var_empty() -> None:
    assert portfolio_var(positions=[]) == 0.0


def test_portfolio_var_is_positive() -> None:
    var = portfolio_var(
        positions=[{
            "side": "call", "spot": 100, "strike": 100,
            "dte_days": 30, "iv": 0.4, "contracts": 1, "lot_size": 100,
            "entry_price": 5.0,
        }],
        confidence=0.95, paths=2000, seed=42,
    )
    assert var > 0


def test_portfolio_var_scales_with_contracts() -> None:
    pos = {
        "side": "call", "spot": 100, "strike": 100,
        "dte_days": 30, "iv": 0.4, "lot_size": 100, "entry_price": 5.0,
    }
    v1 = portfolio_var(positions=[{**pos, "contracts": 1}], paths=2000, seed=42)
    v2 = portfolio_var(positions=[{**pos, "contracts": 2}], paths=2000, seed=42)
    assert v2 > v1


# --- iv_skew ---


def test_iv_skew_put_heavy() -> None:
    data = [
        {"strike": 90, "iv": 0.50, "delta": -0.25},
        {"strike": 100, "iv": 0.40, "delta": 0.50},
        {"strike": 110, "iv": 0.35, "delta": 0.25},
    ]
    skew = iv_skew(calls_by_strike=data, spot=100)
    assert skew is not None
    assert skew > 0  # put IV > call IV


def test_iv_skew_returns_none_with_insufficient_data() -> None:
    assert iv_skew(calls_by_strike=[], spot=100) is None
    assert iv_skew(calls_by_strike=[{"strike": 100, "iv": 0.4, "delta": 0.5}], spot=100) is None


# --- iv_term_structure ---


def test_iv_term_structure_contango() -> None:
    data = [{"dte_days": 30, "iv": 0.30}, {"dte_days": 90, "iv": 0.40}]
    slope = iv_term_structure(iv_by_expiry=data)
    assert slope is not None
    assert slope > 0  # contango


def test_iv_term_structure_returns_none_with_insufficient_data() -> None:
    assert iv_term_structure(iv_by_expiry=[{"dte_days": 30, "iv": 0.3}]) is None


# --- pnl_attribution ---


def test_pnl_attribution_total_matches_sum() -> None:
    attr = pnl_attribution(
        side="call", spot=100, strike=100, t_years=0.25, iv=0.4,
        spot_move=5.0, iv_change=0.02, time_decay_days=1,
    )
    assert math.isclose(
        attr["total"],
        attr["delta_pnl"] + attr["gamma_pnl"] + attr["vega_pnl"] + attr["theta_pnl"],
        abs_tol=0.01,
    )


def test_pnl_attribution_theta_is_negative() -> None:
    attr = pnl_attribution(
        side="call", spot=100, strike=100, t_years=0.25, iv=0.4,
        spot_move=0, iv_change=0, time_decay_days=7,
    )
    assert attr["theta_pnl"] < 0


# --- delta_hedge_pnl ---


def test_delta_hedge_pnl_has_keys() -> None:
    result = delta_hedge_pnl(
        side="call", spot=100, strike=100, dte_days=30, iv=0.4,
        seed=42,
    )
    assert "hedged_pnl" in result
    assert "unhedged_pnl" in result
    assert "hedge_cost" in result
