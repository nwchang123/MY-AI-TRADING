import math

import pytest

from trading_agent.domain.montecarlo import (
    black_scholes_price,
    stable_seed,
    win_probability,
)


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
