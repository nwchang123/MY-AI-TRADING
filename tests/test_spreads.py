from trading_agent.domain.proposals import SpreadLeg
from trading_agent.domain.spreads import analyze_spread, classify_spread


def _leg(code: str, side: str, action: str, price: float) -> SpreadLeg:
    return SpreadLeg(
        option_code=code,
        option_side=side,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        contracts=1,
        limit_price=price,
    )


def test_analyze_bull_call_spread() -> None:
    legs = [
        _leg("US.ABCD260626C5000", "call", "buy_to_open", 1.20),
        _leg("US.ABCD260626C7000", "call", "sell_to_open", 0.40),
    ]

    analysis = analyze_spread(legs)

    assert analysis.strategy == "bull_call_spread"
    assert analysis.ticker == "ABCD"
    assert analysis.net_debit_usd == 80
    assert analysis.net_credit_usd == 0
    assert analysis.max_loss_usd == 80
    assert analysis.max_profit_usd == 120
    assert analysis.risk_reward == 1.5
    assert analysis.breakeven_prices == [5.8]
    assert not analysis.unlimited_profit
    assert not analysis.unlimited_loss


def test_analyze_long_straddle_has_two_breakevens_and_unlimited_profit() -> None:
    legs = [
        _leg("US.ABCD260626C5000", "call", "buy_to_open", 0.60),
        _leg("US.ABCD260626P5000", "put", "buy_to_open", 0.50),
    ]

    analysis = analyze_spread(legs, underlying_price=5.0)

    assert analysis.strategy == "long_straddle"
    assert analysis.net_debit_usd == 110
    assert analysis.max_loss_usd == 110
    assert analysis.max_profit_usd is None
    assert analysis.unlimited_profit
    assert analysis.breakeven_prices == [3.9, 6.1]


def test_classify_iron_condor() -> None:
    legs = [
        _leg("US.ABCD260626P4000", "put", "buy_to_open", 0.10),
        _leg("US.ABCD260626P5000", "put", "sell_to_open", 0.30),
        _leg("US.ABCD260626C7000", "call", "sell_to_open", 0.35),
        _leg("US.ABCD260626C8000", "call", "buy_to_open", 0.15),
    ]

    assert classify_spread(legs) == "iron_condor"
    analysis = analyze_spread(legs)
    assert analysis.net_credit_usd == 40
    assert analysis.max_profit_usd == 40
    assert analysis.max_loss_usd == 60


def test_side_mismatch_is_rejected() -> None:
    legs = [
        _leg("US.ABCD260626C5000", "put", "buy_to_open", 0.20),
        _leg("US.ABCD260626P4000", "put", "buy_to_open", 0.10),
    ]

    try:
        analyze_spread(legs)
    except ValueError as exc:
        assert "side does not match" in str(exc)
    else:
        raise AssertionError("side mismatch should fail")
