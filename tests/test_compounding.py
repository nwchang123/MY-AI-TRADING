from pathlib import Path

from trading_agent.domain.risk import Mandate


def _mandate(compounding: bool = True) -> Mandate:
    mandate = Mandate.load(Path("config/mandate.paper.yaml"))
    account = mandate.account.model_copy(update={"compounding": compounding})
    return mandate.model_copy(update={"account": account})


def test_not_compounding_returns_same_object() -> None:
    mandate = _mandate(compounding=False)
    assert mandate.scaled_for_equity(200.0, 200.0) is mandate


def test_doubled_equity_doubles_dollar_caps() -> None:
    scaled = _mandate().scaled_for_equity(200.0, 200.0)
    assert scaled.options.max_contract_cost_usd == 130.0
    assert scaled.portfolio.max_total_premium_at_risk_usd == 200.0
    assert scaled.portfolio.daily_loss_stop_usd == 70.0
    assert scaled.portfolio.hard_drawdown_stop_usd == 100.0
    # Counts and non-dollar limits never scale.
    assert scaled.portfolio.max_open_positions == 2
    assert scaled.options.contracts_per_order == 1
    assert scaled.options.max_bid_ask_spread_pct == 15
    assert scaled.options.fee_buffer_usd == 1


def test_losses_shrink_caps_but_drawdown_keeps_peak_reference() -> None:
    # Equity fell to 80 while the peak was 100: sizing de-risks to 80%, but the
    # hard drawdown stop stays referenced to the peak (still $50).
    scaled = _mandate().scaled_for_equity(80.0, 100.0)
    assert scaled.options.max_contract_cost_usd == 52.0
    assert scaled.portfolio.daily_loss_stop_usd == 28.0
    assert scaled.portfolio.hard_drawdown_stop_usd == 50.0


def test_scale_floor_prevents_degenerate_caps() -> None:
    scaled = _mandate().scaled_for_equity(1.0, 100.0)
    # 10% floor: caps never shrink below a tenth of their configured values.
    assert scaled.options.max_contract_cost_usd == 6.5
    assert scaled.portfolio.daily_loss_stop_usd == 3.5


def test_unchanged_equity_is_a_noop() -> None:
    mandate = _mandate()
    assert mandate.scaled_for_equity(100.0, 100.0) is mandate
