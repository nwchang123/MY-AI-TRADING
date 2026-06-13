from datetime import date, datetime, timezone
from pathlib import Path

from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.risk import Mandate, QuoteSnapshot

NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)


def _validator() -> LiquidityValidator:
    mandate = Mandate.load(Path("config/mandate.paper.yaml"))
    return LiquidityValidator(mandate.options, mandate.execution)


def _quote(**overrides: object) -> QuoteSnapshot:
    values: dict[str, object] = {
        "option_code": "US.EXAMPLE260626C00005000",
        "bid": 0.19,
        "ask": 0.21,
        "open_interest": 200,
        "daily_volume": 30,
        "lot_size": 100,
        "expiry": date(2026, 6, 26),
        "observed_at": NOW,
    }
    values.update(overrides)
    return QuoteSnapshot(**values)


def test_passes_liquid_contract() -> None:
    result = _validator().validate(_quote(), now=NOW)
    assert result.passed is True
    assert result.reasons == []
    assert result.dte == 24
    assert result.estimated_contract_cost_usd == 22  # 0.21*100 + 1 fee buffer
    assert result.suggested_limit_price == 0.21
    assert result.max_limit_price == 0.2205  # ask * 1.05


def test_rejects_wide_spread() -> None:
    result = _validator().validate(_quote(bid=0.10, ask=0.21), now=NOW)
    assert result.passed is False
    assert "bid-ask spread is too wide" in result.reasons


def test_rejects_low_open_interest() -> None:
    result = _validator().validate(_quote(open_interest=5), now=NOW)
    assert "open interest is below minimum" in result.reasons


def test_rejects_low_volume() -> None:
    result = _validator().validate(_quote(daily_volume=0), now=NOW)
    assert "daily option volume is below minimum" in result.reasons


def test_rejects_dte_out_of_range() -> None:
    near = _validator().validate(_quote(expiry=date(2026, 6, 10)), now=NOW)
    assert "days to expiry violate mandate" in near.reasons
    far = _validator().validate(_quote(expiry=date(2026, 9, 1)), now=NOW)
    assert "days to expiry violate mandate" in far.reasons


def test_rejects_premium_above_cost_cap() -> None:
    validator = _validator()
    # Derive an ask whose 1-contract cost (ask*100 + fee_buffer) just exceeds the
    # configured cap, so this stays correct regardless of the mandate's capital.
    cap = validator.options.max_contract_cost_usd
    fee = validator.options.fee_buffer_usd
    ask = round((cap - fee) / 100 + 0.10, 2)
    result = validator.validate(_quote(bid=round(ask - 0.02, 2), ask=ask), now=NOW)
    assert "contract cost exceeds mandate" in result.reasons


def test_rejects_zero_bid() -> None:
    result = _validator().validate(_quote(bid=0.0, ask=0.10), now=NOW)
    assert "option bid is below minimum" in result.reasons


def test_rejects_crossed_market() -> None:
    result = _validator().validate(_quote(bid=0.21, ask=0.20), now=NOW)
    assert "market is crossed or locked" in result.reasons


def test_rejects_stale_quote() -> None:
    stale = _quote(observed_at=datetime(2026, 6, 2, 14, 59, tzinfo=timezone.utc))
    result = _validator().validate(stale, now=NOW)
    assert "quote is stale" in result.reasons
