from datetime import date, datetime, timezone
from pathlib import Path

from trading_agent.domain.proposals import ExitPlan, OpenPositionProposal
from trading_agent.domain.risk import Mandate, PortfolioState, QuoteSnapshot, RiskGate

NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)


def _mandate() -> Mandate:
    # Pin the canonical $100 risk profile so the gate's dollar-limit assertions
    # (cost cap, drawdown stop, etc.) stay stable when the live paper mandate's
    # capital/caps change (e.g. raised to a $500 base).
    mandate = Mandate.load(Path("config/mandate.paper.yaml"))
    account = mandate.account.model_copy(update={"initial_capital_usd": 100.0})
    options = mandate.options.model_copy(update={"max_contract_cost_usd": 65.0})
    portfolio = mandate.portfolio.model_copy(
        update={
            "max_total_premium_at_risk_usd": 100.0,
            "daily_loss_stop_usd": 35.0,
            "hard_drawdown_stop_usd": 50.0,
        }
    )
    return mandate.model_copy(
        update={"account": account, "options": options, "portfolio": portfolio}
    )


def _proposal() -> OpenPositionProposal:
    return OpenPositionProposal(
        decision="open_position",
        ticker="EXAMPLE",
        option_code="US.EXAMPLE260626C00005000",
        option_side="call",
        action="buy_to_open",
        contracts=1,
        limit_price=0.2,
        max_limit_price=0.21,
        thesis="Public catalyst with bounded premium risk.",
        evidence_ids=["evt_1"],
        confidence=0.7,
        expected_catalyst_window="2026-06-10/2026-06-20",
        exit_plan=ExitPlan(
            take_profit_pct=100,
            stop_loss_pct=50,
            time_stop=date(2026, 6, 24),
        ),
        invalidation=["Catalyst is delayed"],
    )


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


def _portfolio(**overrides: object) -> PortfolioState:
    values: dict[str, object] = {
        "open_positions": 0,
        "total_premium_at_risk_usd": 0,
        "new_positions_today": 0,
        "daily_pnl_usd": 0,
        "total_drawdown_usd": 0,
        "consecutive_losses": 0,
        "duplicate_order_exists": False,
    }
    values.update(overrides)
    return PortfolioState(**values)


def test_approves_proposal_inside_mandate(tmp_path: Path) -> None:
    result = RiskGate(_mandate(), tmp_path).evaluate_open(
        _proposal(), _quote(), _portfolio(), NOW
    )

    assert result.approved is True
    assert result.estimated_contract_cost_usd == 21


def test_rejects_low_estimated_win_probability(tmp_path: Path) -> None:
    # Paper mandate floor is 0.40; the proposal's confidence is the PM's win
    # estimate, so 0.3 must be rejected even if everything else is fine.
    low = _proposal().model_copy(update={"confidence": 0.3})
    result = RiskGate(_mandate(), tmp_path).evaluate_open(
        low, _quote(), _portfolio(), NOW
    )

    assert result.approved is False
    assert "estimated win probability below mandate minimum" in result.reasons


def test_rejects_wide_spread(tmp_path: Path) -> None:
    result = RiskGate(_mandate(), tmp_path).evaluate_open(
        _proposal(), _quote(bid=0.10, ask=0.21), _portfolio(), NOW
    )

    assert result.approved is False
    assert "bid-ask spread is too wide" in result.reasons


def test_rejects_cost_above_contract_limit(tmp_path: Path) -> None:
    # 0.70*100 + 1 = 71 > the $65 cap.
    proposal = _proposal().model_copy(update={"limit_price": 0.70, "max_limit_price": 0.70})
    result = RiskGate(_mandate(), tmp_path).evaluate_open(
        proposal, _quote(bid=0.69, ask=0.70), _portfolio(), NOW
    )

    assert result.approved is False
    assert "contract cost exceeds mandate" in result.reasons


def test_rejects_when_halt_file_exists(tmp_path: Path) -> None:
    halt_path = tmp_path / "runtime" / "HALT"
    halt_path.parent.mkdir(parents=True)
    halt_path.write_text("halt\n", encoding="ascii")

    result = RiskGate(_mandate(), tmp_path).evaluate_open(
        _proposal(), _quote(), _portfolio(), NOW
    )

    assert result.approved is False
    assert "kill switch is active" in result.reasons


def test_rejects_stale_quote(tmp_path: Path) -> None:
    result = RiskGate(_mandate(), tmp_path).evaluate_open(
        _proposal(),
        _quote(observed_at=datetime(2026, 6, 2, 14, 59, tzinfo=timezone.utc)),
        _portfolio(),
        NOW,
    )

    assert result.approved is False
    assert "quote is stale" in result.reasons


def test_rejects_hard_drawdown(tmp_path: Path) -> None:
    result = RiskGate(_mandate(), tmp_path).evaluate_open(
        _proposal(), _quote(), _portfolio(total_drawdown_usd=50), NOW
    )

    assert result.approved is False
    assert "hard drawdown stop is active" in result.reasons

