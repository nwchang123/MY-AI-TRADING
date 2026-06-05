import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from shutil import copyfile

from trading_agent.backtest import (
    BacktestScenario,
    BacktestStep,
    CostModel,
    run_backtest,
)
from trading_agent.cli import main
from trading_agent.domain.proposals import ExitPlan, OpenPositionProposal
from trading_agent.domain.risk import Mandate, QuoteSnapshot

ROOT = Path(__file__).parents[1]
NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)
NEXT = datetime(2026, 6, 3, 15, 0, tzinfo=timezone.utc)
DAY3 = datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc)
DAY4 = datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc)
OPTION_CODE = "US.EXAMPLE260626C00005000"
OTHER_CODE = "US.OTHER260626C00005000"


def _mandate() -> Mandate:
    return Mandate.load(ROOT / "config" / "mandate.paper.yaml")


def _proposal(
    *,
    code: str = OPTION_CODE,
    ticker: str = "EXAMPLE",
    limit_price: float = 0.21,
    max_limit_price: float = 0.22,
) -> OpenPositionProposal:
    return OpenPositionProposal(
        decision="open_position",
        ticker=ticker,
        option_code=code,
        option_side="call",
        action="buy_to_open",
        contracts=1,
        limit_price=limit_price,
        max_limit_price=max_limit_price,
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


def _quote(
    *,
    code: str = OPTION_CODE,
    bid: float = 0.19,
    ask: float = 0.21,
    now: datetime = NOW,
) -> QuoteSnapshot:
    return QuoteSnapshot(
        option_code=code,
        bid=bid,
        ask=ask,
        open_interest=200,
        daily_volume=30,
        lot_size=100,
        expiry=date(2026, 6, 26),
        observed_at=now,
    )


def test_backtest_opens_and_closes_at_bid(tmp_path: Path) -> None:
    scenario = BacktestScenario(
        steps=[
            BacktestStep(now=NOW, quote=_quote(), proposal=_proposal()),
            BacktestStep(now=NEXT, quote=_quote(bid=0.42, ask=0.44, now=NEXT)),
        ]
    )

    result = run_backtest(scenario, mandate=_mandate(), root_dir=tmp_path)

    assert result.summary["trades_opened"] == 1
    assert result.summary["trades_closed"] == 1
    assert result.summary["realized_pnl_usd"] == 19
    assert result.summary["ending_equity_usd"] == 119
    assert result.summary["return_pct"] == 19
    assert result.summary["fees_paid_usd"] == 2
    assert result.summary["win_rate"] == 1
    assert result.summary["max_drawdown_usd"] == 3
    assert result.summary["avg_entry_spread_pct"] == 10
    assert result.summary["avg_exit_spread_pct"] == 4.6512
    assert result.closed_trades[0]["gross_pnl_usd"] == 21
    assert result.closed_trades[0]["fees_usd"] == 2
    assert result.closed_trades[0]["realized_pnl_usd"] == 19
    assert result.closed_trades[0]["exit_price"] == 0.42
    assert result.events[-1]["event_type"] == "position_closed"


def test_backtest_rejects_unmarketable_buy_limit(tmp_path: Path) -> None:
    scenario = BacktestScenario(
        steps=[
            BacktestStep(
                now=NOW,
                quote=_quote(),
                proposal=_proposal(limit_price=0.20, max_limit_price=0.21),
            )
        ]
    )

    result = run_backtest(scenario, mandate=_mandate(), root_dir=tmp_path)

    assert result.summary["trades_opened"] == 0
    assert result.summary["rejected_by_stage"] == {"unfilled": 1}
    assert result.events[0]["payload"]["reasons"] == ["ask is above the limit price"]


def test_backtest_daily_loss_stop_blocks_same_day_entry(tmp_path: Path) -> None:
    scenario = BacktestScenario(
        steps=[
            BacktestStep(now=NOW, quote=_quote(), proposal=_proposal()),
            BacktestStep(
                now=NEXT,
                quotes=[
                    _quote(bid=0.09, ask=0.11, now=NEXT),
                    _quote(code=OTHER_CODE, bid=0.19, ask=0.21, now=NEXT),
                ],
                proposal=_proposal(code=OTHER_CODE, ticker="OTHER"),
            ),
        ]
    )

    result = run_backtest(scenario, mandate=_mandate(), root_dir=tmp_path)

    assert result.summary["realized_pnl_usd"] == -14
    assert result.summary["trades_opened"] == 1
    assert result.summary["rejected_by_stage"] == {"risk_gate": 1}
    assert result.events[-1]["payload"]["reasons"] == ["daily loss stop is active"]


def test_backtest_commission_model_per_contract_min_and_platform(tmp_path: Path) -> None:
    # Per side: max(min 1.00, 0.65 * 1) + 0.50 platform = 1.50 -> 3.00 round trip.
    scenario = BacktestScenario(
        costs=CostModel(
            commission_per_contract_usd=0.65,
            commission_min_usd=1.00,
            platform_fee_per_order_usd=0.50,
        ),
        steps=[
            BacktestStep(now=NOW, quote=_quote(), proposal=_proposal()),
            BacktestStep(now=NEXT, quote=_quote(bid=0.42, ask=0.44, now=NEXT)),
        ],
    )

    result = run_backtest(scenario, mandate=_mandate(), root_dir=tmp_path)

    assert result.closed_trades[0]["gross_pnl_usd"] == 21
    assert result.closed_trades[0]["fees_usd"] == 3.0
    assert result.closed_trades[0]["realized_pnl_usd"] == 18.0
    assert result.summary["fees_paid_usd"] == 3.0


def test_backtest_commission_waiver_runs_out_across_trades(tmp_path: Path) -> None:
    # USD 2.50 waiver, USD 1.00/side: trade 1 fully free; trade 2 pays 1.50.
    scenario = BacktestScenario(
        costs=CostModel(commission_min_usd=1.00, commission_waiver_usd=2.50),
        steps=[
            BacktestStep(now=NOW, quote=_quote(), proposal=_proposal()),
            BacktestStep(now=NEXT, quote=_quote(bid=0.42, ask=0.44, now=NEXT)),
            BacktestStep(
                now=DAY3,
                quote=_quote(code=OTHER_CODE, now=DAY3),
                proposal=_proposal(code=OTHER_CODE, ticker="OTHER"),
            ),
            BacktestStep(now=DAY4, quote=_quote(code=OTHER_CODE, bid=0.42, ask=0.44, now=DAY4)),
        ],
    )

    result = run_backtest(scenario, mandate=_mandate(), root_dir=tmp_path)

    assert result.summary["trades_closed"] == 2
    assert result.closed_trades[0]["fees_usd"] == 0.0  # first trade rides the waiver
    assert result.closed_trades[1]["fees_usd"] == 1.5  # waiver exhausted mid-trade
    assert result.summary["fees_paid_usd"] == 1.5
    assert result.summary["commission_waiver_remaining_usd"] == 0.0


def test_backtest_entry_slippage_capped_at_limit_exit_slippage_applied(tmp_path: Path) -> None:
    # ask 0.20 + 0.05 slippage = 0.25, but the 0.21 limit caps the entry at 0.21.
    # Exit bid 0.45 - 0.05 slippage = 0.40.
    scenario = BacktestScenario(
        costs=CostModel(slippage_usd_per_share=0.05),  # zero commission isolates slippage
        steps=[
            BacktestStep(
                now=NOW,
                quote=_quote(bid=0.18, ask=0.20),
                proposal=_proposal(limit_price=0.21, max_limit_price=0.21),
            ),
            BacktestStep(now=NEXT, quote=_quote(bid=0.45, ask=0.47, now=NEXT)),
        ],
    )

    result = run_backtest(scenario, mandate=_mandate(), root_dir=tmp_path)

    assert result.closed_trades[0]["entry_price"] == 0.21  # capped at the limit
    assert result.closed_trades[0]["exit_price"] == 0.40  # bid minus slippage
    assert result.closed_trades[0]["fees_usd"] == 0.0


def test_backtest_wide_spread_blocks_exit_fill(tmp_path: Path) -> None:
    # Take-profit triggers, but the 35% exit spread exceeds the 10% fill cap, so
    # the close does not fill and the position lingers.
    scenario = BacktestScenario(
        costs=CostModel(max_exit_fill_spread_pct=10.0),
        steps=[
            BacktestStep(now=NOW, quote=_quote(), proposal=_proposal()),
            BacktestStep(now=NEXT, quote=_quote(bid=0.42, ask=0.60, now=NEXT)),
        ],
    )

    result = run_backtest(scenario, mandate=_mandate(), root_dir=tmp_path)

    assert result.summary["trades_opened"] == 1
    assert result.summary["trades_closed"] == 0
    assert result.summary["open_positions"] == 1
    assert result.summary["exits_unfilled"] == 1
    assert any(event["event_type"] == "exit_unfilled" for event in result.events)


def test_backtest_cli_outputs_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    copyfile(ROOT / "config" / "mandate.paper.yaml", config_dir / "mandate.paper.yaml")
    input_path = tmp_path / "backtest.json"
    input_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "now": NOW.isoformat(),
                        "quote": _quote().model_dump(mode="json"),
                        "proposal": _proposal().model_dump(mode="json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["trading-agent", "backtest", "--input", str(input_path)]
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["trades_opened"] == 1
    assert payload["summary"]["open_positions"] == 1
