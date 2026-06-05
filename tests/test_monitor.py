from datetime import date, datetime, timezone

from trading_agent.domain.positions import MonitoredPosition, trading_days_until
from trading_agent.execution.monitor import PositionMonitor

NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)  # a Tuesday


def _monitor() -> PositionMonitor:
    return PositionMonitor(force_close_before_expiry_trading_days=2, stale_quote_seconds=15)


def _position(**overrides: object) -> MonitoredPosition:
    values: dict[str, object] = {
        "option_code": "US.EXAMPLE260626C00005000",
        "option_side": "call",
        "entry_price": 0.20,
        "contracts": 1,
        "lot_size": 100,
        "expiry": date(2026, 6, 26),
        "take_profit_pct": 100,
        "stop_loss_pct": 50,
        "time_stop": date(2026, 6, 24),
        "bid": 0.19,
        "ask": 0.21,
        "observed_at": NOW,
    }
    values.update(overrides)
    return MonitoredPosition(**values)


def test_trading_days_until_excludes_weekends() -> None:
    # Tue 2026-06-02 -> Fri 2026-06-05 is 3 trading days.
    assert trading_days_until(date(2026, 6, 5), NOW) == 3
    # Expired contract trips immediately.
    assert trading_days_until(date(2026, 6, 1), NOW) == 0


def test_mark_price_uses_bid_not_mid() -> None:
    # A long option is closed at the bid; the mark must be the bid, never the mid.
    assert _position(bid=0.30, ask=0.50).mark_price() == 0.30
    # No bid means no exit liquidity -> worthless mark.
    assert _position(bid=0.0, ask=0.50).mark_price() == 0.0


def test_exit_pnl_reflects_the_full_round_trip_spread() -> None:
    # Entry paid the ask (0.20). With bid at 0.20 the mid would show +5% (ask 0.22),
    # but marking at the bid correctly shows break-even, not a phantom gain.
    pos = _position(entry_price=0.20, bid=0.20, ask=0.22)
    signals = _monitor().evaluate([pos], NOW)
    assert signals == []  # +0% at the bid, so neither take-profit nor stop fires


def test_no_signal_when_inside_plan() -> None:
    assert _monitor().evaluate([_position()], NOW) == []


def test_take_profit_fires() -> None:
    signals = _monitor().evaluate([_position(bid=0.40, ask=0.42)], NOW)
    assert len(signals) == 1
    assert signals[0].reason == "take profit"
    assert signals[0].pnl_pct >= 100


def test_stop_loss_fires() -> None:
    signals = _monitor().evaluate([_position(bid=0.09, ask=0.11)], NOW)
    assert signals[0].reason == "stop loss"


def test_time_stop_fires() -> None:
    later = datetime(2026, 6, 24, 15, 0, tzinfo=timezone.utc)
    # Far expiry so only the time stop (not forced close) is in play.
    signals = _monitor().evaluate(
        [_position(expiry=date(2026, 7, 31), observed_at=later)], later
    )
    assert signals[0].reason == "time stop reached"


def test_forced_close_before_expiry_fires() -> None:
    # Two trading days before the 2026-06-26 (Fri) expiry is Wed 2026-06-24.
    near = datetime(2026, 6, 24, 15, 0, tzinfo=timezone.utc)
    signals = _monitor().evaluate(
        [_position(time_stop=date(2026, 6, 30), observed_at=near)], near
    )
    assert signals[0].reason == "forced close before expiry"


def test_stale_quote_blocks_pnl_exit_but_not_time_exit() -> None:
    stale_obs = datetime(2026, 6, 2, 14, 0, tzinfo=timezone.utc)  # > 15s old
    # P/L would say take-profit, but the quote is stale, so no P/L exit.
    assert _monitor().evaluate([_position(bid=0.40, ask=0.42, observed_at=stale_obs)], NOW) == []
    # A forced close still fires on a stale quote (expiry safety).
    near = datetime(2026, 6, 24, 15, 0, tzinfo=timezone.utc)
    stale_near = datetime(2026, 6, 24, 14, 0, tzinfo=timezone.utc)
    signals = _monitor().evaluate(
        [_position(time_stop=date(2026, 6, 30), observed_at=stale_near)], near
    )
    assert signals[0].reason == "forced close before expiry"
