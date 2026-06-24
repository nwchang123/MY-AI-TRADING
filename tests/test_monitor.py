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


def test_catalyst_window_elapsed_exits_even_on_stale_quote() -> None:
    # The committee's catalyst window ended yesterday; the edge is gone, so the
    # position exits regardless of a stale quote (time-based, like the time stop).
    pos = _position(
        catalyst_window_end=date(2026, 6, 1),  # NOW is 2026-06-02
        observed_at=datetime(2026, 5, 1, tzinfo=timezone.utc),  # very stale
    )
    signals = _monitor().evaluate([pos], NOW)
    assert len(signals) == 1
    assert signals[0].reason == "catalyst window elapsed"


def test_catalyst_window_not_yet_elapsed_does_not_exit() -> None:
    pos = _position(catalyst_window_end=date(2026, 6, 20))  # still ahead of NOW
    assert _monitor().evaluate([pos], NOW) == []


def test_forced_close_takes_priority_over_catalyst_window() -> None:
    # Expiry is imminent (within the 2-trading-day force window) AND the catalyst
    # window has elapsed: the forced-close reason wins.
    pos = _position(
        expiry=date(2026, 6, 3),
        time_stop=date(2026, 6, 3),
        catalyst_window_end=date(2026, 6, 1),
    )
    signals = _monitor().evaluate([pos], NOW)
    assert signals[0].reason == "forced close before expiry"


def test_pre_earnings_exit_fires_even_on_stale_quote() -> None:
    # The pre-earnings exit date has arrived: get out before the print, even on a
    # stale quote (missing it means holding through the event IV crush).
    pos = _position(
        pre_earnings_exit_date=date(2026, 6, 2),  # == market date of NOW
        observed_at=datetime(2026, 5, 1, tzinfo=timezone.utc),  # very stale
    )
    signals = _monitor().evaluate([pos], NOW)
    assert len(signals) == 1
    assert signals[0].reason == "pre-earnings exit"


def test_pre_earnings_exit_not_yet_due_does_not_exit() -> None:
    pos = _position(pre_earnings_exit_date=date(2026, 6, 20))  # still ahead of NOW
    assert _monitor().evaluate([pos], NOW) == []


def test_forced_close_takes_priority_over_pre_earnings_exit() -> None:
    # Expiry is imminent AND the pre-earnings exit is due: forced close wins.
    pos = _position(
        expiry=date(2026, 6, 3),
        time_stop=date(2026, 6, 3),
        pre_earnings_exit_date=date(2026, 6, 1),
    )
    signals = _monitor().evaluate([pos], NOW)
    assert signals[0].reason == "forced close before expiry"


def test_take_profit_fires() -> None:
    signals = _monitor().evaluate([_position(bid=0.40, ask=0.42)], NOW)
    assert len(signals) == 1
    assert signals[0].reason == "take profit"
    assert signals[0].pnl_pct >= 100


def test_trailing_profit_stop_fires_after_profit_giveback() -> None:
    monitor = PositionMonitor(
        force_close_before_expiry_trading_days=2,
        stale_quote_seconds=15,
        trailing_profit_activation_pct=30,
        trailing_profit_giveback_pct=35,
    )
    # Entry 1.00, peak bid 1.50 = +50%. A 35% giveback of open profit triggers
    # at 1.325, so a current bid of 1.30 should exit and still lock profit.
    pos = _position(
        entry_price=1.00,
        take_profit_pct=100,
        bid=1.30,
        ask=1.34,
        peak_bid=1.50,
    )

    signals = monitor.evaluate([pos], NOW)

    assert signals[0].reason == "trailing profit stop"
    assert signals[0].pnl_pct == 30.0


def test_trailing_profit_stop_waits_for_activation() -> None:
    monitor = PositionMonitor(
        force_close_before_expiry_trading_days=2,
        stale_quote_seconds=15,
        trailing_profit_activation_pct=30,
        trailing_profit_giveback_pct=35,
    )
    pos = _position(
        entry_price=1.00,
        take_profit_pct=100,
        bid=1.10,
        ask=1.12,
        peak_bid=1.20,
    )

    assert monitor.evaluate([pos], NOW) == []


def test_stop_loss_fires() -> None:
    signals = _monitor().evaluate([_position(bid=0.09, ask=0.11)], NOW)
    assert signals[0].reason == "stop loss"


def test_iv_crush_exit_fires() -> None:
    monitor = PositionMonitor(
        force_close_before_expiry_trading_days=2,
        stale_quote_seconds=15,
        iv_crush_exit_drop_pct=25,
    )
    pos = _position(entry_iv=1.0, current_iv=0.70)

    signals = monitor.evaluate([pos], NOW)

    assert signals[0].reason == "IV crush exit"


def test_theta_decay_exit_fires() -> None:
    monitor = PositionMonitor(
        force_close_before_expiry_trading_days=2,
        stale_quote_seconds=15,
        theta_decay_exit_pct_per_day=8,
    )
    pos = _position(theta_decay_pct_per_day=9.0)

    signals = monitor.evaluate([pos], NOW)

    assert signals[0].reason == "theta decay exit"


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
