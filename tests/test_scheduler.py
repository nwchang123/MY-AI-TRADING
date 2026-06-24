from datetime import datetime, timezone

from trading_agent.domain.calendar import is_market_hours
from trading_agent.execution.scheduler import run_scheduler

# 2026-06-02 is a Tuesday. 14:00 UTC = 10:00 ET (open); 21:30 UTC = 17:30 ET (closed).
OPEN_UTC = datetime(2026, 6, 2, 14, 0, tzinfo=timezone.utc)
CLOSED_UTC = datetime(2026, 6, 2, 21, 30, tzinfo=timezone.utc)
WEEKEND_UTC = datetime(2026, 6, 6, 14, 0, tzinfo=timezone.utc)  # Saturday


def _runner():
    calls = {"n": 0}

    def run_cycle() -> None:
        calls["n"] += 1

    return run_cycle, calls


def test_market_hours_detects_open_and_closed() -> None:
    assert is_market_hours(OPEN_UTC) is True
    assert is_market_hours(CLOSED_UTC) is False
    assert is_market_hours(WEEKEND_UTC) is False  # weekend
    # MLK Day 2026-01-19 (Mon) is a holiday, even at 10:00 ET.
    holiday = datetime(2026, 1, 19, 15, 0, tzinfo=timezone.utc)
    assert is_market_hours(holiday) is False


def test_loop_runs_each_tick_in_hours() -> None:
    run_cycle, calls = _runner()
    sleeps: list = []
    ran = run_scheduler(
        run_cycle=run_cycle,
        is_halted=lambda: False,
        interval_seconds=60,
        sleep_fn=sleeps.append,
        now_fn=lambda: OPEN_UTC,
        max_iterations=3,
    )
    assert ran == 3
    assert calls["n"] == 3
    assert sleeps == [60, 60]  # no sleep after the final tick


def test_loop_skips_when_halted() -> None:
    run_cycle, calls = _runner()
    skips: list = []
    ran = run_scheduler(
        run_cycle=run_cycle,
        is_halted=lambda: True,
        interval_seconds=60,
        sleep_fn=lambda _s: None,
        now_fn=lambda: OPEN_UTC,
        max_iterations=2,
        on_skip=skips.append,
    )
    assert ran == 0
    assert calls["n"] == 0
    assert skips == ["halted", "halted"]


def test_loop_skips_outside_market_hours() -> None:
    run_cycle, calls = _runner()
    skips: list = []
    ran = run_scheduler(
        run_cycle=run_cycle,
        is_halted=lambda: False,
        interval_seconds=60,
        sleep_fn=lambda _s: None,
        now_fn=lambda: CLOSED_UTC,
        max_iterations=2,
        on_skip=skips.append,
    )
    assert ran == 0
    assert skips == ["market closed", "market closed"]


def test_loop_can_ignore_market_hours() -> None:
    run_cycle, calls = _runner()
    ran = run_scheduler(
        run_cycle=run_cycle,
        is_halted=lambda: False,
        interval_seconds=60,
        sleep_fn=lambda _s: None,
        now_fn=lambda: CLOSED_UTC,
        max_iterations=2,
        market_hours_only=False,
    )
    assert ran == 2
    assert calls["n"] == 2


def test_stop_after_close_ends_session_once_market_closes() -> None:
    # Near-continuous loop: open for 2 ticks, then closed. With stop_after_close
    # it runs the 2 in-hours cycles, then exits on the first closed tick instead
    # of skip-spinning to max_iterations.
    run_cycle, calls = _runner()
    times = iter([OPEN_UTC, OPEN_UTC, CLOSED_UTC])
    skips: list = []
    ran = run_scheduler(
        run_cycle=run_cycle,
        is_halted=lambda: False,
        interval_seconds=60,
        sleep_fn=lambda _s: None,
        now_fn=lambda: next(times),
        max_iterations=50,
        on_skip=skips.append,
        stop_after_close=True,
    )
    assert ran == 2
    assert calls["n"] == 2
    assert skips == ["market closed -- session ended"]


def test_stop_after_close_does_not_exit_during_preopen() -> None:
    # Launched pre-open (closed), then market opens, then closes. The pre-open
    # skip must NOT end the session (ran == 0); only a close AFTER running does.
    run_cycle, calls = _runner()
    times = iter([CLOSED_UTC, OPEN_UTC, CLOSED_UTC])
    skips: list = []
    ran = run_scheduler(
        run_cycle=run_cycle,
        is_halted=lambda: False,
        interval_seconds=60,
        sleep_fn=lambda _s: None,
        now_fn=lambda: next(times),
        max_iterations=50,
        on_skip=skips.append,
        stop_after_close=True,
    )
    assert ran == 1
    assert calls["n"] == 1
    assert skips == ["market closed", "market closed -- session ended"]


def test_halt_takes_priority_over_market_hours() -> None:
    run_cycle, calls = _runner()
    skips: list = []
    run_scheduler(
        run_cycle=run_cycle,
        is_halted=lambda: True,
        interval_seconds=60,
        sleep_fn=lambda _s: None,
        now_fn=lambda: OPEN_UTC,
        max_iterations=1,
        on_skip=skips.append,
    )
    assert skips == ["halted"]
