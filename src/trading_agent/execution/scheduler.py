from __future__ import annotations

from datetime import datetime
from typing import Callable

from trading_agent.domain.calendar import is_market_hours


def run_scheduler(
    *,
    run_cycle: Callable[[], object],
    is_halted: Callable[[], bool],
    interval_seconds: float,
    sleep_fn: Callable[[float], None],
    now_fn: Callable[[], datetime],
    max_iterations: int | None = None,
    market_hours_only: bool = True,
    on_skip: Callable[[str], None] | None = None,
) -> int:
    """Drive ``run_cycle`` on a fixed interval, autonomously.

    Each tick: if the kill switch is active the cycle is skipped; if
    ``market_hours_only`` and the U.S. session is closed it is skipped; otherwise
    the cycle runs. Either way the loop sleeps ``interval_seconds`` before the
    next tick (no sleep after the final iteration). ``max_iterations=None`` runs
    forever. All side-effecting dependencies are injected so the control flow is
    unit-testable without a real clock, broker, or sleep.

    Returns the number of cycles actually executed.
    """

    ran = 0
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1

        if is_halted():
            _note(on_skip, "halted")
        elif market_hours_only and not is_market_hours(now_fn()):
            _note(on_skip, "market closed")
        else:
            run_cycle()
            ran += 1

        if max_iterations is None or iteration < max_iterations:
            sleep_fn(interval_seconds)

    return ran


def _note(on_skip: Callable[[str], None] | None, reason: str) -> None:
    if on_skip is not None:
        on_skip(reason)
