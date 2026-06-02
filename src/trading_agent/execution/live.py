from __future__ import annotations

# Phase 6 ramp: cap to one open position until the first N live trades are done
# and their execution, fees, and exits have been reviewed (plan section 8).
LIVE_RAMP_TRADES = 10
LIVE_RAMP_POSITION_CAP = 1

LIVE_UNLOCK_CHECKLIST = """\
Live pre-flight checklist (perform manually before each live session):
  1. OpenD GUI is running and logged into the correct Moomoo MY account.
  2. Live trading is unlocked in the OpenD GUI (the agent never unlocks it).
  3. TRADING_AGENT_MODE=live and TRADING_AGENT_ENABLE_LIVE acknowledgment set.
  4. TRADING_AGENT_ACCOUNT_ID is the intended live account and is allowlisted.
  5. runtime/HALT is absent (no active kill switch).
  6. You have reviewed the audit log from the previous session.
The agent caps to one open position for the first 10 live trades; review the
audit log after each of those trades before widening exposure."""


def live_position_cap(
    mandate_max: int,
    live_trades: int,
    *,
    ramp_trades: int = LIVE_RAMP_TRADES,
    ramp_cap: int = LIVE_RAMP_POSITION_CAP,
) -> int:
    """One position until the ramp is complete, then the mandate maximum."""

    if live_trades < ramp_trades:
        return min(ramp_cap, mandate_max)
    return mandate_max
