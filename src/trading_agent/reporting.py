from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


def read_audit_events(path: Path) -> list[dict[str, Any]]:
    """Load append-only audit JSONL records; missing file yields no events."""

    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _event_date(event: dict[str, Any]) -> date | None:
    raw = event.get("recorded_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def build_daily_report(events: list[dict[str, Any]], on_date: date) -> dict[str, Any]:
    """Aggregate one UTC day of audit events into a report.

    Counts committee decisions, risk-gate verdicts, validated contracts, orders,
    closed positions with realized P/L, and failures — the evidence behind a
    paper shadow session (plan section 8, Phase 5).
    """

    day_events = [e for e in events if _event_date(e) == on_date]

    committee_decisions: dict[str, int] = {"open_position": 0, "hold": 0, "reject": 0}
    gate = {"approved": 0, "rejected": 0}
    contracts = {"passed": 0, "failed": 0}
    spreads: list[float] = []
    orders = {"placed": 0, "filled": 0, "cancelled": 0}
    close_reasons: dict[str, int] = {}
    realized_pnl = 0.0
    positions_closed = 0
    failures = 0
    llm = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    for event in day_events:
        etype = event.get("event_type", "")
        payload = event.get("payload", {})

        if etype == "llm_usage":
            for key in llm:
                value = payload.get(key)
                if isinstance(value, int):
                    llm[key] += value
        elif etype == "committee_run":
            decision = payload.get("decision")
            if decision in committee_decisions:
                committee_decisions[decision] += 1
        elif etype == "proposal_checked":
            approved = payload.get("decision", {}).get("approved")
            gate["approved" if approved else "rejected"] += 1
        elif etype == "candidate_validated":
            result = payload.get("result", {})
            contracts["passed" if result.get("passed") else "failed"] += 1
            if isinstance(result.get("spread_pct"), (int, float)):
                spreads.append(float(result["spread_pct"]))
        elif etype == "order_placed":
            orders["placed"] += 1
        elif etype == "order_filled":
            orders["filled"] += 1
        elif etype == "order_cancelled":
            orders["cancelled"] += 1
        elif etype == "position_closed":
            positions_closed += 1
            reason = payload.get("reason", "unknown")
            close_reasons[reason] = close_reasons.get(reason, 0) + 1
            pnl = payload.get("realized_pnl_usd")
            if isinstance(pnl, (int, float)):
                realized_pnl += float(pnl)
        elif etype.endswith("_failed") or etype.endswith("_error"):
            failures += 1

    return {
        "date": on_date.isoformat(),
        "events": len(day_events),
        "proposals_processed": committee_decisions["open_position"]
        + committee_decisions["hold"]
        + committee_decisions["reject"],
        "committee_decisions": committee_decisions,
        "gate": gate,
        "contracts": contracts,
        "avg_spread_pct": _mean(spreads),
        "orders": orders,
        "positions_closed": positions_closed,
        "close_reasons": close_reasons,
        "realized_pnl_usd": round(realized_pnl, 4),
        "failures": failures,
        "llm_usage": {**llm, "total_tokens": llm["prompt_tokens"] + llm["completion_tokens"]},
    }
