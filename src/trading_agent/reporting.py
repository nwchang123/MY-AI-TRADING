from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.data.moomoo_market import parse_us_option_code
from trading_agent.domain.calendar import market_date, parse_iso
from trading_agent.domain.montecarlo import black_scholes_price

_UNDERLYING_SHOCKS = {
    "underlying_-5pct": -0.05,
    "underlying_-2pct": -0.02,
    "underlying_+2pct": 0.02,
    "underlying_+5pct": 0.05,
}


def read_audit_events(path: Path, on_date: date | None = None) -> list[dict[str, Any]]:
    """Load append-only audit JSONL records; missing file yields no events.

    When ``on_date`` is supplied, filtering happens while streaming the file so
    daily reports do not need to materialize the full audit log first.
    """

    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if on_date is not None and _event_date(event) != on_date:
                continue
            events.append(event)
    return events


def _event_date(event: dict[str, Any]) -> date | None:
    raw = event.get("recorded_at")
    if not raw:
        return None
    try:
        return market_date(parse_iso(str(raw)))
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _int(value: Any) -> int:
    return int(value) if isinstance(value, int) else 0


def _events_on(events: list[dict[str, Any]], on_date: date | None) -> list[dict[str, Any]]:
    if on_date is None:
        return events
    return [e for e in events if _event_date(e) == on_date]


def build_portfolio_risk_report(
    open_positions: list[dict[str, Any]], *, as_of: date | None = None
) -> dict[str, Any]:
    """Portfolio-level Greeks and underlying-shock P/L from the ledger."""

    rows = [row for row in open_positions if row.get("status", "open") == "open"]
    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta_usd_per_day": 0.0}
    premium = 0.0
    positions: list[dict[str, Any]] = []
    missing_greeks = 0

    for row in rows:
        contracts = _int(row.get("contracts"))
        lot_size = _int(row.get("lot_size"))
        multiplier = contracts * lot_size
        premium_at_risk = _num(row.get("entry_price")) * multiplier
        premium += premium_at_risk
        exposures = {
            "delta": _num(row.get("entry_delta")) * multiplier,
            "gamma": _num(row.get("entry_gamma")) * multiplier,
            "vega": _num(row.get("entry_vega")) * multiplier,
            "theta_usd_per_day": _num(row.get("entry_theta")) * multiplier,
        }
        if any(row.get(key) is None for key in ("entry_delta", "entry_gamma", "entry_vega", "entry_theta")):
            missing_greeks += 1
        for key, value in exposures.items():
            totals[key] += value
        positions.append(
            {
                "ticker": row.get("ticker"),
                "option_code": row.get("option_code"),
                "premium_at_risk_usd": round(premium_at_risk, 4),
                "delta": round(exposures["delta"], 4),
                "gamma": round(exposures["gamma"], 4),
                "vega": round(exposures["vega"], 4),
                "theta_usd_per_day": round(exposures["theta_usd_per_day"], 4),
                "theta_decay_pct_per_day": row.get("entry_theta_decay_pct_per_day"),
                "iv_rank": row.get("entry_iv_rank"),
                "entry_spot": row.get("entry_spot"),
            }
        )

    shock_pnl = {
        label: round(_shock_pnl(rows, move), 4)
        for label, move in _UNDERLYING_SHOCKS.items()
    }
    reval_inputs, reval_skipped = _scenario_revaluation_inputs(
        rows, as_of or datetime.now(timezone.utc).date()
    )
    scenario_revaluation_pnl = {
        label: round(_scenario_revaluation_pnl(reval_inputs, move), 4)
        for label, move in _UNDERLYING_SHOCKS.items()
    }
    return {
        "open_positions": len(rows),
        "premium_at_risk_usd": round(premium, 4),
        "totals": {key: round(value, 4) for key, value in totals.items()},
        "theta_decay_usd_per_day": round(abs(min(totals["theta_usd_per_day"], 0.0)), 4),
        "shock_pnl_usd": shock_pnl,
        "scenario_revaluation_pnl_usd": scenario_revaluation_pnl,
        "scenario_revaluation_skipped": reval_skipped,
        "positions": positions,
        "missing_greeks": missing_greeks,
    }


def _shock_pnl(rows: list[dict[str, Any]], move_pct: float) -> float:
    total = 0.0
    for row in rows:
        spot = _num(row.get("entry_spot"))
        if spot <= 0:
            continue
        contracts = _int(row.get("contracts"))
        lot_size = _int(row.get("lot_size"))
        multiplier = contracts * lot_size
        delta = _num(row.get("entry_delta")) * multiplier
        gamma = _num(row.get("entry_gamma")) * multiplier
        move = spot * move_pct
        total += delta * move + 0.5 * gamma * move * move
    return total


def _scenario_revaluation_inputs(
    rows: list[dict[str, Any]], as_of: date
) -> tuple[list[tuple[str, float, float, float, float, int]], int]:
    inputs: list[tuple[str, float, float, float, float, int]] = []
    skipped = 0
    for row in rows:
        try:
            option_code = str(row.get("option_code") or "")
            _, expiry, side, strike = parse_us_option_code(option_code)
        except ValueError:
            skipped += 1
            continue
        spot = _num(row.get("entry_spot"))
        iv = _num(row.get("entry_iv"))
        multiplier = _int(row.get("contracts")) * _int(row.get("lot_size"))
        if spot <= 0 or iv <= 0 or strike <= 0 or multiplier <= 0:
            skipped += 1
            continue
        t_years = max((expiry - as_of).days, 0) / 365.0
        inputs.append((side, spot, strike, t_years, iv, multiplier))
    return inputs, skipped


def _scenario_revaluation_pnl(
    inputs: list[tuple[str, float, float, float, float, int]], move_pct: float
) -> float:
    total = 0.0
    for side, spot, strike, t_years, iv, multiplier in inputs:
        base = black_scholes_price(
            side=side,
            spot=spot,
            strike=strike,
            t_years=t_years,
            iv=iv,
        )
        shocked = black_scholes_price(
            side=side,
            spot=max(0.01, spot * (1.0 + move_pct)),
            strike=strike,
            t_years=t_years,
            iv=iv,
        )
        total += (shocked - base) * multiplier
    return total


def build_funnel_report(
    events: list[dict[str, Any]], on_date: date | None = None
) -> dict[str, Any]:
    """Selection-to-entry funnel with the drop reason at each stage.

    Answers the question the daily report cannot: of everything the scanner
    surfaced, where did candidates die? Without this the only way to tune a
    threshold is guesswork; with it every stage shows its own attrition. Spans
    all events unless ``on_date`` narrows it to one UTC day.
    """

    scoped = _events_on(events, on_date)

    selected = 0
    probe_fetches = 0
    probe_cache_hits = 0
    event_seeds = 0
    no_eligible_contracts = 0
    reached_committee = 0
    cache_served = 0
    committee_decisions: dict[str, int] = {"open_position": 0, "hold": 0, "reject": 0}
    mc_checked = 0
    mc_rejected = 0
    gate = {"approved": 0, "rejected": 0}
    liquidity = {"passed": 0, "failed": 0}
    orders_filled = 0

    for event in scoped:
        etype = event.get("event_type", "")
        payload = event.get("payload", {})
        if etype == "universe_selected":
            selected += len(payload.get("tickers") or [])
            probe_fetches += int(payload.get("probe_fetches") or 0)
            probe_cache_hits += int(payload.get("probe_cache_hits") or 0)
            event_seeds += len(payload.get("event_seeds") or [])
        elif etype == "no_eligible_contracts":
            no_eligible_contracts += 1
        elif etype == "committee_run":
            reached_committee += 1
            decision = payload.get("decision")
            if decision in committee_decisions:
                committee_decisions[decision] += 1
        elif etype == "committee_cache_hit":
            cache_served += 1
        elif etype == "monte_carlo_pop":
            mc_checked += 1
            pop, floor = payload.get("pop"), payload.get("floor")
            if isinstance(pop, (int, float)) and isinstance(floor, (int, float)) and pop < floor:
                mc_rejected += 1
        elif etype == "candidate_validated":
            passed = (payload.get("result") or {}).get("passed")
            liquidity["passed" if passed else "failed"] += 1
        elif etype == "proposal_checked":
            approved = (payload.get("decision") or {}).get("approved")
            gate["approved" if approved else "rejected"] += 1
        elif etype == "order_filled" and payload.get("side") == "buy":
            orders_filled += 1

    return {
        "scope": on_date.isoformat() if on_date else "all",
        "universe_selected": selected,
        "event_seeds": event_seeds,
        "probe_fetches": probe_fetches,
        "probe_cache_hits": probe_cache_hits,
        "dropped_no_eligible_contract": no_eligible_contracts,
        "reached_committee": reached_committee,
        "served_from_cache": cache_served,
        "committee_decisions": committee_decisions,
        "monte_carlo_checked": mc_checked,
        "monte_carlo_rejected": mc_rejected,
        "liquidity_at_entry": liquidity,
        "risk_gate": gate,
        "orders_filled": orders_filled,
    }


# Predicted-win-probability buckets for the calibration report.
_CALIBRATION_BUCKETS = [(0.0, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01)]


def build_calibration_report(
    events: list[dict[str, Any]], on_date: date | None = None
) -> dict[str, Any]:
    """Predicted win probability vs realized outcome, bucketed.

    Joins each opened position's committee win_probability to whether the trade
    actually closed green, so after the paper run the score weights and the
    win-probability floor can be tuned against evidence instead of intuition.
    Empty buckets until trades close -- the tool is built now so the data is
    ready to read later.
    """

    scoped = _events_on(events, on_date)

    predicted: dict[str, float] = {}
    for event in scoped:
        if event.get("event_type") != "committee_run":
            continue
        output = (event.get("payload") or {}).get("output") or {}
        proposal = output.get("proposal") or {}
        code = proposal.get("option_code")
        win = output.get("win_probability")
        if code and isinstance(win, (int, float)):
            predicted[code] = float(win)

    buckets = [
        {"range": f"{lo:.2f}-{hi:.2f}", "trades": 0, "wins": 0, "predicted_sum": 0.0}
        for lo, hi in _CALIBRATION_BUCKETS
    ]
    matched = 0
    for event in scoped:
        if event.get("event_type") != "position_closed":
            continue
        payload = event.get("payload") or {}
        code = payload.get("option_code")
        pnl = payload.get("realized_pnl_usd")
        if code not in predicted or not isinstance(pnl, (int, float)):
            continue
        win_prob = predicted[code]
        for (lo, hi), bucket in zip(_CALIBRATION_BUCKETS, buckets):
            if lo <= win_prob < hi:
                bucket["trades"] += 1
                bucket["wins"] += 1 if pnl > 0 else 0
                bucket["predicted_sum"] += win_prob
                matched += 1
                break

    for bucket in buckets:
        trades = bucket["trades"]
        predicted_sum = bucket.pop("predicted_sum")  # internal accumulator
        bucket["predicted_avg"] = round(predicted_sum / trades, 4) if trades else None
        bucket["actual_win_rate"] = round(bucket["wins"] / trades, 4) if trades else None

    return {
        "scope": on_date.isoformat() if on_date else "all",
        "proposals_with_estimate": len(predicted),
        "closed_trades_matched": matched,
        "buckets": buckets,
    }


def build_daily_report(events: list[dict[str, Any]], on_date: date) -> dict[str, Any]:
    """Aggregate one U.S. market date of audit events into a report.

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
    llm = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for event in day_events:
        etype = event.get("event_type", "")
        payload = event.get("payload", {})

        if etype == "llm_usage":
            for key in ("calls", "prompt_tokens", "completion_tokens"):
                value = payload.get(key)
                if isinstance(value, int):
                    llm[key] += value
            total = payload.get("total_tokens")
            if isinstance(total, int):
                llm["total_tokens"] += total
            else:
                llm["total_tokens"] += int(payload.get("prompt_tokens") or 0) + int(
                    payload.get("completion_tokens") or 0
                )
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
        "llm_usage": llm,
    }
