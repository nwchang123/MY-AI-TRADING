"""Read and aggregate runtime state for the web dashboard.

Every read is independent (no shared mutable state) so concurrent requests
from the browser's 5-second polling never race.  Errors in any single read
are caught and surfaced as ``ok: false`` responses.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Simple in-process cache to avoid hammering disk on concurrent requests.
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 1.5  # seconds


def _cached(key: str, ttl: float = _CACHE_TTL) -> Any | None:
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < ttl:
        return entry[1]
    return None


def _store_cache(key: str, value: Any) -> None:
    _cache[key] = (time.monotonic(), value)


def _clear_cache() -> None:
    """Clear in-process cache (used by tests)."""
    _cache.clear()


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _envelope(data: Any, ok: bool = True, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": ok,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if ok:
        result["data"] = data
    else:
        result["error"] = error or "unknown error"
    return result


def get_health() -> dict[str, Any]:
    return _envelope({"status": "running"})


def get_status(root_dir: Path) -> dict[str, Any]:
    cache_key = "status"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    runtime = root_dir / "runtime"
    halt_path = runtime / "HALT"
    heartbeat_path = runtime / "heartbeat.json"

    heartbeat = _read_json(heartbeat_path) if heartbeat_path.exists() else None
    heartbeat_age = None
    heartbeat_stale = False
    if heartbeat and heartbeat.get("at"):
        try:
            hb_time = datetime.fromisoformat(heartbeat["at"])
            age_seconds = (datetime.now(timezone.utc) - hb_time).total_seconds()
            heartbeat_age = round(age_seconds)
            heartbeat_stale = age_seconds > 300  # 5 minutes
        except ValueError:
            heartbeat_stale = True

    is_halted = halt_path.exists()
    halt_reason = None
    if is_halted:
        try:
            halt_reason = halt_path.read_text(encoding="utf-8").strip()
        except OSError:
            halt_reason = "unknown"

    result = _envelope({
        "halted": is_halted,
        "halt_reason": halt_reason,
        "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_stale": heartbeat_stale,
        "heartbeat_note": heartbeat.get("note") if heartbeat else None,
    })
    _store_cache(cache_key, result)
    return result


def get_positions(root_dir: Path, mode: str = "paper") -> dict[str, Any]:
    cache_key = f"positions_{mode}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        from trading_agent.storage.positions import PositionStore

        db_path = root_dir / "runtime" / f"positions.{mode}.sqlite"
        if not db_path.exists():
            result = _envelope({"positions": [], "count": 0})
        else:
            store = PositionStore(db_path)
            try:
                positions = store.open_positions()
            finally:
                store.close()
            result = _envelope({"positions": positions, "count": len(positions)})
    except Exception as exc:
        result = _envelope(None, ok=False, error=str(exc))

    _store_cache(cache_key, result)
    return result


def get_portfolio_risk(root_dir: Path, mode: str = "paper") -> dict[str, Any]:
    cache_key = f"portfolio_risk_{mode}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        from trading_agent.reporting import build_portfolio_risk_report
        from trading_agent.storage.positions import PositionStore

        db_path = root_dir / "runtime" / f"positions.{mode}.sqlite"
        if not db_path.exists():
            result = _envelope({
                "open_positions": 0,
                "premium_at_risk_usd": 0,
                "totals": {"delta": 0, "gamma": 0, "vega": 0, "theta_usd_per_day": 0},
                "shock_pnl_usd": {},
                "scenario_revaluation_pnl_usd": {},
                "positions": [],
            })
        else:
            store = PositionStore(db_path)
            try:
                positions = store.open_positions()
            finally:
                store.close()
            report = build_portfolio_risk_report(positions)
            result = _envelope(report)
    except Exception as exc:
        result = _envelope(None, ok=False, error=str(exc))

    _store_cache(cache_key, result)
    return result


def get_report_today(root_dir: Path) -> dict[str, Any]:
    cache_key = "report_today"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        from trading_agent.reporting import build_daily_report, read_audit_events

        today = datetime.now(timezone.utc).date()
        audit_path = root_dir / "runtime" / "audit.jsonl"
        events = read_audit_events(audit_path, today)
        report = build_daily_report(events, today)
        result = _envelope(report)
    except Exception as exc:
        result = _envelope(None, ok=False, error=str(exc))

    _store_cache(cache_key, result)
    return result


def get_funnel_today(root_dir: Path) -> dict[str, Any]:
    cache_key = "funnel_today"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        from trading_agent.reporting import build_funnel_report, read_audit_events

        today = datetime.now(timezone.utc).date()
        audit_path = root_dir / "runtime" / "audit.jsonl"
        events = read_audit_events(audit_path, today)
        report = build_funnel_report(events, today)
        result = _envelope(report)
    except Exception as exc:
        result = _envelope(None, ok=False, error=str(exc))

    _store_cache(cache_key, result)
    return result


def get_calibration_today(root_dir: Path) -> dict[str, Any]:
    cache_key = "calibration_today"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        from trading_agent.reporting import build_calibration_report, read_audit_events

        today = datetime.now(timezone.utc).date()
        audit_path = root_dir / "runtime" / "audit.jsonl"
        events = read_audit_events(audit_path, today)
        report = build_calibration_report(events, today)
        result = _envelope(report)
    except Exception as exc:
        result = _envelope(None, ok=False, error=str(exc))

    _store_cache(cache_key, result)
    return result


def get_audit_24h(root_dir: Path, limit: int = 200) -> dict[str, Any]:
    cache_key = f"audit_24h_{limit}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        from trading_agent.web_dashboard.audit_tail import read_recent_audit_events

        audit_path = root_dir / "runtime" / "audit.jsonl"
        all_events = read_recent_audit_events(audit_path, limit=limit)
        cutoff = datetime.now(timezone.utc).timestamp() - 86400
        events = []
        for e in all_events:
            recorded = e.get("recorded_at", "")
            try:
                ts = datetime.fromisoformat(recorded).timestamp()
                if ts >= cutoff:
                    events.append(e)
            except ValueError:
                events.append(e)  # include if can't parse
        result = _envelope({"events": events, "count": len(events)})
    except Exception as exc:
        result = _envelope(None, ok=False, error=str(exc))

    _store_cache(cache_key, result)
    return result


def get_audit_recent(root_dir: Path, limit: int = 80) -> dict[str, Any]:
    cache_key = f"audit_recent_{limit}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        from trading_agent.web_dashboard.audit_tail import read_recent_audit_events

        audit_path = root_dir / "runtime" / "audit.jsonl"
        events = read_recent_audit_events(audit_path, limit)
        result = _envelope({"events": events, "count": len(events)})
    except Exception as exc:
        result = _envelope(None, ok=False, error=str(exc))

    _store_cache(cache_key, result)
    return result


def get_iv(root_dir: Path) -> dict[str, Any]:
    cache_key = "iv"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    iv_path = root_dir / "runtime" / "iv30_history.json"
    data = _read_json(iv_path)
    result = _envelope(data or {})
    _store_cache(cache_key, result)
    return result


def get_config_mandate(root_dir: Path, mode: str = "paper") -> dict[str, Any]:
    cache_key = f"mandate_{mode}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    try:
        from trading_agent.domain.risk import Mandate

        mandate = Mandate.load(root_dir / "config" / f"mandate.{mode}.yaml")
        result = _envelope(mandate.model_dump(mode="json"))
    except Exception as exc:
        result = _envelope(None, ok=False, error=str(exc))

    _store_cache(cache_key, result)
    return result


def execute_halt(root_dir: Path, reason: str = "web dashboard halt") -> dict[str, Any]:
    try:
        halt_path = root_dir / "runtime" / "HALT"
        halt_path.parent.mkdir(parents=True, exist_ok=True)
        halt_path.write_text(f"{reason}\n", encoding="utf-8")

        from trading_agent.storage.audit import AuditWriter

        AuditWriter(root_dir / "runtime" / "audit.jsonl").append(
            "kill_switch_activated",
            {"reason": reason, "source": "web_dashboard"},
        )
        _clear_cache()
        return _envelope({"halted": True, "reason": reason})
    except Exception as exc:
        return _envelope(None, ok=False, error=str(exc))


def execute_resume(root_dir: Path) -> dict[str, Any]:
    try:
        from trading_agent.storage.audit import AuditWriter

        halt_path = root_dir / "runtime" / "HALT"
        was_active = halt_path.exists()
        halt_path.unlink(missing_ok=True)
        AuditWriter(root_dir / "runtime" / "audit.jsonl").append(
            "kill_switch_cleared",
            {"was_active": was_active, "source": "web_dashboard"},
        )
        _clear_cache()
        return _envelope({"halted": False, "was_active": was_active})
    except Exception as exc:
        return _envelope(None, ok=False, error=str(exc))
