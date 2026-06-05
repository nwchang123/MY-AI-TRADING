from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.backtest import run_backtest_file
from trading_agent.brokers.moomoo import MoomooBroker, MoomooConnection
from trading_agent.data.moomoo_market import MoomooMarket
from trading_agent.data.sec_edgar import SecEdgarClient
from trading_agent.domain.evidence import CandidateContext
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.proposals import OpenPositionProposal
from trading_agent.domain.risk import Mandate, PortfolioState, QuoteSnapshot, RiskGate
from trading_agent.execution.live import LIVE_UNLOCK_CHECKLIST, live_position_cap
from trading_agent.execution.lock import single_instance_lock
from trading_agent.execution.orchestrator import PaperTradingCycle
from trading_agent.execution.scheduler import run_scheduler
from trading_agent.reporting import build_daily_report, read_audit_events
from trading_agent.research.catalysts import build_candidate_context, derive_score_inputs
from trading_agent.research.committee import Committee
from trading_agent.research.llm import OpenAICompatibleClient, usage_delta
from trading_agent.research.scoring import ScoreInputs, score_candidate
from trading_agent.settings import Settings
from trading_agent.storage.audit import AuditWriter
from trading_agent.storage.positions import PositionStore
from trading_agent.storage.sqlite import SnapshotStore


def _broker(settings: Settings) -> MoomooBroker:
    return MoomooBroker(
        MoomooConnection(
            host=settings.moomoo_host,
            port=settings.moomoo_port,
            security_firm=settings.security_firm,
        )
    )


def _market(settings: Settings) -> MoomooMarket:
    return MoomooMarket(
        MoomooConnection(
            host=settings.moomoo_host,
            port=settings.moomoo_port,
            security_firm=settings.security_firm,
        )
    )


def _snapshot_store(settings: Settings) -> SnapshotStore:
    return SnapshotStore(settings.root_dir / "runtime" / "snapshots.sqlite")


def _position_store(settings: Settings) -> PositionStore:
    return PositionStore(
        settings.root_dir / "runtime" / f"positions.{settings.mode}.sqlite"
    )


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str, sort_keys=True))


def _halt_path(settings: Settings) -> Path:
    return settings.root_dir / "runtime" / "HALT"


def _display_path(path: Path, settings: Settings) -> str:
    return path.relative_to(settings.root_dir).as_posix()


def _audit_writer(settings: Settings) -> AuditWriter:
    return AuditWriter(settings.root_dir / "runtime" / "audit.jsonl")


def _check_proposal(settings: Settings, input_path: Path) -> bool:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    proposal = OpenPositionProposal.model_validate(payload["proposal"])
    quote = QuoteSnapshot.model_validate(payload["quote"])
    portfolio = PortfolioState.model_validate(payload["portfolio"])
    now = datetime.fromisoformat(payload["now"]) if payload.get("now") else None
    decision = RiskGate(Mandate.load(settings.mandate_path), settings.root_dir).evaluate_open(
        proposal, quote, portfolio, now
    )
    decision_payload = decision.model_dump(mode="json")
    _audit_writer(settings).append(
        "proposal_checked",
        {
            "option_code": proposal.option_code,
            "decision": decision_payload,
        },
    )
    _print_json(decision_payload)
    return decision.approved


def _llm_client(
    settings: Settings,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key=api_key or settings.llm_api_key,
        base_url=base_url or settings.llm_base_url,
        model=model,
    )


def _adversary_client(settings: Settings) -> OpenAICompatibleClient | None:
    """Optional cross-provider client for the skeptic / risk_manager roles.

    Returns None when no adversary model is configured, so the committee falls
    back to the primary model for those roles. Endpoint and key default to the
    primary ones when only a model is given.
    """
    if not settings.llm_adversary_model:
        return None
    return _llm_client(
        settings,
        settings.llm_adversary_model,
        base_url=settings.llm_adversary_base_url or settings.llm_base_url,
        api_key=settings.llm_adversary_api_key or settings.llm_api_key,
    )


def _build_committee(settings: Settings) -> Committee:
    return Committee(
        _llm_client(settings, settings.llm_model),
        _llm_client(settings, settings.llm_model_pro),
        adversary_client=_adversary_client(settings),
    )


def _run_committee(settings: Settings, input_path: Path) -> bool:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    context = CandidateContext.model_validate(payload["candidate"])
    scores = score_candidate(ScoreInputs.model_validate(payload["score_inputs"]))
    committee = _build_committee(settings)
    before = committee.usage_total()
    output = committee.run(context, scores)
    usage = usage_delta(before, committee.usage_total())
    output_payload = output.model_dump(mode="json")
    output_payload["llm_usage"] = usage.model_dump(mode="json")
    audit = _audit_writer(settings)
    audit.append(
        "committee_run",
        {
            "ticker": context.ticker,
            "flash_model": settings.llm_model,
            "pro_model": settings.llm_model_pro,
            "decision": output.decision,
            "output": output_payload,
        },
    )
    audit.append("llm_usage", {"ticker": context.ticker, **usage.model_dump(mode="json")})
    _print_json(output_payload)
    return output.decision == "open_position"


def _scan(settings: Settings) -> None:
    mandate = Mandate.load(settings.mandate_path)
    market = _market(settings)
    candidates = market.scan_small_caps(mandate.universe)
    optionable = [c for c in candidates if c.get("code") and market.is_optionable(c["code"])]
    _audit_writer(settings).append(
        "universe_scanned",
        {"scanned": len(candidates), "optionable": len(optionable)},
    )
    _print_json(optionable)


def _validate_contract(settings: Settings, input_path: Path) -> bool:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    quote = QuoteSnapshot.model_validate(payload["quote"])
    ticker = str(payload.get("ticker") or quote.option_code)
    contracts = int(payload.get("contracts", 1))
    now = datetime.fromisoformat(payload["now"]) if payload.get("now") else None
    mandate = Mandate.load(settings.mandate_path)
    result = LiquidityValidator(mandate.options, mandate.execution).validate(
        quote, contracts, now
    )
    result_payload = result.model_dump(mode="json")
    _snapshot_store(settings).record_candidate(
        ticker=ticker,
        option_code=quote.option_code,
        passed=result.passed,
        reasons=result.reasons,
        payload=result_payload,
    )
    _audit_writer(settings).append(
        "candidate_validated",
        {"ticker": ticker, "option_code": quote.option_code, "result": result_payload},
    )
    _print_json(result_payload)
    return result.passed


def _collect_catalysts(
    settings: Settings,
    ticker: str,
    forms: list[str] | None,
    limit: int,
    out_path: Path | None,
) -> None:
    as_of = datetime.now(timezone.utc)
    client = SecEdgarClient(settings.sec_user_agent)
    evidence = client.fetch_evidence(ticker, forms=forms, limit=limit)
    context = build_candidate_context(ticker, evidence, as_of)
    score_inputs = derive_score_inputs(context.evidence, as_of)
    bundle = {
        "candidate": context.model_dump(mode="json"),
        "score_inputs": score_inputs.model_dump(mode="json"),
    }
    _audit_writer(settings).append(
        "catalysts_collected",
        {
            "ticker": context.ticker,
            "evidence_count": len(context.evidence),
            "score_inputs": bundle["score_inputs"],
        },
    )
    if out_path is not None:
        out_path.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    _print_json(bundle)


def _build_cycle(
    settings: Settings,
    *,
    trd_env: str,
    max_open_positions_override: int | None = None,
) -> PaperTradingCycle:
    mandate = Mandate.load(settings.mandate_path)
    committee = _build_committee(settings)
    return PaperTradingCycle(
        mandate=mandate,
        account_id=settings.account_id,  # type: ignore[arg-type]
        market=_market(settings),
        broker=_broker(settings),
        sec_client=SecEdgarClient(settings.sec_user_agent),
        committee=committee,
        gate=RiskGate(mandate, settings.root_dir),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=_position_store(settings),
        audit=_audit_writer(settings),
        trd_env=trd_env,
        max_open_positions_override=max_open_positions_override,
    )


def _cycle_payload(result: Any) -> dict[str, Any]:
    return {
        "halted": result.halted,
        "circuit_breaker": result.circuit_breaker,
        "reconciled_closed": result.reconciled_closed,
        "exits": result.exits,
        "entries": result.entries,
        "rejected": result.rejected,
        "errors": result.errors,
    }


def _cycle_lock_path(settings: Settings) -> Path:
    return settings.root_dir / "runtime" / "cycle.lock"


def _run_cycle(settings: Settings, tickers: list[str]) -> None:
    if settings.account_id is None:
        raise RuntimeError(
            "run-cycle requires a pinned account: set TRADING_AGENT_ACCOUNT_ID."
        )
    with single_instance_lock(_cycle_lock_path(settings)):
        result = _build_cycle(settings, trd_env="SIMULATE").run_once(tickers)
    _print_json(_cycle_payload(result))


def _run_loop(
    settings: Settings,
    tickers: list[str],
    *,
    interval_seconds: float,
    max_iterations: int | None,
    market_hours_only: bool,
) -> None:
    if settings.account_id is None:
        raise RuntimeError(
            "run-loop requires a pinned account: set TRADING_AGENT_ACCOUNT_ID."
        )

    def run_cycle() -> None:
        with single_instance_lock(_cycle_lock_path(settings)):
            result = _build_cycle(settings, trd_env="SIMULATE").run_once(tickers)
        _print_json(_cycle_payload(result))

    def is_halted() -> bool:
        return _halt_path(settings).exists()

    ran = run_scheduler(
        run_cycle=run_cycle,
        is_halted=is_halted,
        interval_seconds=interval_seconds,
        sleep_fn=time.sleep,
        now_fn=lambda: datetime.now(timezone.utc),
        max_iterations=max_iterations,
        market_hours_only=market_hours_only,
        on_skip=lambda reason: print(f"skip cycle ({reason})", file=sys.stderr),
    )
    print(f"run-loop finished: {ran} cycle(s) executed", file=sys.stderr)


def _run_live(settings: Settings, tickers: list[str]) -> None:
    if settings.mode != "live":
        raise RuntimeError("run-live requires TRADING_AGENT_MODE=live.")
    # Enforces acknowledgment, pinned account, and the operator allowlist.
    settings.assert_live_startup_allowed()

    mandate = Mandate.load(settings.mandate_path)
    live_trades = len(_position_store(settings).all_positions())
    cap = live_position_cap(mandate.portfolio.max_open_positions, live_trades)

    print(LIVE_UNLOCK_CHECKLIST, file=sys.stderr)
    _audit_writer(settings).append(
        "live_session_start",
        {
            "account_id": settings.account_id,
            "live_trades_so_far": live_trades,
            "position_cap": cap,
            "tickers": tickers,
        },
    )
    with single_instance_lock(_cycle_lock_path(settings)):
        result = _build_cycle(
            settings, trd_env="REAL", max_open_positions_override=cap
        ).run_once(tickers)
    _print_json(_cycle_payload(result))


def _report(settings: Settings, on_date: date | None) -> None:
    events = read_audit_events(settings.root_dir / "runtime" / "audit.jsonl")
    target = on_date or datetime.now(timezone.utc).date()
    _print_json(build_daily_report(events, target))


def _run_backtest(settings: Settings, input_path: Path) -> None:
    mandate = Mandate.load(settings.mandate_path)
    result = run_backtest_file(
        input_path,
        mandate=mandate,
        # Keep offline replays independent from the operator's live/paper HALT
        # file while still exercising RiskGate's kill-switch path.
        root_dir=settings.root_dir / "runtime" / "backtest_sandbox",
    )
    _print_json(result.model_dump(mode="json"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check OpenD connectivity")
    subparsers.add_parser("accounts", help="List Moomoo MY U.S. accounts")
    subparsers.add_parser(
        "scan", help="Screen U.S. small-cap optionable equities via OpenD"
    )
    validate_parser = subparsers.add_parser(
        "validate-contract",
        help="Validate an option quote against the deterministic liquidity rules",
    )
    validate_parser.add_argument("--input", required=True, type=Path)
    catalysts_parser = subparsers.add_parser(
        "catalysts",
        help="Collect SEC EDGAR catalyst evidence and score it for a ticker",
    )
    catalysts_parser.add_argument("--ticker", required=True)
    catalysts_parser.add_argument(
        "--forms",
        default=None,
        help="Comma-separated SEC form filter, e.g. '8-K,S-3'",
    )
    catalysts_parser.add_argument("--limit", type=int, default=20)
    catalysts_parser.add_argument("--out", type=Path, default=None)
    proposal_parser = subparsers.add_parser(
        "proposal-check", help="Evaluate a proposal JSON through the deterministic risk gate"
    )
    proposal_parser.add_argument("--input", required=True, type=Path)
    committee_parser = subparsers.add_parser(
        "committee-run",
        help="Run the LLM research committee over a candidate evidence bundle",
    )
    committee_parser.add_argument("--input", required=True, type=Path)
    cycle_parser = subparsers.add_parser(
        "run-cycle", help="Run one autonomous paper trading cycle (requires OpenD)"
    )
    cycle_parser.add_argument(
        "--tickers", required=True, help="Comma-separated tickers to evaluate"
    )
    loop_parser = subparsers.add_parser(
        "run-loop",
        help="Run paper cycles on an interval (HALT- and market-hours-aware)",
    )
    loop_parser.add_argument(
        "--tickers", required=True, help="Comma-separated tickers to evaluate"
    )
    loop_parser.add_argument(
        "--interval-seconds", type=float, default=900.0, help="Seconds between cycles"
    )
    loop_parser.add_argument(
        "--max-iterations", type=int, default=None, help="Stop after N ticks (default: forever)"
    )
    loop_parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="Run cycles even when the U.S. market is closed",
    )
    live_parser = subparsers.add_parser(
        "run-live",
        help="Run one controlled LIVE cycle (real money; requires explicit setup)",
    )
    live_parser.add_argument(
        "--tickers", required=True, help="Comma-separated tickers to evaluate"
    )
    subparsers.add_parser("positions", help="List locally tracked open positions")
    report_parser = subparsers.add_parser(
        "report", help="Build a daily report from the audit log"
    )
    report_parser.add_argument("--date", default=None, help="UTC date YYYY-MM-DD")
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Replay offline proposals and option quotes through the risk/exit engine",
    )
    backtest_parser.add_argument("--input", required=True, type=Path)
    halt_parser = subparsers.add_parser("halt", help="Activate the local kill switch")
    halt_parser.add_argument("--reason", default="operator halt")
    subparsers.add_parser("resume", help="Clear the local kill switch")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = Settings.from_env()

    if args.command == "doctor":
        _print_json(_broker(settings).doctor())
        return
    if args.command == "accounts":
        _print_json(_broker(settings).list_us_accounts())
        return
    if args.command == "scan":
        _scan(settings)
        return
    if args.command == "validate-contract":
        if not _validate_contract(settings, args.input):
            raise SystemExit(2)
        return
    if args.command == "catalysts":
        forms = (
            [f.strip() for f in args.forms.split(",") if f.strip()]
            if args.forms
            else None
        )
        _collect_catalysts(settings, args.ticker, forms, args.limit, args.out)
        return
    if args.command == "proposal-check":
        if not _check_proposal(settings, args.input):
            raise SystemExit(2)
        return
    if args.command == "committee-run":
        if not _run_committee(settings, args.input):
            raise SystemExit(2)
        return
    if args.command == "run-cycle":
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        _run_cycle(settings, tickers)
        return
    if args.command == "run-loop":
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        _run_loop(
            settings,
            tickers,
            interval_seconds=args.interval_seconds,
            max_iterations=args.max_iterations,
            market_hours_only=not args.ignore_market_hours,
        )
        return
    if args.command == "run-live":
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        _run_live(settings, tickers)
        return
    if args.command == "positions":
        _print_json(_position_store(settings).open_positions())
        return
    if args.command == "report":
        on_date = date.fromisoformat(args.date) if args.date else None
        _report(settings, on_date)
        return
    if args.command == "backtest":
        _run_backtest(settings, args.input)
        return
    if args.command == "halt":
        halt_path = _halt_path(settings)
        halt_path.parent.mkdir(parents=True, exist_ok=True)
        reason = " ".join(args.reason.split()) or "operator halt"
        halt_path.write_text(f"{reason}\n", encoding="utf-8")
        _audit_writer(settings).append("kill_switch_activated", {"reason": reason})
        print(f"Kill switch active: {_display_path(halt_path, settings)}")
        return
    if args.command == "resume":
        halt_path = _halt_path(settings)
        was_active = halt_path.exists()
        halt_path.unlink(missing_ok=True)
        _audit_writer(settings).append(
            "kill_switch_cleared", {"was_active": was_active}
        )
        print(f"Kill switch cleared: {_display_path(halt_path, settings)}")
        return

    raise RuntimeError(f"Unsupported command: {args.command}")
