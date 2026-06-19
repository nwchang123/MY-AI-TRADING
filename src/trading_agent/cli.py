from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from trading_agent.backtest import run_backtest_file
from trading_agent.brokers.moomoo import (
    MoomooBroker,
    MoomooConnection,
    assert_opend_reachable,
)
from trading_agent.data.earnings import YahooEarningsCalendar
from trading_agent.data.earnings_calendar import build_earnings_calendar
from trading_agent.data.iv_history import IV30History
from trading_agent.data.moomoo_market import MoomooMarket
from trading_agent.data.news_feeds import GoogleNewsClient
from trading_agent.data.price_history import YahooPriceHistory
from trading_agent.data.option_data import OptionDataProvider, build_option_provider
from trading_agent.data.sec_edgar import SecEdgarClient
from trading_agent.domain.evidence import CandidateContext
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.proposals import OpenPositionProposal
from trading_agent.domain.risk import Mandate, PortfolioState, QuoteSnapshot, RiskGate
from trading_agent.execution.live import LIVE_UNLOCK_CHECKLIST, live_position_cap
from trading_agent.execution.lock import single_instance_lock
from trading_agent.notify import TelegramNotifier, format_cycle_alert
from trading_agent.execution.orchestrator import PaperTradingCycle
from trading_agent.execution.scheduler import run_scheduler
from trading_agent.reporting import (
    build_calibration_report,
    build_daily_report,
    build_funnel_report,
    read_audit_events,
)
from trading_agent.research.catalysts import build_candidate_context, derive_score_inputs
from trading_agent.research.committee import Committee
from trading_agent.research.universe import (
    DEFAULT_MAX_TICKERS,
    CachedProbe,
    EligibleContractProbe,
    select_universe,
)
from trading_agent.research.llm import OpenAICompatibleClient, usage_delta
from trading_agent.research.scoring import ScoreInputs, score_candidate
from trading_agent.settings import Settings
from trading_agent.storage.audit import AuditWriter
from trading_agent.storage.budget import DailyTokenBudget
from trading_agent.storage.decisions import DecisionCache
from trading_agent.storage.positions import PositionStore
from trading_agent.storage.probes import ProbeCache
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


def _option_provider(settings: Settings) -> OptionDataProvider:
    # Option chains/quotes come from Yahoo Finance (free, ~15min delayed).
    # Moomoo is used for execution only. Real-time stock prices come from Moomoo.
    return build_option_provider(
        settings.option_data_source,
        tradier_token=settings.tradier_token,
        tradier_base_url=settings.tradier_base_url,
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
    mandate = Mandate.load(settings.mandate_path)
    return Committee(
        _llm_client(settings, settings.llm_model),
        _llm_client(settings, settings.llm_model),
        adversary_client=_adversary_client(settings),
        min_win_probability=mandate.options.min_estimated_win_probability,
        account_capital_usd=mandate.account.initial_capital_usd,
        max_contract_cost_usd=mandate.options.max_contract_cost_usd,
        pre_earnings_exit_trading_days=mandate.options.pre_earnings_exit_trading_days,
        earnings_window_max_days=mandate.options.earnings_window_max_days,
        veto_win_prob_penalty=mandate.options.veto_win_prob_penalty,
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
    # Universe screen uses Moomoo's stock filter (entitled); optionability is
    # checked against the free option-data feed (Moomoo option data is not).
    market = _market(settings)
    provider = _option_provider(settings)
    candidates = market.scan_small_caps(mandate.universe)
    optionable = [
        c for c in candidates if c.get("code") and provider.is_optionable(c["code"])
    ]
    _audit_writer(settings).append(
        "universe_scanned",
        {"scanned": len(candidates), "optionable": len(optionable)},
    )
    _print_json(optionable)


def _chain(settings: Settings, ticker: str, option_type: str) -> None:
    """Fetch a ticker's option chain (DTE window from the mandate) for review."""

    mandate = Mandate.load(settings.mandate_path)
    provider = _option_provider(settings)
    today = datetime.now(timezone.utc).date()
    start = today + timedelta(days=mandate.options.min_dte)
    end = today + timedelta(days=mandate.options.max_dte)
    rows = provider.option_chain(ticker, start, end, option_type.upper())
    payload = [
        {
            "code": r["code"],
            "expiry": r["expiry"].isoformat(),
            "side": r["side"],
            "strike": r["strike"],
            "bid": r["bid"],
            "ask": r["ask"],
            "open_interest": r["open_interest"],
            "daily_volume": r["daily_volume"],
            "iv": r["iv"],
        }
        for r in rows
    ]
    _print_json(payload)


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
        market=_option_provider(settings),
        broker=_broker(settings),
        sec_client=SecEdgarClient(settings.sec_user_agent),
        committee=committee,
        gate=RiskGate(mandate, settings.root_dir),
        liquidity=LiquidityValidator(mandate.options, mandate.execution),
        position_store=_position_store(settings),
        audit=_audit_writer(settings),
        trd_env=trd_env,
        max_open_positions_override=max_open_positions_override,
        news_client=GoogleNewsClient(),
        decision_cache=DecisionCache(
            settings.root_dir / "runtime" / f"decisions.{settings.mode}.json"
        ),
        llm_budget=(
            DailyTokenBudget(
                settings.root_dir / "runtime" / f"llm_budget.{settings.mode}.json",
                mandate.execution.max_daily_llm_tokens,
            )
            if mandate.execution.max_daily_llm_tokens > 0
            else None
        ),
        moomoo_market=_market(settings),
        earnings_client=YahooEarningsCalendar(),
        price_history=YahooPriceHistory(),
        earnings_calendar=build_earnings_calendar(settings.finnhub_api_key),
        iv_history=IV30History(settings.root_dir / "runtime" / "iv30_history.json"),
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
    assert_opend_reachable(settings.moomoo_host, settings.moomoo_port)
    with single_instance_lock(_cycle_lock_path(settings)):
        result = _build_cycle(settings, trd_env="SIMULATE").run_once(tickers)
    _alert_cycle(settings, result)
    _print_json(_cycle_payload(result))


def _notifier(settings: Settings) -> TelegramNotifier | None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return None
    return TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)


def _alert_cycle(settings: Settings, result: Any) -> None:
    notifier = _notifier(settings)
    if notifier is None:
        return
    text = format_cycle_alert(result, mode=settings.mode)
    if text:
        notifier.send(text)


def _update_dashboard(settings: Settings, result: Any) -> None:
    """Regenerate the HTML dashboard after each cycle (best-effort)."""
    try:
        from trading_agent.dashboard import update_dashboard
        from trading_agent.storage.positions import PositionStore

        store = PositionStore(settings.root_dir / "runtime" / "positions.sqlite")
        open_pos = [p.__dict__ for p in store.open_positions()]

        from trading_agent.data.iv_history import IV30History
        iv_hist = IV30History(settings.root_dir / "runtime" / "iv30_history.json")
        iv_data = iv_hist._data

        import sqlite3
        audit_path = settings.root_dir / "runtime" / "audit.jsonl"
        audit_summary = {}
        if audit_path.exists():
            for line in audit_path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    d = json.loads(line)
                    et = d.get("event_type", "")
                    audit_summary[et] = audit_summary.get(et, 0) + 1
                except (json.JSONDecodeError, ValueError):
                    pass

        update_dashboard(
            root_dir=settings.root_dir,
            positions=open_pos,
            recent_exits=getattr(result, "exits", []),
            recent_entries=getattr(result, "entries", []),
            iv_history=iv_data,
            audit_summary=audit_summary,
        )
    except Exception:  # noqa: BLE001 - dashboard never blocks trading
        pass


def _write_heartbeat(settings: Settings, note: str) -> None:
    """Liveness marker for the unattended loop (stale file = dead process)."""

    path = settings.root_dir / "runtime" / "heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"at": datetime.now(timezone.utc).isoformat(), "note": note}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _auto_universe(settings: Settings, max_tickers: int) -> list[str]:
    """Let the agent pick its own tickers.

    scan (full, paginated) -> rank by volume ratio with fresh 8-K filers first
    -> skip fresh cached rejections -> cap per industry -> probe for a
    mandate-eligible contract (negative probes cached on disk).
    """

    mandate = Mandate.load(settings.mandate_path)
    opt = mandate.options
    market = _market(settings)
    provider = _option_provider(settings)
    probe = CachedProbe(
        EligibleContractProbe(
            provider=provider,
            options=opt,
            execution=mandate.execution,
            max_entry_iv=opt.max_entry_iv,
        ),
        ProbeCache(settings.root_dir / "runtime" / "probe_cache.json"),
    )
    # Bench rejected names for 2h, not the full 6h decision TTL: the bench
    # only saves a probe + evidence fetch, but it blocks the re-evaluation a
    # changed digest would trigger (and the eligible-name flow is scarce).
    skip = DecisionCache(
        settings.root_dir / "runtime" / f"decisions.{settings.mode}.json"
    ).fresh_rejections(within_hours=2.0)
    try:
        priority = SecEdgarClient(settings.sec_user_agent).recent_8k_tickers()
    except Exception as exc:  # noqa: BLE001 - seeds are an enrichment, never block
        priority = set()
        print(f"auto universe: 8-K seed feed unavailable ({exc})", file=sys.stderr)

    def industry_of(tickers: list[str]) -> dict[str, str]:
        plates = market.industry_plates([f"US.{t}" for t in tickers])
        return {code.removeprefix("US."): name for code, name in plates.items()}

    # Pre-catalyst (earnings IV-ramp) mode: only active when the mandate opens
    # the window. The bulk calendar intersects the scan with names reporting in
    # [today+min, today+max] days; select_universe then ranks farther-earnings
    # (lower IV) first. earnings_of=None keeps the legacy volume-ratio ranking.
    earnings_of: Callable[[list[str]], dict[str, date]] | None = None
    earnings_window: dict[str, str] | None = None
    if opt.earnings_window_max_days > 0:
        calendar = build_earnings_calendar(settings.finnhub_api_key)
        today = datetime.now(timezone.utc).date()
        win_start = today + timedelta(days=opt.earnings_window_min_days)
        win_end = today + timedelta(days=opt.earnings_window_max_days)
        earnings_window = {"start": win_start.isoformat(), "end": win_end.isoformat()}

        def earnings_of(tickers: list[str]) -> dict[str, date]:
            try:
                return calendar.upcoming(tickers, win_start, win_end)
            except Exception as exc:  # noqa: BLE001 - calendar never blocks a cycle
                print(
                    f"auto universe: earnings calendar unavailable ({exc})",
                    file=sys.stderr,
                )
                return {}

    watchlist = {t.strip().upper() for t in mandate.universe.watchlist if t.strip()}
    tickers = select_universe(
        market=market,
        provider=provider,
        universe=mandate.universe,
        max_tickers=max_tickers,
        probe=probe,
        skip_tickers=skip,
        priority_tickers=priority,
        industry_of=industry_of,
        earnings_of=earnings_of,
        watchlist=watchlist,
    )
    _audit_writer(settings).append(
        "universe_selected",
        {
            "tickers": tickers,
            "max_tickers": max_tickers,
            "skipped_fresh_rejections": sorted(skip),
            "event_seeds": sorted(priority & set(tickers)),
            "probe_fetches": probe.fetch_count,
            "probe_cache_hits": probe.cache_hits,
            "earnings_window": earnings_window,
        },
    )
    print(f"auto universe: {', '.join(tickers) or '(none)'}", file=sys.stderr)
    return tickers


def _resolve_tickers(settings: Settings, args: Any) -> list[str]:
    manual = [t.strip().upper() for t in (args.tickers or "").split(",") if t.strip()]
    if args.auto_universe and manual:
        raise SystemExit("Use either --tickers or --auto-universe, not both.")
    if args.auto_universe:
        return _auto_universe(settings, args.max_tickers)
    if not manual:
        raise SystemExit("Provide --tickers or --auto-universe.")
    return manual


def _run_loop(
    settings: Settings,
    tickers_fn: Callable[[], list[str]],
    *,
    interval_seconds: float,
    max_iterations: int | None,
    market_hours_only: bool,
    stop_after_close: bool = False,
) -> None:
    if settings.account_id is None:
        raise RuntimeError(
            "run-loop requires a pinned account: set TRADING_AGENT_ACCOUNT_ID."
        )

    notifier = _notifier(settings)
    totals = {"entries": 0, "exits": 0, "errors": 0, "crashes": 0}
    last_crash: list[str] = [""]

    def run_cycle() -> None:
        # Resolved per tick so an auto universe follows the market day by day.
        # A crashed cycle is alerted and absorbed: one transient failure (OpenD
        # restart, feed outage) must not kill a multi-week unattended run, and
        # the fixed interval below means this cannot become a tight retry loop.
        _write_heartbeat(settings, "cycle start")
        try:
            # Fail fast on a dead gateway: the SDK would otherwise retry the
            # connection forever and silently hang the whole session.
            assert_opend_reachable(settings.moomoo_host, settings.moomoo_port)
            tickers = tickers_fn()
            with single_instance_lock(_cycle_lock_path(settings)):
                result = _build_cycle(settings, trd_env="SIMULATE").run_once(tickers)
        except Exception as exc:  # noqa: BLE001
            _audit_writer(settings).append("cycle_crashed", {"error": str(exc)})
            print(f"cycle crashed: {exc}", file=sys.stderr)
            totals["crashes"] += 1
            # Only alert on a NEW failure: a gateway that stays down all night
            # must not page the operator once per tick.
            if notifier is not None and str(exc) != last_crash[0]:
                mode_label = "模拟盘" if settings.mode == "paper" else "实盘"
                notifier.send(
                    f"【{mode_label}】⚠️ 本周期运行崩溃：{exc}\n"
                    "循环未中断，下个周期自动继续（相同故障不再重复通知）"
                )
            last_crash[0] = str(exc)
            _write_heartbeat(settings, f"cycle crashed: {exc}")
            return
        last_crash[0] = ""
        totals["entries"] += len(result.entries)
        totals["exits"] += len(result.exits)
        totals["errors"] += len(result.errors)
        _alert_cycle(settings, result)
        _update_dashboard(settings, result)
        _write_heartbeat(settings, "cycle done")
        _print_json(_cycle_payload(result))

    def is_halted() -> bool:
        return _halt_path(settings).exists()

    def on_skip(reason: str) -> None:
        _write_heartbeat(settings, f"skipped: {reason}")
        print(f"skip cycle ({reason})", file=sys.stderr)

    if notifier is not None:
        mode_label = "模拟盘" if settings.mode == "paper" else "实盘"
        notifier.send(
            f"【{mode_label}】🟢 自主交易循环已启动\n"
            f"间隔 {interval_seconds:.0f} 秒 | 账户 {settings.account_id}\n"
            f"开仓/平仓/熔断/错误时会通知你，安静周期不打扰"
        )
    ran = run_scheduler(
        run_cycle=run_cycle,
        is_halted=is_halted,
        interval_seconds=interval_seconds,
        sleep_fn=time.sleep,
        now_fn=lambda: datetime.now(timezone.utc),
        max_iterations=max_iterations,
        market_hours_only=market_hours_only,
        on_skip=on_skip,
        stop_after_close=stop_after_close,
    )
    if notifier is not None:
        mode_label = "模拟盘" if settings.mode == "paper" else "实盘"
        notifier.send(
            f"【{mode_label}】🔴 今日交易循环结束，共执行 {ran} 个周期\n"
            f"开仓 {totals['entries']} | 平仓 {totals['exits']} | "
            f"错误 {totals['errors']} | 崩溃 {totals['crashes']}"
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
    _alert_cycle(settings, result)
    _print_json(_cycle_payload(result))


def _backup(settings: Settings) -> None:
    """Copy the ledgers and audit log into a dated backup folder."""

    import shutil

    runtime = settings.root_dir / "runtime"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target = runtime / "backups" / stamp
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for pattern in ("*.sqlite", "audit.jsonl", "heartbeat.json"):
        for source in runtime.glob(pattern):
            shutil.copy2(source, target / source.name)
            copied.append(source.name)
    _print_json({"backup_dir": str(target), "files": sorted(copied)})


def _report(settings: Settings, on_date: date | None) -> None:
    events = read_audit_events(settings.root_dir / "runtime" / "audit.jsonl")
    target = on_date or datetime.now(timezone.utc).date()
    _print_json(build_daily_report(events, target))


def _funnel(settings: Settings, on_date: date | None) -> None:
    events = read_audit_events(settings.root_dir / "runtime" / "audit.jsonl")
    _print_json(build_funnel_report(events, on_date))


def _calibration(settings: Settings, on_date: date | None) -> None:
    events = read_audit_events(settings.root_dir / "runtime" / "audit.jsonl")
    _print_json(build_calibration_report(events, on_date))


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
    chain_parser = subparsers.add_parser(
        "chain",
        help="Fetch a ticker's option chain (mandate DTE window) from the free feed",
    )
    chain_parser.add_argument("--ticker", required=True)
    chain_parser.add_argument(
        "--type", default="ALL", choices=["ALL", "CALL", "PUT", "all", "call", "put"]
    )
    news_parser = subparsers.add_parser(
        "news", help="Fetch recent headlines for a ticker (Google News RSS)"
    )
    news_parser.add_argument("--ticker", required=True)
    subparsers.add_parser(
        "backup", help="Copy ledgers and audit log to runtime/backups/<date>"
    )
    subparsers.add_parser(
        "bot", help="Run the Telegram control bot (commands + read-only AI Q&A)"
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
        "--tickers", default=None, help="Comma-separated tickers to evaluate"
    )
    cycle_parser.add_argument(
        "--auto-universe",
        action="store_true",
        help="Let the agent pick tickers itself (scan + optionability)",
    )
    cycle_parser.add_argument(
        "--max-tickers", type=int, default=DEFAULT_MAX_TICKERS,
        help="Maximum tickers an auto universe returns",
    )
    loop_parser = subparsers.add_parser(
        "run-loop",
        help="Run paper cycles on an interval (HALT- and market-hours-aware)",
    )
    loop_parser.add_argument(
        "--tickers", default=None, help="Comma-separated tickers to evaluate"
    )
    loop_parser.add_argument(
        "--auto-universe",
        action="store_true",
        help="Re-select tickers automatically before every cycle",
    )
    loop_parser.add_argument(
        "--max-tickers", type=int, default=DEFAULT_MAX_TICKERS,
        help="Maximum tickers an auto universe returns",
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
    loop_parser.add_argument(
        "--stop-after-close",
        action="store_true",
        help="For near-continuous loops: exit cleanly once the market closes "
        "(after running), instead of spinning skip-ticks until max-iterations",
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
    funnel_parser = subparsers.add_parser(
        "funnel",
        help="Selection-to-entry funnel with per-stage drop counts from the audit log",
    )
    funnel_parser.add_argument(
        "--date", default=None, help="UTC date YYYY-MM-DD (default: all events)"
    )
    calibration_parser = subparsers.add_parser(
        "calibration",
        help="Predicted win-probability vs realized outcome, bucketed",
    )
    calibration_parser.add_argument(
        "--date", default=None, help="UTC date YYYY-MM-DD (default: all events)"
    )
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
    if args.command == "chain":
        _chain(settings, args.ticker, args.type)
        return
    if args.command == "news":
        items = GoogleNewsClient().fetch_evidence(args.ticker)
        _print_json([i.model_dump(mode="json") for i in items])
        return
    if args.command == "backup":
        _backup(settings)
        return
    if args.command == "bot":
        from trading_agent.bot import run_bot

        run_bot(settings)
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
        _run_cycle(settings, _resolve_tickers(settings, args))
        return
    if args.command == "run-loop":
        if args.auto_universe:
            tickers_fn = lambda: _auto_universe(settings, args.max_tickers)  # noqa: E731
        else:
            static = _resolve_tickers(settings, args)
            tickers_fn = lambda: static  # noqa: E731
        _run_loop(
            settings,
            tickers_fn,
            interval_seconds=args.interval_seconds,
            max_iterations=args.max_iterations,
            market_hours_only=not args.ignore_market_hours,
            stop_after_close=args.stop_after_close,
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
    if args.command == "funnel":
        on_date = date.fromisoformat(args.date) if args.date else None
        _funnel(settings, on_date)
        return
    if args.command == "calibration":
        on_date = date.fromisoformat(args.date) if args.date else None
        _calibration(settings, on_date)
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
