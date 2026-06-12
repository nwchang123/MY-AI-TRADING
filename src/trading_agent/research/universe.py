from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

from trading_agent.data.moomoo_market import US_OPTION_LOT_SIZE
from trading_agent.domain.liquidity import LiquidityValidator
from trading_agent.domain.risk import (
    ExecutionMandate,
    OptionsMandate,
    QuoteSnapshot,
    UniverseMandate,
)
from trading_agent.storage.probes import (
    PROBE_ELIGIBLE,
    PROBE_ERROR,
    PROBE_NO_CHAIN,
    PROBE_NO_CONTRACT,
    ProbeCache,
)

# Each optionability probe downloads a full option chain from the data feed, so
# only the highest-ranked scan hits are probed.
#
# Breadth over frequency: most scan hits have NO mandate-eligible contract
# (cheap enough, tight spread, OI/volume), so a wide per-cycle sweep finds far
# more opportunities than re-evaluating the same short list more often (which
# the decision cache would just serve unchanged). Names with no tradeable
# contract are dropped by the probe -- only genuinely tradeable ones reach the
# LLM committee -- so a wider sweep is nearly free.
#
# Ranking is by VOLUME_RATIO (today's volume vs the name's own recent average),
# not raw turnover: turnover ranks are sticky day to day, while an elevated
# volume ratio marks names where something is happening NOW -- the catalyst
# signal this strategy actually trades. Turnover stays as the tiebreak so a
# scan without volume-ratio data degrades to the old behavior.
DEFAULT_MAX_TICKERS = 25
DEFAULT_PROBE_LIMIT = 60
# Hot small-cap themes (one day it's all miners, the next all biotech) would
# otherwise fill every slot with correlated names the 2-position portfolio can
# never use; capping per industry keeps the committee's tokens spread out.
DEFAULT_MAX_PER_INDUSTRY = 5
# Wash-trade / data-artifact guard: a volume ratio above WASH_VOLUME_RATIO with
# an absolute day change below FLAT_CHANGE_PCT (%) is a halt-resumption or
# thin-float print topping the volume-ratio sort, not a tradeable catalyst.
# Live 2026-06-12 the literal #1 was vol_ratio 2718 at 0.0% change.
WASH_VOLUME_RATIO = 100.0
FLAT_CHANGE_PCT = 0.5


class EligibleContractProbe:
    """Does a ticker have >=1 contract passing the cycle's own liquidity rules?

    The plain ``is_optionable`` probe wastes universe slots: it downloads the
    full chain just to confirm the chain exists, while most optionable names
    still fail the mandate downstream (cost cap, spread, OI/volume, DTE). This
    probe runs the SAME LiquidityValidator the cycle applies later over the
    same chain it already fetched, so every selected name is genuinely
    tradeable -- at no extra request cost.

    ``status`` is tri-state-plus-error so ProbeCache can apply per-status TTLs.
    """

    def __init__(
        self,
        *,
        provider: Any,
        options: OptionsMandate,
        execution: ExecutionMandate,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.provider = provider
        self.options = options
        self.validator = LiquidityValidator(options, execution)
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def status(self, ticker: str) -> str:
        now = self._now_fn()
        today = now.date()
        start = today + timedelta(days=self.options.min_dte)
        end = today + timedelta(days=self.options.max_dte)
        try:
            if not self.provider.is_optionable(ticker):
                return PROBE_NO_CHAIN
            chain = self.provider.option_chain(ticker, start, end, "ALL")
        except Exception:  # noqa: BLE001 - feed hiccup: skip now, never cache
            return PROBE_ERROR
        for row in chain:
            ask = float(row.get("ask") or 0.0)
            if ask <= 0:
                continue
            quote = QuoteSnapshot(
                option_code=row["code"],
                bid=float(row.get("bid") or 0.0),
                ask=ask,
                open_interest=int(row.get("open_interest") or 0),
                daily_volume=int(row.get("daily_volume") or 0),
                lot_size=US_OPTION_LOT_SIZE,
                expiry=row["expiry"],
                observed_at=now,
                is_delayed=True,
            )
            verdict = self.validator.validate(
                quote, self.options.contracts_per_order, now
            )
            if verdict.passed:
                return PROBE_ELIGIBLE
        return PROBE_NO_CONTRACT

    def __call__(self, ticker: str) -> bool:
        return self.status(ticker) == PROBE_ELIGIBLE


class CachedProbe:
    """Probe wrapper that serves fresh cached negatives instead of re-fetching.

    Re-probing names that had no chain (or nothing tradeable) hours ago is the
    bulk of the probe budget; skipping them lets the same probe_limit reach
    much deeper into the ranked scan. Positives and errors pass through
    uncached by ProbeCache design.

    ``fetch_count`` exposes how many calls actually hit the data feed:
    select_universe charges only those against probe_limit. Counting cache
    hits too froze selection live on 2026-06-11: after one cycle cached the
    top-60 as negative, every later cycle burned its whole budget on cache
    hits and returned an empty universe for six hours.
    """

    def __init__(self, probe: EligibleContractProbe, cache: ProbeCache):
        self.probe = probe
        self.cache = cache
        self.fetch_count = 0
        self.cache_hits = 0

    def __call__(self, ticker: str) -> bool:
        cached = self.cache.get(ticker)
        if cached is not None:
            self.cache_hits += 1
            return cached == PROBE_ELIGIBLE
        self.fetch_count += 1
        status = self.probe.status(ticker)
        self.cache.put(ticker, status)
        return status == PROBE_ELIGIBLE


def select_universe(
    *,
    market: Any,
    provider: Any,
    universe: UniverseMandate,
    max_tickers: int = DEFAULT_MAX_TICKERS,
    probe_limit: int = DEFAULT_PROBE_LIMIT,
    probe: Callable[[str], bool] | None = None,
    skip_tickers: Iterable[str] = (),
    priority_tickers: Iterable[str] = (),
    industry_of: Callable[[list[str]], dict[str, str]] | None = None,
    max_per_industry: int = DEFAULT_MAX_PER_INDUSTRY,
) -> list[str]:
    """Autonomously pick today's candidate tickers.

    Screens U.S. small caps through the Moomoo stock filter (price, market cap,
    turnover from the mandate), ranks by volume ratio descending (turnover as
    tiebreak) with ``priority_tickers`` (fresh event filers) jumping the queue,
    then keeps the first ``max_tickers`` names that pass ``probe`` (defaults to
    a bare optionability check). ``skip_tickers`` (e.g. names with a fresh
    cached committee rejection) never occupy a slot, no industry takes more
    than ``max_per_industry`` slots, and industries in the mandate's
    ``excluded_industries`` (SPAC shells by default) are never probed at all.
    Returns bare ticker symbols (no ``US.`` prefix), ready for the trading
    cycle.

    ``probe_limit`` charges only probes that actually hit the data feed: a
    probe exposing ``fetch_count`` (CachedProbe) gets its cache hits for free,
    so every cycle digs deeper into the ranked scan instead of re-spending the
    budget on names already known to be dead.
    """

    if max_tickers <= 0:
        return []
    if probe is None:
        probe = provider.is_optionable
    skip = {str(t).strip().upper() for t in skip_tickers}
    priority = {str(t).strip().upper() for t in priority_tickers}
    excluded_industries = {
        name.strip().lower() for name in getattr(universe, "excluded_industries", [])
    }

    rows = market.scan_small_caps(universe)
    normalized: list[tuple[str, float, float]] = []
    for row in rows:
        code = str(row.get("code") or "")
        ticker = code.removeprefix("US.").strip().upper()
        # Option roots are plain alphabetic tickers; skip warrants/units/etc.
        if not ticker or not ticker.isalpha():
            continue
        volume_ratio = float(row.get("volume_ratio") or 0.0)
        change_rate = float(row.get("change_rate") or 0.0)
        # Wash-trade / artifact guard: a volume ratio in the hundreds with no
        # price move at all is a halt-resumption or thin-float print, not a
        # catalyst. Drop only the degenerate extreme so genuine pre-move
        # accumulation (elevated ratio, small but real move) still ranks.
        if volume_ratio > WASH_VOLUME_RATIO and abs(change_rate) < FLAT_CHANGE_PCT:
            continue
        normalized.append(
            (
                ticker,
                volume_ratio,
                float(row.get("turnover") or 0.0),
            )
        )
    normalized.sort(
        key=lambda c: (c[0] in priority, c[1], c[2]), reverse=True
    )

    ranked: list[str] = []
    seen: set[str] = set()
    for ticker, _, _ in normalized:
        if ticker in seen or ticker in skip:
            continue
        seen.add(ticker)
        ranked.append(ticker)

    industries: dict[str, str] = {}
    if industry_of is not None and (max_per_industry > 0 or excluded_industries):
        # 200 = get_owner_plate's per-call code cap. With cache hits free, a
        # cycle can walk far past probe_limit names, so look up as many as one
        # batched call allows; deeper names simply go uncapped/unexcluded.
        try:
            industries = {
                str(t).strip().upper(): name
                for t, name in (industry_of(ranked[:200]) or {}).items()
            }
        except Exception:  # noqa: BLE001 - diversity is best-effort, never blocks
            industries = {}

    selected: list[str] = []
    per_industry: Counter[str] = Counter()
    fetches_locally = 0
    fetch_baseline = getattr(probe, "fetch_count", 0)
    for ticker in ranked:
        # CachedProbe reports real feed fetches; a plain probe charges every call.
        spent = getattr(probe, "fetch_count", fetch_baseline + fetches_locally)
        if spent - fetch_baseline >= probe_limit:
            break
        industry = industries.get(ticker)
        if industry is not None:
            if industry.strip().lower() in excluded_industries:
                continue
            if max_per_industry > 0 and per_industry[industry] >= max_per_industry:
                continue
        fetches_locally += 1
        if not probe(ticker):
            continue
        selected.append(ticker)
        if industry is not None:
            per_industry[industry] += 1
        if len(selected) >= max_tickers:
            break
    return selected
