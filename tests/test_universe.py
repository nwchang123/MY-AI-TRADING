from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from trading_agent.domain.risk import Mandate
from trading_agent.research.universe import (
    CachedProbe,
    EligibleContractProbe,
    select_universe,
)
from trading_agent.storage.probes import (
    PROBE_ELIGIBLE,
    PROBE_ERROR,
    PROBE_NO_CHAIN,
    PROBE_NO_CONTRACT,
    ProbeCache,
)

NOW = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)


def _mandate() -> Mandate:
    # Pin the canonical $100 cost cap so the probe-eligibility fixtures (costs
    # calibrated to the $65 cap) stay stable when the live paper mandate's
    # capital/caps change (e.g. raised to a $500 base).
    mandate = Mandate.load(Path("config/mandate.paper.yaml"))
    options = mandate.options.model_copy(update={"max_contract_cost_usd": 65.0})
    return mandate.model_copy(update={"options": options})


def _universe():
    return _mandate().universe


class FakeScanMarket:
    def __init__(self, rows):
        self.rows = rows

    def scan_small_caps(self, universe):
        return self.rows


class FakeProvider:
    def __init__(self, optionable: set[str]):
        self.optionable = optionable
        self.probed: list[str] = []

    def is_optionable(self, ticker: str) -> bool:
        self.probed.append(ticker)
        return ticker in self.optionable


def _row(
    code: str, turnover: float, volume_ratio: float = 0.0, change_rate: float = 1.0
) -> dict:
    return {
        "code": code,
        "name": code,
        "cur_price": 5.0,
        "turnover": turnover,
        "volume_ratio": volume_ratio,
        "change_rate": change_rate,
    }


def test_select_universe_drops_wash_trade_artifacts() -> None:
    # Extreme volume ratio with no price move at all is a halt-resumption/thin
    # print artifact, not a catalyst -> dropped before probing. A genuine
    # pre-move (elevated ratio, small but real change) survives.
    market = FakeScanMarket(
        [
            _row("US.WASH", 5e7, volume_ratio=2718.0, change_rate=0.0),
            _row("US.REAL", 4e7, volume_ratio=40.0, change_rate=0.3),
        ]
    )
    provider = FakeProvider(optionable={"WASH", "REAL"})

    picked = select_universe(
        market=market, provider=provider, universe=_universe(), max_tickers=5
    )

    assert picked == ["REAL"]
    assert "WASH" not in provider.probed


def test_select_universe_ranks_by_volume_ratio_over_turnover() -> None:
    # HOT has modest turnover but the highest relative volume: it must outrank
    # the perpetually liquid BIG (volume_ratio is the catalyst signal).
    market = FakeScanMarket(
        [
            _row("US.BIG", 9e7, volume_ratio=1.0),
            _row("US.HOT", 2e7, volume_ratio=6.0),
            _row("US.MID", 5e7, volume_ratio=2.5),
        ]
    )
    provider = FakeProvider(optionable={"BIG", "HOT", "MID"})

    picked = select_universe(
        market=market, provider=provider, universe=_universe(), max_tickers=2
    )

    assert picked == ["HOT", "MID"]
    assert provider.probed == ["HOT", "MID"]


def test_select_universe_premove_penalty_prefers_accumulation() -> None:
    # Two names with the SAME volume ratio: MOVED has already run +20% today,
    # CALM has barely moved. With the pre-move penalty OFF, the turnover tiebreak
    # puts MOVED first; with it ON, MOVED's big move halves its rank score so the
    # accumulation name (CALM, catalyst not yet priced in) is selected first.
    def rows():
        return [
            _row("US.MOVED", 9e7, volume_ratio=10.0, change_rate=20.0),
            _row("US.CALM", 5e7, volume_ratio=10.0, change_rate=0.5),
        ]

    off = select_universe(
        market=FakeScanMarket(rows()),
        provider=FakeProvider(optionable={"MOVED", "CALM"}),
        universe=_universe().model_copy(update={"premove_change_penalty": 0.0}),
        max_tickers=2,
    )
    assert off == ["MOVED", "CALM"]  # equal score -> turnover tiebreak

    on = select_universe(
        market=FakeScanMarket(rows()),
        provider=FakeProvider(optionable={"MOVED", "CALM"}),
        universe=_universe().model_copy(update={"premove_change_penalty": 0.1}),
        max_tickers=2,
    )
    assert on == ["CALM", "MOVED"]  # +20% mover deprioritized


def test_select_universe_falls_back_to_turnover_without_volume_ratio() -> None:
    # Off-hours scans can return no volume-ratio data; ranking degrades to the
    # old turnover order instead of becoming arbitrary.
    market = FakeScanMarket(
        [
            _row("US.LOW", 1e6),
            _row("US.BIG", 9e7),
            _row("US.MID", 5e7),
            _row("US.DEAD", 8e7),  # not optionable
        ]
    )
    provider = FakeProvider(optionable={"BIG", "MID", "LOW"})

    picked = select_universe(
        market=market, provider=provider, universe=_universe(), max_tickers=2
    )

    assert picked == ["BIG", "MID"]
    assert provider.probed == ["BIG", "DEAD", "MID"]


def test_select_universe_priority_tickers_jump_the_queue() -> None:
    market = FakeScanMarket(
        [
            _row("US.BIG", 9e7, volume_ratio=5.0),
            _row("US.NEWS", 1e6, volume_ratio=0.5),  # fresh 8-K filer
        ]
    )
    provider = FakeProvider(optionable={"BIG", "NEWS"})

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=2,
        priority_tickers={"NEWS"},
    )

    assert picked == ["NEWS", "BIG"]


def test_select_universe_watchlist_is_fallback_not_queue_jump() -> None:
    # The watchlist must NOT prepend ahead of the catalyst-ranked scan -- doing
    # so forced static mega-caps to the top of every cycle. A scanned catalyst
    # name ranks first; the watchlist name only backfills the remaining slot.
    market = FakeScanMarket(
        [
            _row("US.CAT", 9e7, volume_ratio=8.0),  # real catalyst (high vol ratio)
        ]
    )
    provider = FakeProvider(optionable={"CAT", "WL"})

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=5,
        watchlist={"WL"},
    )

    # Catalyst first, watchlist appended as fallback -- not the other way round.
    assert picked == ["CAT", "WL"]


def test_select_universe_skips_fresh_rejections_without_probing() -> None:
    market = FakeScanMarket(
        [
            _row("US.AAA", 5e7, volume_ratio=3.0),
            _row("US.BBB", 4e7, volume_ratio=2.0),
            _row("US.CCC", 3e7, volume_ratio=1.0),
        ]
    )
    provider = FakeProvider(optionable={"AAA", "BBB", "CCC"})

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=2,
        skip_tickers={"AAA"},
    )

    # AAA never occupies a slot or a probe; the next names move up.
    assert picked == ["BBB", "CCC"]
    assert provider.probed == ["BBB", "CCC"]


def test_select_universe_caps_per_industry() -> None:
    market = FakeScanMarket(
        [
            _row("US.GOLD", 9e7, volume_ratio=9.0),
            _row("US.SILV", 8e7, volume_ratio=8.0),
            _row("US.MINE", 7e7, volume_ratio=7.0),
            _row("US.BIO", 1e6, volume_ratio=0.1),
        ]
    )
    provider = FakeProvider(optionable={"GOLD", "SILV", "MINE", "BIO"})

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=3,
        industry_of=lambda tickers: {
            "GOLD": "Metals & Mining",
            "SILV": "Metals & Mining",
            "MINE": "Metals & Mining",
            "BIO": "Biotech",
        },
        max_per_industry=2,
    )

    # Two miners max; the capped third miner is skipped without a probe.
    assert picked == ["GOLD", "SILV", "BIO"]
    assert "MINE" not in provider.probed


def test_select_universe_industry_lookup_failure_never_blocks() -> None:
    market = FakeScanMarket([_row("US.AAA", 5e7, volume_ratio=1.0)])
    provider = FakeProvider(optionable={"AAA"})

    def broken(tickers):
        raise RuntimeError("plates unavailable")

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=1,
        industry_of=broken,
    )

    assert picked == ["AAA"]


def test_select_universe_skips_non_alpha_and_dupes_and_caps_probes() -> None:
    market = FakeScanMarket(
        [
            _row("US.BRK.A", 9e9),  # non-alpha root: skipped without probing
            _row("US.AAA", 5e7),
            _row("US.AAA", 4e7),  # duplicate
            _row("US.BBB", 3e7),
            _row("US.CCC", 2e7),
        ]
    )
    provider = FakeProvider(optionable={"AAA", "BBB", "CCC"})

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=5,
        probe_limit=2,  # only the first two distinct names get probed
    )

    assert picked == ["AAA", "BBB"]
    assert provider.probed == ["AAA", "BBB"]


def test_select_universe_zero_max_returns_empty() -> None:
    market = FakeScanMarket([_row("US.AAA", 5e7)])
    provider = FakeProvider(optionable={"AAA"})
    assert (
        select_universe(
            market=market, provider=provider, universe=_universe(), max_tickers=0
        )
        == []
    )


# --- EligibleContractProbe -----------------------------------------------


def _contract(code: str, *, bid: float, ask: float, oi: int = 200, vol: int = 30):
    return {
        "code": code,
        "bid": bid,
        "ask": ask,
        "open_interest": oi,
        "daily_volume": vol,
        "expiry": (NOW + timedelta(days=30)).date(),
    }


class FakeChainProvider:
    def __init__(self, *, optionable: bool, chain=None, error: bool = False):
        self.optionable = optionable
        self.chain = chain or []
        self.error = error

    def is_optionable(self, ticker: str) -> bool:
        return self.optionable

    def option_chain(self, ticker, start, end, option_type="ALL"):
        if self.error:
            raise RuntimeError("feed down")
        return self.chain


def _probe(provider) -> EligibleContractProbe:
    mandate = _mandate()
    return EligibleContractProbe(
        provider=provider,
        options=mandate.options,
        execution=mandate.execution,
        now_fn=lambda: NOW,
    )


def test_probe_eligible_when_one_contract_passes_the_mandate() -> None:
    provider = FakeChainProvider(
        optionable=True,
        chain=[
            _contract("US.X1", bid=0.0, ask=0.2),  # no bid: fails
            _contract("US.X2", bid=0.19, ask=0.21),  # passes everything
        ],
    )
    assert _probe(provider).status("XXX") == PROBE_ELIGIBLE
    assert _probe(provider)("XXX") is True


def test_probe_no_contract_when_chain_exists_but_nothing_tradeable() -> None:
    provider = FakeChainProvider(
        optionable=True,
        chain=[
            # Cost 0.70 * 100 + fee buffer = $71 > the $65 mandate cap.
            _contract("US.X1", bid=0.68, ask=0.70),
            # Open interest below the mandate floor.
            _contract("US.X2", bid=0.19, ask=0.21, oi=5),
        ],
    )
    assert _probe(provider).status("XXX") == PROBE_NO_CONTRACT
    assert _probe(provider)("XXX") is False


def test_probe_no_chain_when_not_optionable() -> None:
    provider = FakeChainProvider(optionable=False)
    assert _probe(provider).status("XXX") == PROBE_NO_CHAIN


def test_probe_error_when_feed_raises() -> None:
    provider = FakeChainProvider(optionable=True, error=True)
    assert _probe(provider).status("XXX") == PROBE_ERROR
    assert _probe(provider)("XXX") is False


# --- CachedProbe -----------------------------------------------------------


class CountingProbe:
    def __init__(self, statuses: list[str]):
        self.statuses = statuses
        self.calls = 0

    def status(self, ticker: str) -> str:
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return status


def test_cached_probe_serves_fresh_negatives_without_reprobing(tmp_path) -> None:
    probe = CountingProbe([PROBE_NO_CONTRACT])
    cached = CachedProbe(probe, ProbeCache(tmp_path / "probes.json"))

    assert cached("XXX") is False
    assert cached("XXX") is False
    assert probe.calls == 1  # second answer came from the cache


def test_cached_probe_never_caches_eligible_or_error(tmp_path) -> None:
    probe = CountingProbe([PROBE_ELIGIBLE, PROBE_ERROR, PROBE_ELIGIBLE])
    cached = CachedProbe(probe, ProbeCache(tmp_path / "probes.json"))

    assert cached("XXX") is True
    assert cached("XXX") is False  # transient error: not eligible this pass
    assert cached("XXX") is True  # ...and not remembered against the name
    assert probe.calls == 3


def test_cached_probe_counts_fetches_and_cache_hits_separately(tmp_path) -> None:
    probe = CountingProbe([PROBE_NO_CONTRACT])
    cached = CachedProbe(probe, ProbeCache(tmp_path / "probes.json"))

    cached("AAA")
    cached("AAA")
    cached("AAA")

    assert cached.fetch_count == 1
    assert cached.cache_hits == 2


def test_probe_limit_charges_only_real_fetches_not_cache_hits(tmp_path) -> None:
    # Live incident 2026-06-11: after one cycle cached the whole top of the
    # ranking as negative, later cycles burned probe_limit on cache hits and
    # returned an empty universe forever. Cache hits must be free so each
    # cycle digs deeper instead.
    market = FakeScanMarket(
        [
            _row("US.AAA", 5e7, volume_ratio=3.0),
            _row("US.BBB", 4e7, volume_ratio=2.0),
            _row("US.CCC", 3e7, volume_ratio=1.0),
        ]
    )
    cache = ProbeCache(tmp_path / "probes.json")
    cache.put("AAA", PROBE_NO_CONTRACT)
    cache.put("BBB", PROBE_NO_CHAIN)
    fetcher = CountingProbe([PROBE_ELIGIBLE])
    probe = CachedProbe(fetcher, cache)

    picked = select_universe(
        market=market,
        provider=FakeProvider(optionable=set()),
        universe=_universe(),
        max_tickers=1,
        probe_limit=1,  # both cached negatives must not consume this
        probe=probe,
    )

    assert picked == ["CCC"]
    assert fetcher.calls == 1
    assert probe.cache_hits == 2


# --- pre-catalyst (earnings IV-ramp) selection ---------------------------


def test_select_universe_earnings_mode_ranks_farther_earnings_first() -> None:
    market = FakeScanMarket(
        [
            _row("US.AAA", 5e7, volume_ratio=9.0),  # would top the legacy ranking
            _row("US.BBB", 4e7, volume_ratio=1.0),
            _row("US.CCC", 3e7, volume_ratio=8.0),  # no upcoming earnings
        ]
    )
    provider = FakeProvider(optionable={"AAA", "BBB", "CCC"})
    earnings = {"AAA": date(2026, 6, 23), "BBB": date(2026, 7, 1)}

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=5,
        earnings_of=lambda tickers: earnings,
    )

    # Earnings names come FIRST, farther-earnings-first (BBB 7/1 ahead of AAA 6/23
    # -- lower current IV), so the volume-spike bias is gone among them. CCC has no
    # upcoming earnings so it is BACKFILLED last (by volume ratio) rather than
    # dropped -- the universe is never starved.
    assert picked == ["BBB", "AAA", "CCC"]


def test_select_universe_earnings_mode_backfills_when_no_upcoming_earnings() -> None:
    # The 2026-06-15 starvation: zero names in the earnings window must NOT empty
    # the universe -- it falls back to the volume-ratio ranking.
    market = FakeScanMarket(
        [
            _row("US.AAA", 5e7, volume_ratio=9.0),
            _row("US.BBB", 4e7, volume_ratio=3.0),
        ]
    )
    provider = FakeProvider(optionable={"AAA", "BBB"})

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=5,
        earnings_of=lambda tickers: {},
    )

    assert picked == ["AAA", "BBB"]  # volume-ratio backfill, not []


def test_select_universe_earnings_mode_fills_to_max_with_backfill() -> None:
    # One name has upcoming earnings; the slot budget is larger, so volume-ratio
    # names backfill the rest -- earnings name still ranks first.
    market = FakeScanMarket(
        [
            _row("US.AAA", 5e7, volume_ratio=9.0),  # no earnings, top volume
            _row("US.BBB", 4e7, volume_ratio=1.0),  # has earnings
            _row("US.CCC", 3e7, volume_ratio=5.0),  # no earnings
        ]
    )
    provider = FakeProvider(optionable={"AAA", "BBB", "CCC"})

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),
        max_tickers=3,
        earnings_of=lambda tickers: {"BBB": date(2026, 7, 1)},
    )

    # BBB (earnings) first, then AAA, CCC by volume ratio.
    assert picked == ["BBB", "AAA", "CCC"]


def _probe_iv(provider, max_entry_iv: float) -> EligibleContractProbe:
    mandate = _mandate()
    return EligibleContractProbe(
        provider=provider,
        options=mandate.options,
        execution=mandate.execution,
        now_fn=lambda: NOW,
        max_entry_iv=max_entry_iv,
    )


def test_probe_iv_ceiling_rejects_high_iv_contracts() -> None:
    high = {**_contract("US.HI", bid=0.19, ask=0.21), "iv": 2.0}  # 200% > ceiling
    low = {**_contract("US.LO", bid=0.19, ask=0.21), "iv": 1.0}  # 100% <= ceiling
    # Only a high-IV contract: nothing tradeable under the ceiling (peak-IV trap).
    assert (
        _probe_iv(FakeChainProvider(optionable=True, chain=[high]), 1.5).status("X")
        == PROBE_NO_CONTRACT
    )
    # A low-IV contract clears the ceiling -> eligible.
    assert (
        _probe_iv(
            FakeChainProvider(optionable=True, chain=[high, low]), 1.5
        ).status("X")
        == PROBE_ELIGIBLE
    )
    # Ceiling disabled (0) -> the high-IV contract is eligible again (legacy).
    assert (
        _probe_iv(FakeChainProvider(optionable=True, chain=[high]), 0.0).status("X")
        == PROBE_ELIGIBLE
    )


def test_select_universe_excluded_industries_are_never_probed() -> None:
    market = FakeScanMarket(
        [
            _row("US.SPAC", 9e7, volume_ratio=99.0),
            _row("US.REAL", 1e7, volume_ratio=1.0),
        ]
    )
    provider = FakeProvider(optionable={"SPAC", "REAL"})

    picked = select_universe(
        market=market,
        provider=provider,
        universe=_universe(),  # mandate excludes "Shell Companies"
        max_tickers=2,
        industry_of=lambda tickers: {"SPAC": "Shell Companies", "REAL": "Biotech"},
    )

    assert picked == ["REAL"]
    assert provider.probed == ["REAL"]  # the shell never costs a probe
