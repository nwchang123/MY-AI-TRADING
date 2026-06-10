from pathlib import Path

from trading_agent.domain.risk import Mandate
from trading_agent.research.universe import select_universe


def _universe():
    return Mandate.load(Path("config/mandate.paper.yaml")).universe


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


def _row(code: str, turnover: float) -> dict:
    return {"code": code, "name": code, "cur_price": 5.0, "turnover": turnover}


def test_select_universe_ranks_by_turnover_and_filters_optionable() -> None:
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

    # Turnover order BIG > DEAD > MID; DEAD fails optionability.
    assert picked == ["BIG", "MID"]
    assert provider.probed == ["BIG", "DEAD", "MID"]


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
