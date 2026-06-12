from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_agent.storage.probes import (
    PROBE_ELIGIBLE,
    PROBE_ERROR,
    PROBE_NO_CHAIN,
    PROBE_NO_CONTRACT,
    ProbeCache,
)

NOW = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)


def test_negative_statuses_roundtrip(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    cache.put("aaa", PROBE_NO_CHAIN, NOW)
    cache.put("BBB", PROBE_NO_CONTRACT, NOW)

    assert cache.get("AAA", NOW) == PROBE_NO_CHAIN  # case-insensitive
    assert cache.get("BBB", NOW) == PROBE_NO_CONTRACT
    assert cache.get("CCC", NOW) is None


def test_positive_and_error_statuses_are_not_persisted(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    cache.put("AAA", PROBE_ELIGIBLE, NOW)
    cache.put("BBB", PROBE_ERROR, NOW)

    assert cache.get("AAA", NOW) is None
    assert cache.get("BBB", NOW) is None


def test_ttls_differ_per_status(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    cache.put("DEAD", PROBE_NO_CHAIN, NOW)
    cache.put("QUIET", PROBE_NO_CONTRACT, NOW)

    # "No eligible contract" flips when a catalyst wakes the options up, so it
    # is rechecked after 2h; "no chain at all" holds for the whole day.
    later = NOW + timedelta(hours=3)
    assert cache.get("QUIET", later) is None
    assert cache.get("DEAD", later) == PROBE_NO_CHAIN
    assert cache.get("DEAD", NOW + timedelta(hours=25)) is None


def test_corrupt_file_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "probes.json"
    path.write_text("not json", encoding="utf-8")
    cache = ProbeCache(path)
    assert cache.get("AAA", NOW) is None
    cache.put("AAA", PROBE_NO_CHAIN, NOW)  # recovers by rewriting
    assert cache.get("AAA", NOW) == PROBE_NO_CHAIN
