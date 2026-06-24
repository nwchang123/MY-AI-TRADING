from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_agent.storage.decisions import (
    DecisionCache,
    candidate_digest,
    decision_digest,
    thesis_digest,
)

NOW = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)


def test_digest_is_order_insensitive_but_content_sensitive() -> None:
    a = decision_digest("sofi", ["e1", "e2"], ["US.X1", "US.X2"])
    b = decision_digest("SOFI", ["e2", "e1"], ["US.X2", "US.X1"])
    c = decision_digest("SOFI", ["e2", "e1", "e3"], ["US.X2", "US.X1"])
    d = decision_digest("SOFI", ["e1", "e2"], ["US.X1"])
    assert a == b
    assert a != c  # new evidence changes the digest
    assert a != d  # changed candidate list changes the digest


def test_cache_roundtrip_and_digest_mismatch(tmp_path: Path) -> None:
    cache = DecisionCache(tmp_path / "decisions.json")
    cache.put("SOFI", "digest-1", '{"decision":"reject"}', NOW)

    assert cache.get("SOFI", "digest-1", NOW) == '{"decision":"reject"}'
    assert cache.get("sofi", "digest-1", NOW) == '{"decision":"reject"}'  # case
    assert cache.get("SOFI", "digest-2", NOW) is None  # inputs changed
    assert cache.get("PLUG", "digest-1", NOW) is None  # other ticker


def test_cache_expires_after_ttl(tmp_path: Path) -> None:
    cache = DecisionCache(tmp_path / "decisions.json", ttl_hours=6)
    cache.put("SOFI", "digest-1", "{}", NOW)

    assert cache.get("SOFI", "digest-1", NOW + timedelta(hours=5)) == "{}"
    assert cache.get("SOFI", "digest-1", NOW + timedelta(hours=7)) is None


def test_fresh_rejections_lists_unexpired_non_opens(tmp_path: Path) -> None:
    cache = DecisionCache(tmp_path / "decisions.json", ttl_hours=6)
    cache.put("REJ", "d1", '{"decision":"no_action"}', NOW)
    cache.put("OPEN", "d2", '{"decision":"open_position"}', NOW)
    cache.put("STALE", "d3", '{"decision":"no_action"}', NOW - timedelta(hours=7))
    cache.put("BROKEN", "d4", "not json", NOW)

    # Only the fresh, parseable rejection is benched; an expired one competes
    # for a universe slot again, and an open decision never blocks.
    assert cache.fresh_rejections(NOW) == {"REJ"}


def test_fresh_rejections_empty_cache(tmp_path: Path) -> None:
    cache = DecisionCache(tmp_path / "decisions.json")
    assert cache.fresh_rejections(NOW) == set()


def test_fresh_rejections_within_hours_shortens_the_bench(tmp_path: Path) -> None:
    cache = DecisionCache(tmp_path / "decisions.json", ttl_hours=6)
    cache.put("OLD", "d1", '{"decision":"no_action"}', NOW - timedelta(hours=3))
    cache.put("NEW", "d2", '{"decision":"no_action"}', NOW - timedelta(hours=1))

    # Both are inside the 6h cache TTL, but only the recent one stays benched
    # under a 2h selection window; the older name competes for a slot again.
    assert cache.fresh_rejections(NOW) == {"OLD", "NEW"}
    assert cache.fresh_rejections(NOW, within_hours=2.0) == {"NEW"}


def test_thesis_digest_ignores_contract_drift() -> None:
    # Same evidence, different eligible contracts: the full digest changes
    # (busting the cache) but the thesis sub-digest is stable.
    full_a = decision_digest("SOFI", ["e1", "e2"], ["US.X1"])
    full_b = decision_digest("SOFI", ["e1", "e2"], ["US.X2"])
    assert full_a != full_b
    assert thesis_digest("SOFI", ["e1", "e2"]) == thesis_digest("sofi", ["e2", "e1"])
    assert candidate_digest(["US.X1"]) != candidate_digest(["US.X2"])


def test_classify_miss_distinguishes_drift_from_fresh_evidence(tmp_path: Path) -> None:
    cache = DecisionCache(tmp_path / "decisions.json", ttl_hours=6)
    th = thesis_digest("SOFI", ["e1", "e2"])
    cd = candidate_digest(["US.X1"])
    cache.put("SOFI", "d1", '{"decision":"reject"}', NOW, thesis=th, candidates=cd)

    # Thesis unchanged, contracts drifted -> the dominant, tolerable cause.
    assert cache.classify_miss("SOFI", th, candidate_digest(["US.X2"]), NOW) == (
        "candidates_changed"
    )
    # New evidence, same contracts -> a genuine re-evaluation.
    assert cache.classify_miss(
        "SOFI", thesis_digest("SOFI", ["e1", "e2", "e3"]), cd, NOW
    ) == "evidence_changed"
    assert cache.classify_miss(
        "SOFI", thesis_digest("SOFI", ["e9"]), candidate_digest(["US.X9"]), NOW
    ) == "both_changed"
    assert cache.classify_miss("SOFI", th, cd, NOW) == "match"
    assert cache.classify_miss("PLUG", th, cd, NOW) == "no_prior"
    assert cache.classify_miss(
        "SOFI", th, cd, NOW + timedelta(hours=7)
    ) == "expired"


def test_classify_miss_legacy_entry_without_subdigests(tmp_path: Path) -> None:
    cache = DecisionCache(tmp_path / "decisions.json")
    cache.put("SOFI", "d1", "{}", NOW)  # written before sub-digests existed
    assert cache.classify_miss("SOFI", "th", "cd", NOW) == "legacy_entry"


def test_corrupt_file_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    path.write_text("not json", encoding="utf-8")
    cache = DecisionCache(path)
    assert cache.get("SOFI", "digest-1", NOW) is None
    cache.put("SOFI", "digest-1", "{}", NOW)  # recovers by rewriting
    assert cache.get("SOFI", "digest-1", NOW) == "{}"
