from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_agent.storage.decisions import DecisionCache, decision_digest

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


def test_corrupt_file_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    path.write_text("not json", encoding="utf-8")
    cache = DecisionCache(path)
    assert cache.get("SOFI", "digest-1", NOW) is None
    cache.put("SOFI", "digest-1", "{}", NOW)  # recovers by rewriting
    assert cache.get("SOFI", "digest-1", NOW) == "{}"
