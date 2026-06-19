import json

from trading_agent.data.iv_history import IV30History


def test_record_and_extremes(tmp_path):
    h = IV30History(tmp_path / "iv.json")
    h.record("AAPL", 0.30, today=__import__("datetime").date(2026, 1, 1))
    h.record("AAPL", 0.20, today=__import__("datetime").date(2026, 1, 2))
    h.record("AAPL", 0.50, today=__import__("datetime").date(2026, 1, 3))
    ext = h.extremes("AAPL")
    assert ext is not None
    assert ext[0] == 0.50  # high
    assert ext[1] == 0.20  # low


def test_same_date_replaces(tmp_path):
    h = IV30History(tmp_path / "iv.json")
    h.record("AAPL", 0.30, today=__import__("datetime").date(2026, 1, 1))
    h.record("AAPL", 0.40, today=__import__("datetime").date(2026, 1, 1))
    ext = h.extremes("AAPL")
    assert ext is None  # only 1 unique observation


def test_empty_returns_none(tmp_path):
    h = IV30History(tmp_path / "iv.json")
    assert h.extremes("AAPL") is None


def test_single_observation_returns_none(tmp_path):
    h = IV30History(tmp_path / "iv.json")
    h.record("AAPL", 0.30)
    assert h.extremes("AAPL") is None


def test_zero_iv30_is_ignored(tmp_path):
    h = IV30History(tmp_path / "iv.json")
    h.record("AAPL", 0.0)
    h.record("AAPL", 0.30)
    assert h.extremes("AAPL") is None


def test_persistence(tmp_path):
    path = tmp_path / "iv.json"
    h1 = IV30History(path)
    h1.record("AAPL", 0.25, today=__import__("datetime").date(2026, 6, 1))
    h1.record("AAPL", 0.35, today=__import__("datetime").date(2026, 6, 2))
    h2 = IV30History(path)
    ext = h2.extremes("AAPL")
    assert ext == (0.35, 0.25)


def test_max_days_trims(tmp_path):
    h = IV30History(tmp_path / "iv.json", max_days=3)
    d = __import__("datetime").date
    for i in range(5):
        h.record("AAPL", 0.1 * (i + 1), today=d(2026, 1, i + 1))
    ext = h.extremes("AAPL")
    assert ext is not None
    # Should only have last 3: 0.3, 0.4, 0.5
    assert ext[0] == 0.5
    assert abs(ext[1] - 0.3) < 1e-9
