"""Tests for the live-trade ramp counter."""

from __future__ import annotations

from pathlib import Path

from trading_agent.storage.live_counter import LiveTradeCounter


def test_default_counts_are_zero(tmp_path: Path) -> None:
    counter = LiveTradeCounter(tmp_path / "lc.json")
    state = counter.load()
    assert state.trades_completed == 0
    assert state.reviewed_count == 0


def test_increment_completed(tmp_path: Path) -> None:
    counter = LiveTradeCounter(tmp_path / "lc.json")
    assert counter.increment_completed() == 1
    assert counter.increment_completed() == 2
    assert counter.load().trades_completed == 2


def test_increment_reviewed(tmp_path: Path) -> None:
    counter = LiveTradeCounter(tmp_path / "lc.json")
    assert counter.increment_reviewed() == 1
    assert counter.increment_reviewed() == 2
    assert counter.load().reviewed_count == 2


def test_counters_are_independent(tmp_path: Path) -> None:
    counter = LiveTradeCounter(tmp_path / "lc.json")
    counter.increment_completed()
    counter.increment_completed()
    counter.increment_reviewed()
    state = counter.load()
    assert state.trades_completed == 2
    assert state.reviewed_count == 1


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "lc.json"
    c1 = LiveTradeCounter(path)
    c1.increment_completed()
    c1.increment_reviewed()
    # Fresh instance reads from disk.
    c2 = LiveTradeCounter(path)
    state = c2.load()
    assert state.trades_completed == 1
    assert state.reviewed_count == 1


def test_corrupt_file_returns_zeros(tmp_path: Path) -> None:
    path = tmp_path / "lc.json"
    path.write_text("NOT JSON", encoding="utf-8")
    counter = LiveTradeCounter(path)
    state = counter.load()
    assert state.trades_completed == 0
    assert state.reviewed_count == 0


def test_missing_file_returns_zeros(tmp_path: Path) -> None:
    counter = LiveTradeCounter(tmp_path / "nope.json")
    state = counter.load()
    assert state.trades_completed == 0
    assert state.reviewed_count == 0
