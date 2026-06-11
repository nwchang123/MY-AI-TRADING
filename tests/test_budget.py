from datetime import date
from pathlib import Path

from trading_agent.storage.budget import DailyTokenBudget

DAY = date(2026, 6, 11)
NEXT = date(2026, 6, 12)


def test_budget_accumulates_and_caps(tmp_path: Path) -> None:
    budget = DailyTokenBudget(tmp_path / "b.json", limit_tokens=1000)
    assert budget.remaining(DAY) == 1000
    budget.add(400, DAY)
    budget.add(700, DAY)
    assert budget.used(DAY) == 1100
    assert budget.remaining(DAY) == 0  # clamped, never negative


def test_budget_rolls_over_on_new_day(tmp_path: Path) -> None:
    budget = DailyTokenBudget(tmp_path / "b.json", limit_tokens=1000)
    budget.add(900, DAY)
    assert budget.remaining(DAY) == 100
    assert budget.remaining(NEXT) == 1000  # fresh day, fresh budget


def test_zero_limit_disables_the_ceiling(tmp_path: Path) -> None:
    budget = DailyTokenBudget(tmp_path / "b.json", limit_tokens=0)
    budget.add(10**9, DAY)
    assert budget.remaining(DAY) > 10**8


def test_corrupt_file_recovers(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text("garbage", encoding="utf-8")
    budget = DailyTokenBudget(path, limit_tokens=100)
    assert budget.used(DAY) == 0
    budget.add(50, DAY)
    assert budget.used(DAY) == 50
