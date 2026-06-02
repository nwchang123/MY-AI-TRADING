import os
import time
from pathlib import Path

import pytest

from trading_agent.execution.lock import CycleLockError, single_instance_lock


def test_lock_blocks_concurrent_acquire(tmp_path: Path) -> None:
    lock = tmp_path / "cycle.lock"
    with single_instance_lock(lock):
        assert lock.exists()
        with pytest.raises(CycleLockError):
            with single_instance_lock(lock):
                pass
    # Released on exit.
    assert not lock.exists()


def test_stale_lock_is_reclaimed(tmp_path: Path) -> None:
    lock = tmp_path / "cycle.lock"
    lock.write_text("pid=999 ts=0\n", encoding="utf-8")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    # Stale (older than the default 3600s) -> reclaimed, no error.
    with single_instance_lock(lock):
        assert lock.exists()
    assert not lock.exists()
