from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class CycleLockError(RuntimeError):
    """Raised when another cycle appears to be running."""


@contextmanager
def single_instance_lock(path: Path, stale_seconds: float = 3600.0) -> Iterator[None]:
    """Prevent two cycles from running at once (which would double-order).

    A stale lock (older than ``stale_seconds``, e.g. from a crashed run) is
    reclaimed rather than blocking forever.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age <= stale_seconds:
            raise CycleLockError(
                f"Another cycle holds the lock at {path} (age {age:.0f}s). "
                "Refusing to start a concurrent cycle."
            )
        # Stale lock from a crashed run; reclaim it.

    path.write_text(f"pid={os.getpid()} ts={time.time()}\n", encoding="utf-8")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
