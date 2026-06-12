from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class CycleLockError(RuntimeError):
    """Raised when another cycle appears to be running."""


def _pid_alive(pid: int) -> bool:
    """Check whether *pid* refers to a running process (Windows & POSIX)."""
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, PermissionError):
        return False


def _read_pid(path: Path) -> int | None:
    """Read PID from lock file."""
    try:
        text = path.read_text(encoding="utf-8")
        m = re.search(r"pid=(\d+)", text)
        return int(m.group(1)) if m else None
    except OSError:
        return None


@contextmanager
def single_instance_lock(path: Path, stale_seconds: float = 3600.0) -> Iterator[None]:
    """Prevent two cycles from running at once (which would double-order).

    Uses atomic file creation (O_CREAT | O_EXCL) to avoid TOCTOU race
    conditions between checking and creating the lock file.

    A stale lock (older than stale_seconds, e.g. from a crashed run) is
    reclaimed rather than blocking forever. The lock is also reclaimed
    immediately when the PID recorded in the file is no longer alive.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    # Try atomic lock creation
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Lock file exists - check if it is stale or held by a dead process
        age = time.time() - path.stat().st_mtime
        holding_pid = _read_pid(path)

        if age <= stale_seconds:
            if holding_pid is not None and not _pid_alive(holding_pid):
                pass  # Process is dead, reclaim the lock
            else:
                raise CycleLockError(
                    f"Another cycle holds the lock at {path} (age {age:.0f}s). "
                    "Refusing to start a concurrent cycle."
                )

        # Stale lock or dead process - remove and recreate atomically
        try:
            path.unlink()
        except OSError:
            pass
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    # Write PID and timestamp
    try:
        os.write(fd, f"pid={os.getpid()} ts={time.time()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)
