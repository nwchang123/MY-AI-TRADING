from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_SIZE_MB = 50


class AuditWriter:
    def __init__(self, path: Path):
        self.path = path

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_if_needed()
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _rotate_if_needed(self) -> None:
        if not self.path.exists():
            return
        size_mb = self.path.stat().st_size / (1024 * 1024)
        if size_mb < MAX_SIZE_MB:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archived = self.path.with_name(f"{self.path.stem}.{ts}.jsonl")
        shutil.move(str(self.path), str(archived))
        compressed = archived.with_suffix(".jsonl.gz")
        try:
            with archived.open("rb") as f_in, gzip.open(compressed, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            archived.unlink()
        except Exception:
            pass
