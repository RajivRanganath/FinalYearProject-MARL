"""Crash-safe, irreversible consumption markers for locked evaluation splits."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping


class SplitAlreadyConsumed(RuntimeError):
    pass


class FinalSplitLock:
    """Create STARTED before the first episode and COMPLETE after atomic outputs."""

    def __init__(self, output_dir: Path, metadata: Mapping[str, Any]):
        self.output_dir = Path(output_dir)
        self.started = self.output_dir / "CONSUMPTION_STARTED.json"
        self.complete = self.output_dir / "CONSUMPTION_COMPLETE.json"
        self.metadata = dict(metadata)

    def acquire(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.metadata,
            "state": "STARTED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "warning": "Existence means the locked split was exposed; never delete to rerun.",
        }
        try:
            descriptor = os.open(self.started, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise SplitAlreadyConsumed(f"Locked split already consumed: {self.started}") from exc
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def mark_complete(self, evidence: Mapping[str, Any]) -> None:
        payload = {
            **self.metadata,
            **dict(evidence),
            "state": "COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        temporary = self.complete.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, self.complete)


def atomic_dataframe_csv(frame: Any, path: Path) -> None:
    """Write a dataframe through a sibling temporary file and atomic rename."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
