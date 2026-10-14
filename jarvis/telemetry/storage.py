"""Durable storage for captured trace events.

JSONL, not SQLite, for this phase: collection happens over a live,
occasionally-flaky USB/Frida connection, and an append-only log survives
a killed process or a dropped device far better than a SQLite file that
gets caught mid-write. Batch-loading a session's log into a queryable
store is a feature-engineering concern (phase 2), not a collection-time
one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schema import TraceEvent


class TraceStore:
    """Append-only log of TraceEvent records at `path`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json())
            fh.write("\n")

    def append_many(self, events: list[TraceEvent]) -> None:
        if not events:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for event in events:
                fh.write(event.model_dump_json())
                fh.write("\n")

    def iter_events(self) -> Iterator[TraceEvent]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield TraceEvent.model_validate(json.loads(line))

    def read_all(self) -> list[TraceEvent]:
        return list(self.iter_events())
