"""Tamper-evident audit log: JSONL where each entry commits to the previous entry's hash."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

GENESIS = "0" * 64


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + _canon(body)).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | os.PathLike | None, clock=time.time):
        self.path = Path(os.path.expanduser(str(path))) if path else None
        self.clock = clock
        self._lock = threading.Lock()
        self._seq = 0
        self._prev = GENESIS
        self.entries: list[dict] = []
        if self.path and self.path.exists():
            ok, n, err = verify(self.path)
            if not ok:
                raise RuntimeError(f"audit log {self.path} failed verification: {err}")
            if n:
                last = json.loads(self.path.read_text().strip().splitlines()[-1])
                self._seq, self._prev = last["seq"] + 1, last["hash"]

    def append(self, event: str, **data) -> dict:
        with self._lock:
            body = {"seq": self._seq, "ts": round(self.clock(), 3), "event": event, "data": data}
            entry = dict(body, prev=self._prev, hash=_hash(self._prev, body))
            self._seq += 1
            self._prev = entry["hash"]
            self.entries.append(entry)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                new = not self.path.exists()
                with open(self.path, "a") as f:
                    f.write(_canon(entry) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                if new:
                    os.chmod(self.path, 0o600)
            return entry


def verify(path: str | os.PathLike) -> tuple[bool, int, str | None]:
    p = Path(os.path.expanduser(str(path)))
    prev, n = GENESIS, 0
    if not p.exists():
        return True, 0, None
    for i, line in enumerate(p.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            body = {k: e[k] for k in ("seq", "ts", "event", "data")}
        except (json.JSONDecodeError, KeyError):
            return False, n, f"line {i + 1}: malformed"
        if e.get("seq") != n:
            return False, n, f"line {i + 1}: seq gap/reorder (expected {n}, got {e.get('seq')})"
        if e.get("prev") != prev:
            return False, n, f"line {i + 1}: chain broken"
        if e.get("hash") != _hash(prev, body):
            return False, n, f"line {i + 1}: hash mismatch (entry modified)"
        prev, n = e["hash"], n + 1
    return True, n, None
