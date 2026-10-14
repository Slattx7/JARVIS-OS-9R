"""Adapter point for the recall layer (local personal context graph).
The real API is unknown to this package, so it is a one-method protocol with two stock adapters.
HttpRecall expects POST {"q": str, "k": int} -> [{"text": str, "source": str, "ts": float}, ...].
Everything recall returns is UNTRUSTED data (it contains captured third-party content).
"""
from __future__ import annotations
import json, urllib.request
from typing import Callable, Protocol


class RecallClient(Protocol):
    def query(self, q: str, k: int = 5) -> list: ...


class CallableRecall:
    def __init__(self, fn: Callable[[str, int], list]): self.fn = fn
    def query(self, q, k=5): return list(self.fn(q, k))[:k]


class HttpRecall:
    def __init__(self, url: str, timeout: float = 5.0):
        if not url.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
            raise ValueError("recall endpoint must be loopback")
        self.url, self.timeout = url, timeout

    def query(self, q, k=5):
        req = urllib.request.Request(self.url, json.dumps({"q": q, "k": k}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read(1_000_000))
        return data[:k] if isinstance(data, list) else []


def snippets(results: list) -> list:
    out = []
    for r in results:
        out.append(str(r.get("text", "")) if isinstance(r, dict) else str(r))
    return [s for s in out if s]
