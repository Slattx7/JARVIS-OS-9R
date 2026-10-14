"""Taint ledger: arg-level provenance via content matching.

LLMs launder types, so object-level taint can't survive a model call. Instead we record every
untrusted string the agent has *seen* (tool output, recall snippets, screen text, file names) and
check proposed action args against that ledger. Values the owner literally said are trusted.

Limits (documented in docs/THREAT_MODEL.md): paraphrase / encoding / translation by the model
defeats content matching. It is a speed bump + tier bump, not a proof.
"""
from __future__ import annotations
import re, time
from collections import deque
from dataclasses import dataclass, field


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _shingles(s: str, n: int):
    t = s.split(" ")
    return {" ".join(t[i:i + n]) for i in range(max(0, len(t) - n + 1))} if len(t) >= n else set()


def _contains(hay: str, needle: str) -> bool:
    if len(needle) >= 12:
        return needle in hay
    return re.search(r"(?<![\w])" + re.escape(needle) + r"(?![\w])", hay) is not None


@dataclass
class _Entry:
    text: str
    origin: str
    ts: float
    sh: set = field(default_factory=set)


class TaintLedger:
    def __init__(self, max_entries=64, ttl_s=1800.0, min_len=4, shingle=3, clock=time.time):
        self.untrusted: deque = deque(maxlen=max_entries)
        self.trusted: deque = deque(maxlen=8)
        self.ttl, self.min_len, self.n, self.clock = ttl_s, min_len, shingle, clock

    def add(self, text, origin: str = "tool"):
        t = _norm(text)
        if t:
            self.untrusted.append(_Entry(t, origin, self.clock(), _shingles(t, self.n)))

    def trust(self, utterance: str):
        self.trusted.append(_Entry(_norm(utterance), "owner", self.clock()))

    def _live(self, dq):
        now = self.clock()
        return [e for e in dq if now - e.ts <= self.ttl]

    def origin_of(self, value):
        v = _norm(value)
        if len(v) < self.min_len:
            return None
        for e in self._live(self.trusted):
            if _contains(e.text, v):
                return None
        vs = _shingles(v, self.n)
        for e in self._live(self.untrusted):
            if _contains(e.text, v) or (vs and vs & e.sh):
                return e.origin
        return None

    def tainted_keys(self, args: dict) -> tuple:
        out = []
        for k, v in (args or {}).items():
            if isinstance(v, (str, int, float)) and not isinstance(v, bool) and self.origin_of(v):
                out.append(k)
        return tuple(out)
