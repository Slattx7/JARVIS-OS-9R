"""Trace sinks + policy replay.

Audit records are the ground truth. Sinks fan them out (JSONL, memory, or your jarvis_trace emitter).
`to_envelope` is the single mapping point to adapt to the trace-harness schema.
replay() re-decides every recorded action under the *current* policy to catch regressions
(a policy change that loosens a verdict is flagged).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from .core import Action, Audit, Grants, Policy, Registry, Source, Verdict

ENVELOPE_SCHEMA = "jarvis.action.v1"


def to_envelope(rec: dict, trace_id: str = "") -> dict:
    """Map an audit record to a generic trace envelope. Edit here to match jarvis_trace."""
    return {"schema": ENVELOPE_SCHEMA, "trace_id": trace_id or rec.get("action_id", ""),
            "span": rec.get("event"), "ts": rec.get("ts"), "seq": rec.get("seq"),
            "attrs": {k: v for k, v in rec.items() if k not in ("ts", "seq", "event", "prev", "hash")},
            "hash": rec.get("hash")}


class MemorySink:
    def __init__(self): self.events = []
    def emit(self, rec): self.events.append(rec)


class JsonlSink:
    def __init__(self, path, envelope=True, trace_id=""):
        self.path, self.envelope, self.trace_id = path, envelope, trace_id
    def emit(self, rec):
        with open(self.path, "a") as f:
            f.write(json.dumps(to_envelope(rec, self.trace_id) if self.envelope else rec, default=str) + "\n")


class CallbackSink:
    """Wire to jarvis_trace: CallbackSink(lambda ev: tracer.emit(ev), envelope=True)."""
    def __init__(self, fn: Callable[[dict], None], envelope=True):
        self.fn, self.envelope = fn, envelope
    def emit(self, rec):
        self.fn(to_envelope(rec) if self.envelope else rec)


@dataclass
class Diff:
    seq: int
    action: str
    recorded: str
    now: str
    kind: str          # loosened | tightened
    reasons: list = field(default_factory=list)


@dataclass
class ReplayReport:
    total: int = 0
    same: int = 0
    skipped: int = 0
    diffs: list = field(default_factory=list)
    chain_ok: bool = True

    @property
    def loosened(self): return [d for d in self.diffs if d.kind == "loosened"]
    @property
    def tightened(self): return [d for d in self.diffs if d.kind == "tightened"]

    def summary(self) -> str:
        return (f"chain={'ok' if self.chain_ok else 'BROKEN'} decisions={self.total} same={self.same} "
                f"skipped={self.skipped} loosened={len(self.loosened)} tightened={len(self.tightened)}")


_STATEFUL_REASONS = ("lockdown", "rate limit", "covered by grant")


def replay(records, registry: Registry, *, rate=None, root_devices=None) -> ReplayReport:
    rep = ReplayReport(chain_ok=Audit.verify(records))
    for r in records:
        if r.get("event") != "decision":
            continue
        rep.total += 1
        if r.get("grant") or any(any(s in x for s in _STATEFUL_REASONS) for x in r.get("reasons", [])):
            rep.skipped += 1; continue          # depends on runtime state we don't record
        now = [r["ts"]]
        pol = Policy(registry, Grants(lambda: now[0]), lambda: now[0], rate, root_devices)
        # recorded tainted_args already include ledger hits, so replay needs no ledger
        a = Action(r["action"], r.get("args", {}), Source[r["source"]], r.get("device", "nitro"),
                   tuple(r.get("tainted_args", [])))
        d = pol.decide(a)
        rec_v, new_v = Verdict(r["verdict"]), d.verdict
        if rec_v == new_v and int(d.tier) == r["tier"]:
            rep.same += 1
        else:
            kind = "loosened" if new_v.strictness < rec_v.strictness else "tightened"
            if new_v == rec_v:
                kind = "loosened" if int(d.tier) < r["tier"] else "tightened"
            rep.diffs.append(Diff(r["seq"], r["action"], rec_v.value, new_v.value, kind, d.reasons))
    return rep
