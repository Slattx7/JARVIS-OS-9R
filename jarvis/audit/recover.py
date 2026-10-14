"""Crash recovery. Call recover(store, stale_s=0) once at process start (nothing else is writing yet): every action
span that was started but never finished is an action whose outcome is unknown. They are surfaced with their undo
hints and (mark=True) the owning traces get an `incident:orphaned_action` label, once, and every unfinished trace is
closed with a recovered `trace_end` (status error, ProcessDied). Acknowledge a reviewed incident with
`jarvis_trace label <trace> incident:reviewed`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from . import audit
from .store import Store

INCIDENT_LABEL = "incident:orphaned_action"


@dataclass
class Recovery:
    orphaned: List[audit.ActionRow] = field(default_factory=list)
    marked: List[str] = field(default_factory=list)
    closed: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.orphaned

    def lines(self) -> List[str]:
        out = []
        for r in self.orphaned:
            hint = "undo spec recorded" if r.undo else ("snapshot recorded" if r.snapshot else "no undo information")
            out.append(f"{r.trace_id} {r.name} T{'?' if r.tier is None else r.tier} intent={r.intent!r}: outcome unknown ({hint})")
        return out


def recover(store: Store, *, stale_s: float = 300.0, mark: bool = False) -> Recovery:
    rec = Recovery(orphaned=audit.orphaned_actions(store, stale_s))
    if mark:
        for tid in sorted({r.trace_id for r in rec.orphaned}):
            if INCIDENT_LABEL not in store.load_trace(tid).label_set():
                n = sum(1 for r in rec.orphaned if r.trace_id == tid)
                store.label(tid, INCIDENT_LABEL, note=f"{n} action(s) started and never finished")
                rec.marked.append(tid)
        # every trace older than stale_s that never ended belongs to a dead process: close it honestly
        for h in store.list_traces(status="orphaned", stale_s=stale_s):
            if store.close_trace(h.trace_id, "closed by recover(): process ended without finishing"):
                rec.closed.append(h.trace_id)
    return rec
