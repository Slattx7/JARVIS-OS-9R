"""Audit queries over the trace store: what did the agent do, what is unknown, how do I undo it.

Action-span contract (what the action layer should do):
  rt.call("action", action_id, args, fn, meta={"tier": 2, "intent": "...", "dry_run": False})
  inside fn:  annotate(snapshot=rt.put_blob(bytes), undo={...})   # durable for ACTION spans
The span_start (inputs + tier + intent) is persisted durably BEFORE fn runs. If the process dies mid-action
the span has no span_end: outcome "unknown". undo_plan() never executes anything; the action layer does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import BLOCKED, ERROR, OK, Span, Trace
from .store import Store


REVIEWED_LABEL = "incident:reviewed"  # `jarvis_trace label <trace> incident:reviewed` acknowledges an orphaned action


@dataclass
class ActionRow:
    trace_id: str
    trace_name: str
    span_id: str
    ts_ns: int
    name: str
    tier: Optional[int]
    outcome: str  # ok | error | unknown | blocked | dry_run
    intent: str
    args: Any
    result: Any
    undo: Any
    snapshot: Any
    irreversible: bool


@dataclass
class UndoPlan:
    trace_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)         # reverse-chronological undo specs
    irreversible: List[Dict[str, Any]] = field(default_factory=list)  # executed, tier>=1, no undo recorded


def _row(tr: Trace, s: Span) -> ActionRow:
    m = s.meta
    if s.status == OK:
        outcome = "dry_run" if m.get("dry_run") else "ok"
    elif s.status == ERROR:
        outcome = "error"
    elif s.status == BLOCKED:
        outcome = "blocked"
    else:
        outcome = "unknown"
    t = m.get("tier")
    return ActionRow(trace_id=tr.trace_id, trace_name=tr.name, span_id=s.span_id, ts_ns=s.t_start_ns, name=s.name,
                     tier=int(t) if isinstance(t, int) and not isinstance(t, bool) else None, outcome=outcome,
                     intent=str(m.get("intent", "")), args=s.inputs, result=s.outputs, undo=m.get("undo"),
                     snapshot=m.get("snapshot"), irreversible=bool(m.get("irreversible")))


def list_actions(store: Store, *, since_ns: Optional[int] = None, tier_min: int = 0,
                 trace_id: Optional[str] = None, include_blocked: bool = False) -> List[ActionRow]:
    ids = [trace_id] if trace_id else [h.trace_id for h in store.list_traces(since_ns=since_ns, newest_first=False)]
    rows: List[ActionRow] = []
    for tid in ids:
        tr = store.load_trace(tid)
        for s in tr.actions():
            r = _row(tr, s)
            if r.outcome == "blocked" and not include_blocked:
                continue
            if (3 if r.tier is None else r.tier) < tier_min:  # unknown tier counts as max
                continue
            rows.append(r)
    return rows


def orphaned_actions(store: Store, stale_s: float = 3600, include_reviewed: bool = False) -> List[ActionRow]:
    """Actions that were started (write-ahead record exists) but never finished: did it run? unknown.
    Traces labelled `incident:reviewed` are excluded unless include_reviewed."""
    out: List[ActionRow] = []
    for u in store.unfinished_spans(stale_s):
        if u["kind"] != "action":
            continue
        tr = store.load_trace(u["trace_id"], stale_s=stale_s)
        if not include_reviewed and REVIEWED_LABEL in tr.label_set():
            continue
        for s in tr.actions():
            if s.span_id == u["span_id"]:
                out.append(_row(tr, s))
    return out


def undo_plan(store: Store, trace_id: str) -> UndoPlan:
    plan = UndoPlan(trace_id)
    rows = [r for r in list_actions(store, trace_id=trace_id) if r.outcome in ("ok", "error", "unknown")]
    for r in reversed(rows):
        if r.tier == 0:
            continue
        if r.undo:
            plan.steps.append({"span_id": r.span_id, "action": r.name, "args": r.args, "undo": r.undo,
                               "maybe_applied": r.outcome != "ok", "snapshot": r.snapshot})
        else:
            plan.irreversible.append({"span_id": r.span_id, "action": r.name, "args": r.args, "tier": r.tier,
                                      "outcome": r.outcome, "snapshot": r.snapshot})
    return plan
