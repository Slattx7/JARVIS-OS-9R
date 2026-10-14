"""Golden-set curation: pick a small, *diverse* set of recorded turns to label `golden`, instead of the last N.

Signature of a trace = which (action, tier) pairs it contains + which policy decisions / declined confirmations
occurred. Traces carrying tier>=2 actions, denials or declines are prioritised (they are the ones a regression
hurts most); within a signature the most recent are kept. Errors are excluded unless asked for."""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schema import Trace
from .store import Store


def _tier(span) -> int:
    t = span.meta.get("tier")
    return int(t) if isinstance(t, int) and not isinstance(t, bool) else 3


def signature(tr: Trace) -> Tuple:
    sig = {("act", a.name, _tier(a)) for a in tr.spans_of("action") if a.status != "blocked"}
    for p in tr.spans_of("permission"):
        if isinstance(p.outputs, dict) and p.outputs.get("decision"):
            sig.add(("dec", p.outputs["decision"]))
    for c in tr.spans_of("confirm"):
        if isinstance(c.outputs, dict) and c.outputs.get("approved") is False:
            sig.add(("declined",))
    return tuple(sorted(sig))


def _priority(sig: Tuple) -> int:
    hot = any((e[0] == "act" and e[2] >= 2) or e in (("dec", "deny"), ("declined",)) for e in sig)
    return 0 if hot else (1 if sig else 2)


@dataclass
class CurateResult:
    selected: List[str] = field(default_factory=list)
    groups: Dict[Tuple, int] = field(default_factory=dict)   # signature -> traces available
    considered: int = 0
    applied: bool = False


def curate(store: Store, *, since_ns: Optional[int] = None, max_total: int = 100, per_signature: int = 2,
           label: str = "golden", include_failed: bool = False, dry_run: bool = False) -> CurateResult:
    res = CurateResult()
    groups: Dict[Tuple, List[Trace]] = collections.defaultdict(list)
    for h in store.list_traces(since_ns=since_ns, newest_first=True):
        if (h.status != "ok" and not include_failed) or label in h.label_set() or "replay" in h.tags or "undo" in h.tags:
            continue
        tr = store.load_trace(h.trace_id)
        groups[signature(tr)].append(tr)
        res.considered += 1
    res.groups = {k: len(v) for k, v in groups.items()}
    picked: List[str] = []
    for sig in sorted(groups, key=lambda s: (_priority(s), -len(groups[s]), s)):
        for tr in groups[sig][:per_signature]:  # newest first
            if len(picked) >= max_total:
                break
            picked.append(tr.trace_id)
    res.selected = picked
    if not dry_run:
        for tid in picked:
            store.label(tid, label)
        res.applied = True
    return res
