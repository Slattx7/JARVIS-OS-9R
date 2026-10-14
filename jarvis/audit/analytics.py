"""Windowed behaviour stats over live traces + drift comparison between two windows.

    base = window_stats(store, *parse_window("14d..7d"))
    cand = window_stats(store, *parse_window("7d"))
    for f in compare(base, cand): print(f.level, f.message)

Use after a model / prompt / policy / graph change: did the allow/confirm/deny mix move, did people start
declining confirmations, did new action ids appear, did the tier-3 share or p95 latency shift, did anything orphan.
"""
from __future__ import annotations

import collections
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .schema import percentile
from .store import Store

_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _dur_ns(s: str) -> int:
    m = re.fullmatch(r"(\d+)([smhdw])", s.strip())
    if not m:
        raise ValueError(f"bad duration {s!r}; use e.g. 90m, 24h, 7d, 2w")
    return int(m.group(1)) * _UNIT[m.group(2)] * 10**9


def parse_window(spec: str, now_ns: Optional[int] = None) -> Tuple[int, int]:
    """'14d..7d' = from 14 days ago to 7 days ago; '7d' = the last 7 days."""
    now = time.time_ns() if now_ns is None else now_ns
    if ".." in spec:
        a, b = spec.split("..", 1)
        since, until = now - _dur_ns(a), now - _dur_ns(b)
    else:
        since, until = now - _dur_ns(spec), now + 1
    if since >= until:
        raise ValueError(f"empty window {spec!r}")
    return since, until


@dataclass
class WindowStats:
    since_ns: int
    until_ns: int
    traces: int = 0
    trace_errors: int = 0
    actions: Dict[str, int] = field(default_factory=dict)
    tiers: Dict[str, int] = field(default_factory=dict)
    outcomes: Dict[str, int] = field(default_factory=dict)
    decisions: Dict[str, int] = field(default_factory=dict)
    rules: Dict[str, int] = field(default_factory=dict)
    confirms_asked: int = 0
    confirms_approved: int = 0
    latency: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pre_tts: Dict[str, float] = field(default_factory=dict)
    orphans: int = 0

    def rate(self, d: Dict[str, int], key: str) -> float:
        tot = sum(d.values())
        return d.get(key, 0) / tot if tot else 0.0

    @property
    def error_rate(self) -> float:
        return self.trace_errors / self.traces if self.traces else 0.0

    @property
    def approval_rate(self) -> float:
        return self.confirms_approved / self.confirms_asked if self.confirms_asked else 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["error_rate"], d["approval_rate"] = self.error_rate, self.approval_rate
        return d


def window_stats(store: Store, since_ns: int, until_ns: int) -> WindowStats:
    st = WindowStats(since_ns, until_ns)
    actions, tiers, outcomes = collections.Counter(), collections.Counter(), collections.Counter()
    decisions, rules = collections.Counter(), collections.Counter()
    lat: Dict[str, List[float]] = collections.defaultdict(list)
    pre: List[float] = []
    for h in store.list_traces(since_ns=since_ns, until_ns=until_ns, newest_first=False):
        if "replay" in h.tags:
            continue
        tr = store.load_trace(h.trace_id)
        st.traces += 1
        st.trace_errors += int(tr.status == "error")
        for s in tr.spans:
            if s.dur_ns is not None and not s.meta.get("frozen"):
                lat[f"{s.kind}/{s.name}"].append(s.dur_ns / 1e6)
            if s.kind == "action":
                actions[s.name] += 1
                tiers[str(s.meta.get("tier", "?"))] += 1
                outcomes["dry_run" if s.meta.get("dry_run") else s.status] += 1
                st.orphans += int(s.status in ("orphaned", "running"))
            elif s.kind == "permission" and isinstance(s.outputs, dict):
                decisions[str(s.outputs.get("decision"))] += 1
                rules[str(s.outputs.get("rule"))] += 1
            elif s.kind == "confirm" and isinstance(s.outputs, dict):
                st.confirms_asked += 1
                st.confirms_approved += int(s.outputs.get("approved") is True)
        if tr.spans_of("tts"):
            pre.append(tr.first_audio_ms())
    st.actions, st.tiers, st.outcomes = dict(actions), dict(tiers), dict(outcomes)
    st.decisions, st.rules = dict(decisions), dict(rules)
    st.latency = {k: {"n": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95)} for k, v in lat.items()}
    if pre:
        st.pre_tts = {"n": len(pre), "p50": percentile(pre, 50), "p95": percentile(pre, 95)}
    return st


@dataclass
class Finding:
    level: str   # alert | warn | info
    metric: str
    message: str
    base: Any = None
    cand: Any = None


def compare(base: WindowStats, cand: WindowStats, *, decision_shift_pp: float = 10.0, confirm_shift_pp: float = 15.0,
            latency_pct: float = 25.0, latency_min_ms: float = 20.0, error_shift_pp: float = 2.0,
            min_traces: int = 10) -> List[Finding]:
    out: List[Finding] = []
    if base.traces == 0 or cand.traces == 0:  # nothing to compare against: shifts would all be artefacts
        out.append(Finding("info", "volume", f"no data in the {'baseline' if base.traces == 0 else 'candidate'} window "
                                              f"(base {base.traces} traces, candidate {cand.traces}): nothing to compare",
                           base.traces, cand.traces))
        if cand.orphans:
            out.append(Finding("alert", "orphans", f"{cand.orphans} action(s) started and never finished", 0, cand.orphans))
        return out
    if base.traces < min_traces or cand.traces < min_traces:
        out.append(Finding("info", "volume", f"low volume (base {base.traces} traces, candidate {cand.traces}; "
                                              f"need {min_traces}): shifts below may be noise", base.traces, cand.traces))
    if cand.orphans:
        out.append(Finding("alert", "orphans", f"{cand.orphans} action(s) started and never finished", 0, cand.orphans))
    if (cand.error_rate - base.error_rate) * 100 >= error_shift_pp:
        out.append(Finding("alert", "error_rate", f"trace error rate {base.error_rate:.1%} -> {cand.error_rate:.1%}",
                           base.error_rate, cand.error_rate))
    for d in ("allow", "confirm", "deny"):
        b, c = base.rate(base.decisions, d), cand.rate(cand.decisions, d)
        if abs(c - b) * 100 >= decision_shift_pp:
            level = "alert" if (d == "allow" and c > b) else "warn"
            out.append(Finding(level, f"decision.{d}", f"policy '{d}' rate {b:.1%} -> {c:.1%}", b, c))
    if base.confirms_asked and cand.confirms_asked and abs(cand.approval_rate - base.approval_rate) * 100 >= confirm_shift_pp:
        out.append(Finding("warn", "confirm_approval", f"confirmation approval rate {base.approval_rate:.1%} -> "
                                                      f"{cand.approval_rate:.1%}", base.approval_rate, cand.approval_rate))
    new_actions = {a: n for a, n in cand.actions.items() if a not in base.actions}
    for a, n in sorted(new_actions.items()):
        out.append(Finding("warn", "new_action", f"action '{a}' appeared ({n}x) and was never seen in the baseline window", 0, n))
    b3, c3 = base.rate(base.tiers, "3"), cand.rate(cand.tiers, "3")
    if (c3 - b3) * 100 >= decision_shift_pp:
        out.append(Finding("warn", "tier3_share", f"tier-3 share of actions {b3:.1%} -> {c3:.1%}", b3, c3))
    for key, cs in sorted(cand.latency.items()):
        bs = base.latency.get(key)
        if bs and cs["p95"] > bs["p95"] * (1 + latency_pct / 100) and cs["p95"] - bs["p95"] >= latency_min_ms:
            out.append(Finding("warn", f"latency.{key}", f"{key} p95 {bs['p95']:.0f}ms -> {cs['p95']:.0f}ms", bs["p95"], cs["p95"]))
    if base.pre_tts and cand.pre_tts and cand.pre_tts["p95"] > base.pre_tts["p95"] * (1 + latency_pct / 100) \
            and cand.pre_tts["p95"] - base.pre_tts["p95"] >= latency_min_ms:
        out.append(Finding("alert", "latency.pre_tts", f"wake->first-audio p95 {base.pre_tts['p95']:.0f}ms -> "
                                                      f"{cand.pre_tts['p95']:.0f}ms", base.pre_tts["p95"], cand.pre_tts["p95"]))
    order = {"alert": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda f: order[f.level])
