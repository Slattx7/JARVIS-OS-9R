"""python -m jarvis_trace <command> [--store PATH]

Exit codes: replay -> 0 pass, 1 regression over threshold, 2 safety failure.  verify -> 0 ok, 1 problems.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from . import audit, cli_ext, report_html
from .gate import GateError, load_gate
from .harness import run_suite
from .runtime import DEFAULT_LIVE
from .schema import Trace, canonical, percentile
from .store import Store

DEFAULT_STORE = os.environ.get("JARVIS_TRACE_STORE", "jarvis_traces.db")


def _ts(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _ms(ns: Optional[int]) -> str:
    return "   -   " if ns is None else f"{ns / 1e6:7.1f}ms"


def _since(spec: Optional[str]) -> Optional[int]:
    if not spec:
        return None
    m = re.fullmatch(r"(\d+)([smhd])", spec)
    if not m:
        raise SystemExit(f"bad --since {spec!r}; use e.g. 30m, 24h, 7d")
    return time.time_ns() - int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)] * 10**9


def _short(obj: Any, n: int = 90) -> str:
    s = canonical(obj) if obj is not None else ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _load_callable(spec: str) -> Any:
    if ":" not in spec:
        raise SystemExit("--agent must be module:function or path/to/file.py:function")
    mod_s, fn_s = spec.rsplit(":", 1)
    sys.path.insert(0, os.getcwd())
    if mod_s.endswith(".py") and os.path.exists(mod_s):
        name = "jt_agent_" + re.sub(r"\W", "_", os.path.abspath(mod_s))
        s = importlib.util.spec_from_file_location(name, mod_s)
        mod = importlib.util.module_from_spec(s)  # type: ignore[arg-type]
        sys.modules[name] = mod
        s.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        mod = importlib.import_module(mod_s)
    fn = getattr(mod, fn_s, None)
    if not callable(fn):
        raise SystemExit(f"{spec}: not callable")
    return fn, mod


# -- commands -------------------------------------------------------------------- #

def cmd_ls(a: argparse.Namespace) -> int:
    st = Store(a.store)
    for t in st.list_traces(tag=a.tag, label=a.label, name=a.name, since_ns=_since(a.since), limit=a.limit):
        print(f"{t.trace_id}  {_ts(t.t_start_ns)}  {t.name:<14} {t.status:<9} {_ms(t.dur_ns)}  "
              f"tags={','.join(t.tags)} labels={','.join(sorted(t.label_set()))}")
    return 0


def _render(tr: Trace, io: bool) -> None:
    print(f"{tr.trace_id}  {tr.name}  [{tr.status}]  {_ms(tr.dur_ns).strip()}  tags={','.join(tr.tags)}  "
          f"labels={','.join(sorted(tr.label_set()))}")
    if io:
        print(f"   in : {_short(tr.inputs)}\n   out: {_short(tr.outputs)}")
    ids = {s.span_id for s in tr.spans}
    kids: Dict[Optional[str], List[Any]] = collections.defaultdict(list)
    for s in tr.spans:
        kids[s.parent_id if s.parent_id in ids else None].append(s)

    def walk(parent: Optional[str], prefix: str) -> None:
        for i, s in enumerate(kids.get(parent, [])):
            last = i == len(kids[parent]) - 1
            extra = []
            if "tier" in s.meta:
                extra.append(f"tier={s.meta['tier']}")
            if s.meta.get("frozen"):
                extra.append("frozen")
            if s.meta.get("stub"):
                extra.append("STUB")
            if s.status == "error" and s.error:
                extra.append(f"err={s.error.get('type')}")
            print(f"{prefix}{'└─ ' if last else '├─ '}{s.kind:<10} {s.name:<22} {s.status:<8} {_ms(s.dur_ns)} "
                  f"{' '.join(extra)}")
            if io:
                pad = prefix + ("   " if last else "│  ") + "   "
                print(f"{pad}in : {_short(s.inputs)}\n{pad}out: {_short(s.outputs)}")
            walk(s.span_id, prefix + ("   " if last else "│  "))

    walk(None, "")


def cmd_show(a: argparse.Namespace) -> int:
    st = Store(a.store)
    try:
        tr = st.load_trace(a.trace_id)
    except KeyError:
        print(f"no such trace {a.trace_id}", file=sys.stderr)
        return 1
    if a.html:
        with open(a.html, "w") as f:
            f.write(report_html.render_trace_html(tr))
        print(f"wrote {a.html}")
    elif a.json:
        print(json.dumps(tr.to_dict(), indent=2, default=str))
    else:
        _render(tr, a.io)
    return 0


def cmd_label(a: argparse.Namespace) -> int:
    st = Store(a.store)
    try:
        st.label(a.trace_id, a.label, a.note or "", remove=a.remove)
    except KeyError:
        print(f"no such trace {a.trace_id}", file=sys.stderr)
        return 1
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    st = Store(a.store)
    agent, mod = _load_callable(a.agent)
    try:
        gate = load_gate(a.gate)
    except (GateError, OSError, ValueError) as e:
        print(f"gate: {e}", file=sys.stderr)
        return 1
    scorers = getattr(mod, "SCORERS", None) or gate.scorers
    cand = Store(a.candidate_store) if a.candidate_store else None
    select = dict(gate.select)
    for k in ("tag", "label", "name"):
        if getattr(a, k):
            select[k] = getattr(a, k)
    if a.since:
        select["since_ns"] = _since(a.since)
    select["limit"] = a.limit if a.limit is not None else select.get("limit", 200)
    live = [x for x in a.live.split(",") if x] if a.live is not None else gate.live
    report = run_suite(
        st, agent, select=select, live=live, on_miss=a.on_miss or gate.on_miss, scorers=scorers, candidate_store=cand,
        label=a.agent, min_pass=a.min_pass if a.min_pass is not None else gate.min_pass,
        include_failed=a.include_failed)
    a._code = report.exit_code()  # survives a closed stdout (e.g. `| head`): the gate must not be swallowed
    print(report.summary_text())
    if a.out:
        with open(a.out, "w") as f:
            f.write(report.to_markdown())
    if a.json:
        with open(a.json, "w") as f:
            f.write(report.to_json())
    if a.html:
        with open(a.html, "w") as f:
            f.write(report_html.render_report_html(report))
    return report.exit_code()


def cmd_audit(a: argparse.Namespace) -> int:
    st = Store(a.store)
    rows = audit.orphaned_actions(st) if a.orphans else audit.list_actions(
        st, since_ns=_since(a.since), tier_min=a.tier_min, include_blocked=a.blocked)
    for r in rows:
        flag = "  <-- OUTCOME UNKNOWN" if r.outcome == "unknown" else ""
        print(f"{_ts(r.ts_ns)}  {r.trace_id}  T{'?' if r.tier is None else r.tier}  {r.outcome:<8} {r.name:<16} "
              f"{_short(r.args, 60)}  undo={'yes' if r.undo else ('NO' if (r.tier or 0) > 0 else '-')}{flag}")
    if a.orphans and not rows:
        print("no orphaned actions")
    return 0


def cmd_undo_plan(a: argparse.Namespace) -> int:
    st = Store(a.store)
    if not st.trace_exists(a.trace_id):
        print(f"no such trace {a.trace_id}", file=sys.stderr)
        return 1
    p = audit.undo_plan(st, a.trace_id)
    if a.json:
        print(json.dumps({"trace_id": p.trace_id, "steps": p.steps, "irreversible": p.irreversible}, indent=2,
                         default=str))
        return 0
    for i, s in enumerate(p.steps, 1):
        print(f"{i}. undo {s['action']} {_short(s['args'], 50)} -> {_short(s['undo'], 80)}"
              f"{'  (may not have applied)' if s['maybe_applied'] else ''}")
    for s in p.irreversible:
        print(f"!  no undo recorded: {s['action']} {_short(s['args'], 50)} (tier {s['tier']}, {s['outcome']})"
              f"{'  snapshot available' if s['snapshot'] else ''}")
    if not p.steps and not p.irreversible:
        print("nothing to undo")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    st = Store(a.store)
    r = st.verify()
    print(f"events={r.events} head={r.head}")
    if r.rotation:
        print(f"rotated from {r.rotation['prev_path']}: previous chain head {r.rotation['prev_head']}")
        if a.prev:
            old = Store(a.prev)
            ok = old.head() == r.rotation["prev_head"] and old.verify().ok
            print(f"previous store {a.prev}: {'head matches and chain verifies' if ok else 'MISMATCH or invalid chain'}")
            if not ok:
                r.problems.append(f"previous store {a.prev} does not match the recorded rotation head")
                r.ok = False
    for p in r.problems:
        print(f"PROBLEM: {p}")
    for u in r.unfinished:
        print(f"unfinished span: {u['kind']}/{u['name']} trace={u['trace_id']} ({_ts(u['ts_ns'])})")
    print("OK" if r.ok else "FAILED")
    return 0 if r.ok else 1


def cmd_stats(a: argparse.Namespace) -> int:
    st = Store(a.store)
    d: Dict[Any, List[float]] = collections.defaultdict(list)
    for h in st.list_traces(since_ns=_since(a.since), limit=a.limit):
        for s in st.load_trace(h.trace_id).spans:
            if s.dur_ns is not None and not s.meta.get("frozen"):
                d[(s.kind, s.name)].append(s.dur_ns / 1e6)
    print(f"{'kind':<11}{'name':<24}{'n':>5}{'p50':>9}{'p95':>9}{'max':>9}   (ms)")
    for (k, n), xs in sorted(d.items()):
        print(f"{k:<11}{n:<24}{len(xs):>5}{percentile(xs, 50):>9.1f}{percentile(xs, 95):>9.1f}{max(xs):>9.1f}")
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    st = Store(a.store)
    ids = [h.trace_id for h in st.list_traces(tag=a.tag, label=a.label, newest_first=False)] if (a.tag or a.label) else None
    fp = sys.stdout if a.file == "-" else open(a.file, "w")
    try:
        n = st.export_jsonl(fp, ids)
    finally:
        if fp is not sys.stdout:
            fp.close()
    print(f"exported {n} events", file=sys.stderr)
    return 0


def cmd_import(a: argparse.Namespace) -> int:
    st = Store(a.store)
    fp = sys.stdin if a.file == "-" else open(a.file)
    try:
        ev, tr, bl = st.import_jsonl(fp)
    finally:
        if fp is not sys.stdin:
            fp.close()
    print(f"imported {ev} events, {tr} traces, {bl} blobs")
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    from . import demo
    st = Store(a.store)
    ids = demo.record_baseline(st)
    print(f"recorded {len(ids)} golden demo traces into {a.store}")
    print("try:\n  python -m jarvis_trace ls --store %s\n  python -m jarvis_trace show %s --io --store %s\n"
          "  python -m jarvis_trace replay --agent jarvis_trace.demo:baseline --store %s\n"
          "  python -m jarvis_trace replay --agent jarvis_trace.demo:loose_policy --store %s   # exit 2\n"
          "  python -m jarvis_trace audit --store %s --tier-min 2" % (a.store, ids[2], a.store, a.store, a.store, a.store))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="jarvis_trace", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default=DEFAULT_STORE, help="SQLite path (env JARVIS_TRACE_STORE)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def sp(name: str, fn: Callable[..., int], help_: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--store", default=argparse.SUPPRESS)
        s.set_defaults(fn=fn)
        return s

    s = sp("ls", cmd_ls, "list traces")
    s.add_argument("--tag"); s.add_argument("--label"); s.add_argument("--name")
    s.add_argument("--since"); s.add_argument("--limit", type=int, default=50)
    s = sp("show", cmd_show, "render one trace as a tree")
    s.add_argument("trace_id"); s.add_argument("--io", action="store_true"); s.add_argument("--json", action="store_true")
    s.add_argument("--html", help="write a self-contained HTML waterfall to this file")
    s = sp("label", cmd_label, "label a trace (e.g. golden, bad)")
    s.add_argument("trace_id"); s.add_argument("label"); s.add_argument("--note"); s.add_argument("--remove", action="store_true")
    s = sp("replay", cmd_replay, "replay baseline traces through a candidate agent and score")
    s.add_argument("--agent", required=True, help="module:function or file.py:function; fn(rt, inputs)")
    s.add_argument("--tag"); s.add_argument("--label"); s.add_argument("--name"); s.add_argument("--since")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--live", default=None, help=f"kinds that run candidate code (default {','.join(sorted(DEFAULT_LIVE))}; action never does)")
    s.add_argument("--on-miss", default=None, choices=["positional", "raise", "stub"])
    s.add_argument("--min-pass", type=float, default=None)
    s.add_argument("--gate", help="gate config JSON (thresholds, scorers, live kinds, selection)")
    s.add_argument("--html", help="write a self-contained HTML report")
    s.add_argument("--include-failed", action="store_true")
    s.add_argument("--candidate-store", help="persist candidate traces here (default: in-memory)")
    s.add_argument("--out", help="write markdown report"); s.add_argument("--json", help="write JSON report")
    s = sp("audit", cmd_audit, "list actions the agent took")
    s.add_argument("--since"); s.add_argument("--tier-min", type=int, default=0)
    s.add_argument("--orphans", action="store_true", help="only actions started but never finished")
    s.add_argument("--blocked", action="store_true")
    s = sp("undo-plan", cmd_undo_plan, "print (never execute) the reverse-order undo steps for a trace")
    s.add_argument("trace_id"); s.add_argument("--json", action="store_true")
    s = sp("verify", cmd_verify, "check hash chain, blobs, unfinished spans")
    s.add_argument("--prev", help="previous store (after rotate): check its head matches the recorded rotation")
    s = sp("stats", cmd_stats, "per-span latency percentiles (live spans only)")
    s.add_argument("--since"); s.add_argument("--limit", type=int, default=500)
    s = sp("export", cmd_export, "export JSONL (with blobs)")
    s.add_argument("file"); s.add_argument("--tag"); s.add_argument("--label")
    s = sp("import", cmd_import, "import JSONL (idempotent per trace)")
    s.add_argument("file")
    sp("demo", cmd_demo, "record demo golden traces")

    cli_ext.register(sub)
    a = p.parse_args(argv)
    try:
        code = int(a.fn(a))
        sys.stdout.flush()
        return code
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # silence interpreter-exit flush error
        return int(getattr(a, "_code", 0))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
