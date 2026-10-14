"""Operational commands. Registered into the main parser by cli.main()."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from typing import Any, Callable, Dict, List

from . import analytics, audit, killswitch
from .actions import UndoRunner
from .curate import curate
from .invariants import check_trace, live_invariants
from .policy import (PolicyEngine, PolicyError, cases_from_store, diff_policies, run_cases)
from .recall import recall_eval
from .recover import recover
from .schema import canonical
from .store import Store


def _helpers():  # late import: cli imports this module
    from . import cli
    return cli


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _engine(path: str) -> PolicyEngine:
    try:
        return PolicyEngine.from_file(path)
    except (PolicyError, ValueError, OSError) as e:
        print(f"{path}: {e}", file=sys.stderr)
        raise SystemExit(1)


# -- pause / resume / status -------------------------------------------------------------------------------------- #

def cmd_pause(a: argparse.Namespace) -> int:
    d = killswitch.engage(a.file, a.reason, who=os.environ.get("USER", "cli"))
    print(f"PAUSED ({a.file or killswitch.default_path()}): {d['reason'] or 'no reason given'}\n"
          "tier>=1 actions are denied by policy rule 'paused' until `resume`; tier 0 still runs.")
    return 0


def cmd_resume(a: argparse.Namespace) -> int:
    print("resumed" if killswitch.release(a.file) else "was not paused")
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    cli = _helpers()
    st = Store(a.store)
    heads = st.list_traces()
    ps = killswitch.state(a.file)
    orphans = audit.orphaned_actions(st)
    print(f"store      {st.path}\nevents     {st.count_events()}\ntraces     {len(heads)}"
          + (f" (latest {cli._ts(heads[0].t_start_ns)})" if heads else ""))
    print(f"chain head {st.head()}")
    rot = st.rotation_info()
    if rot:
        print(f"rotated    from {rot['prev_path']} (prev head {rot['prev_head'][:16]}…, kept {rot['kept_traces']} traces)")
    print(f"paused     {'YES: ' + str(ps.get('reason')) if ps else 'no'}")
    print(f"orphans    {len(orphans)} action(s) with unknown outcome" + ("  <-- run: recover --mark" if orphans else ""))
    return 0


# -- policy tools -------------------------------------------------------------------------------------------------- #

def cmd_policy_lint(a: argparse.Namespace) -> int:
    e = _engine(a.file)
    w = e.lint()
    print(f"policy {e.policy_id}: valid; {len(e.policy.get('rules', []))} rules, {len(e.policy.get('tiers', {}))} mapped actions")
    for x in w:
        print(f"WARNING: {x}")
    return 1 if (w and a.strict) else 0


def cmd_policy_test(a: argparse.Namespace) -> int:
    fails = run_cases(_engine(a.file), _load_json(a.cases))
    for f in fails:
        print(f"FAIL {f}")
    print(f"{len(fails)} failing case(s)" if fails else "all cases pass")
    return 1 if fails else 0


def cmd_policy_diff(a: argparse.Namespace) -> int:
    cli = _helpers()
    old, new = _engine(a.old), _engine(a.new)
    cases: List[Dict[str, Any]] = []
    if a.cases:
        cases += _load_json(a.cases)
    if a.use_store:
        cases += list(cases_from_store(Store(a.store), since_ns=cli._since(a.since), limit=a.limit))
    if not cases:
        print("no cases: pass --cases FILE and/or --use-store", file=sys.stderr)
        return 1
    d = diff_policies(old, new, cases)
    print(f"policy-diff {old.policy_id} -> {new.policy_id}: {d.n} distinct checks | {len(d.loosened)} LOOSENED | "
          f"{len(d.tightened)} tightened | {len(d.tier_changed)} tier-only changes")
    for title, rows in (("LOOSENED", d.loosened), ("tightened", d.tightened), ("tier changed", d.tier_changed)):
        for r in rows[:25]:
            print(f"  {title}: {r['action']} {canonical(r['args'])[:60]}  {r['old']} -> {r['new']}  (x{r['count']})")
    return 2 if d.loosened else 0


def cmd_policy_explain(a: argparse.Namespace) -> int:
    for line in _engine(a.file).explain(a.action, json.loads(a.args), json.loads(a.ctx)):
        print(line)
    return 0


# -- live audit / drift / curation / recovery / rotation ------------------------------------------------------------ #

def cmd_check(a: argparse.Namespace) -> int:
    cli = _helpers()
    st = Store(a.store)
    n = bad = 0
    for h in st.list_traces(since_ns=cli._since(a.since), limit=a.limit, newest_first=False):
        if "replay" in h.tags:
            continue
        tr = st.load_trace(h.trace_id)
        n += 1
        for v in check_trace(tr, live_invariants(a.max_confirm_age)):
            bad += 1
            print(f"{tr.trace_id} [{v.invariant}] {v.detail}")
    print(f"checked {n} traces: {bad} violation(s)")
    return 1 if bad else 0


def cmd_drift(a: argparse.Namespace) -> int:
    st = Store(a.store)
    base = analytics.window_stats(st, *analytics.parse_window(a.base))
    cand = analytics.window_stats(st, *analytics.parse_window(a.cand))
    findings = analytics.compare(base, cand, decision_shift_pp=a.decision_shift, latency_pct=a.latency_pct,
                                 min_traces=a.min_traces)
    if a.json:
        print(json.dumps({"base": base.to_dict(), "candidate": cand.to_dict(),
                          "findings": [f.__dict__ for f in findings]}, indent=2, default=str))
    else:
        print(f"base      {base.traces} traces | decisions {base.decisions} | confirm approval {base.approval_rate:.0%}")
        print(f"candidate {cand.traces} traces | decisions {cand.decisions} | confirm approval {cand.approval_rate:.0%}")
        for f in findings:
            print(f"{f.level.upper():<6} {f.message}")
        if not findings:
            print("no drift beyond thresholds")
    return 1 if any(f.level == "alert" for f in findings) else 0


def cmd_curate(a: argparse.Namespace) -> int:
    cli = _helpers()
    r = curate(Store(a.store), since_ns=cli._since(a.since), max_total=a.max, per_signature=a.per_signature,
               label=a.label, include_failed=a.include_failed, dry_run=a.dry_run)
    print(f"considered {r.considered} traces in {len(r.groups)} behaviour groups; "
          f"{'would select' if a.dry_run else 'labelled'} {len(r.selected)} as '{a.label}'")
    return 0


def cmd_recover(a: argparse.Namespace) -> int:
    rec = recover(Store(a.store), stale_s=a.stale_s, mark=a.mark)
    if rec.clean:
        print("clean: no action was left unfinished")
        return 0
    for line in rec.lines():
        print(line)
    if a.mark:
        print(f"labelled {len(rec.marked)} trace(s) incident:orphaned_action; closed {len(rec.closed)} unfinished trace(s)")
    print("review with: audit --orphans / undo-plan <trace_id>; acknowledge with: label <trace_id> incident:reviewed")
    return 1


def cmd_rotate(a: argparse.Namespace) -> int:
    cli = _helpers()
    keep_s = None
    if a.keep:
        since = cli._since(a.keep)
        import time
        keep_s = (time.time_ns() - since) / 1e9
    info = Store(a.store).rotate_to(a.to, keep_s=keep_s, keep_labels=a.keep_label)
    print(f"rotated {a.store} -> {a.to}: kept {info['kept_traces']} traces ({info['imported_events']} events, "
          f"{info['imported_blobs']} blobs), dropped {info['dropped_traces']}")
    print(f"old chain head {info['prev_head']}\nnew chain head {info['new_head']}\n"
          f"old store left untouched: archive it, point Jarvis at {a.to}, keep {a.store}.key with both.")
    return 0


def cmd_recall_eval(a: argparse.Namespace) -> int:
    cli = _helpers()
    retriever, _ = cli._load_callable(a.retriever)
    rep = recall_eval(Store(a.store), retriever, since_ns=cli._since(a.since), limit=a.limit, k=a.k, name=a.name,
                      min_jaccard=a.min_overlap)
    print(rep.summary())
    for c in rep.worst(5):
        print(f"  worst: {c.trace_id} {c.name} jaccard={c.jaccard:.2f} recall={c.recall:.2f} (base {c.n_base} / cand {c.n_cand})")
    for c in [c for c in rep.cases if c.error][:5]:
        print(f"  error: {c.trace_id} {c.name}: {c.error}")
    return 0 if rep.passed else 1


def cmd_undo(a: argparse.Namespace) -> int:
    cli = _helpers()
    st = Store(a.store)
    if not st.trace_exists(a.trace_id):
        print(f"no such trace {a.trace_id}", file=sys.stderr)
        return 1
    if a.apply and not a.executor:
        print("--apply needs --executor module:factory (factory(store) -> Executor wired with your registry/policy)",
              file=sys.stderr)
        return 1
    if a.executor:
        factory, _ = cli._load_callable(a.executor)
        ex = factory(st) if inspect.signature(factory).parameters else factory()
    else:
        from .actions import ActionRegistry, Executor
        ex = Executor(ActionRegistry(), PolicyEngine())
    try:
        res = UndoRunner(st, ex).run(a.trace_id, apply=a.apply, force=a.force)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    for s in res["steps"]:
        print(f"{'APPLIED ' if res['applied'] else 'planned '}{s['status']:<14} {s['action']} {canonical(s['args'])[:70]}"
              f"   (undoes {s['undoes']}{', may not have applied' if s['maybe_applied'] else ''})")
    for s in res["irreversible"]:
        print(f"!  irreversible: {s['action']} {canonical(s['args'])[:60]} (tier {s['tier']}, {s['outcome']})"
              f"{'  snapshot recorded' if s['snapshot'] else ''}")
    if not res["steps"] and not res["irreversible"]:
        print("nothing to undo")
    if res["applied"]:
        failed = [s for s in res["steps"] if s["status"] != "ok"]
        print(f"undo trace {res['undo_trace_id']}: {len(res['steps']) - len(failed)} ok, {len(failed)} not applied")
        return 1 if failed else 0
    print("dry run: add --apply (with --executor) to execute")
    return 0


def register(sub: Any) -> None:
    def sp(name: str, fn: Callable[..., int], help_: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--store", default=argparse.SUPPRESS)
        s.set_defaults(fn=fn)
        return s

    def pausefile(s: argparse.ArgumentParser) -> None:
        s.add_argument("--file", help="pause file (default $JARVIS_PAUSE_FILE or ~/.jarvis/PAUSE)")

    s = sp("pause", cmd_pause, "engage the kill switch: deny every action above tier 0")
    pausefile(s); s.add_argument("--reason", default="")
    s = sp("resume", cmd_resume, "release the kill switch"); pausefile(s)
    s = sp("status", cmd_status, "store health, pause state, orphaned actions"); pausefile(s)

    s = sp("policy-lint", cmd_policy_lint, "validate a policy file and warn about risky settings")
    s.add_argument("file"); s.add_argument("--strict", action="store_true", help="exit 1 on warnings")
    s = sp("policy-test", cmd_policy_test, "run table-driven cases against a policy")
    s.add_argument("file"); s.add_argument("cases")
    s = sp("policy-diff", cmd_policy_diff, "diff two policies over recorded/explicit checks; exit 2 if anything loosened")
    s.add_argument("--old", required=True); s.add_argument("--new", required=True)
    s.add_argument("--cases"); s.add_argument("--use-store", action="store_true", help="include every recorded policy check")
    s.add_argument("--since"); s.add_argument("--limit", type=int)
    s = sp("policy-explain", cmd_policy_explain, "show which rules fire for one check")
    s.add_argument("file"); s.add_argument("--action", required=True)
    s.add_argument("--args", default="{}"); s.add_argument("--ctx", default="{}")

    s = sp("check", cmd_check, "audit live traces against the safety invariants (exit 1 on violations)")
    s.add_argument("--since"); s.add_argument("--limit", type=int); s.add_argument("--max-confirm-age", type=float, default=120.0)
    s = sp("drift", cmd_drift, "compare behaviour between two time windows")
    s.add_argument("--base", default="14d..7d"); s.add_argument("--cand", default="7d")
    s.add_argument("--decision-shift", type=float, default=10.0); s.add_argument("--latency-pct", type=float, default=25.0)
    s.add_argument("--min-traces", type=int, default=10); s.add_argument("--json", action="store_true")
    s = sp("curate", cmd_curate, "label a diverse set of recorded turns as the golden suite")
    s.add_argument("--since"); s.add_argument("--max", type=int, default=100); s.add_argument("--per-signature", type=int, default=2)
    s.add_argument("--label", default="golden"); s.add_argument("--include-failed", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s = sp("recover", cmd_recover, "list (and with --mark, flag) actions left unfinished by a crash")
    s.add_argument("--stale-s", type=float, default=300.0); s.add_argument("--mark", action="store_true")
    s = sp("rotate", cmd_rotate, "start a fresh store carrying forward recent + golden + incident traces")
    s.add_argument("--to", required=True); s.add_argument("--keep", help="keep traces newer than e.g. 30d")
    s.add_argument("--keep-label", action="append", default=["golden"])
    s = sp("recall-eval", cmd_recall_eval, "re-run a retriever over recorded recall queries and measure overlap")
    s.add_argument("--retriever", required=True, help="module:function taking the recorded kwargs (q, k, ...)")
    s.add_argument("--k", type=int); s.add_argument("--name"); s.add_argument("--since"); s.add_argument("--limit", type=int)
    s.add_argument("--min-overlap", type=float, default=0.0)
    s = sp("undo", cmd_undo, "plan (default) or apply (--apply) the recorded undo steps of a trace")
    s.add_argument("trace_id"); s.add_argument("--executor", help="module:factory; factory(store) -> Executor")
    s.add_argument("--apply", action="store_true"); s.add_argument("--force", action="store_true")
