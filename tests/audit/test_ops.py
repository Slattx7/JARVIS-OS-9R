import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

from jarvis_trace import (Kind, Recorder, Store, TraceRuntime, demo, load_gate, GateError,
                          recall_call, recall_eval, run_suite)
from jarvis_trace.analytics import WindowStats, compare, parse_window, window_stats
from jarvis_trace.curate import curate, signature
from jarvis_trace.policy import DEFAULT_POLICY
from jarvis_trace.recall import RecallOverlapScorer
from jarvis_trace.recover import INCIDENT_LABEL, recover
from jarvis_trace.report_html import render_report_html, render_trace_html
from jarvis_trace.risk import RiskAdapter, fail_closed
from jarvis_trace.scorers import item_id, overlap_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY = 86400 * 10**9


def old_trace(st, tid, ts_ns, *, labels=(), unfinished_action=False, name="turn"):
    """Write a complete trace with explicit timestamps (the Recorder always stamps 'now')."""
    st.append("trace_start", tid, None, {"schema": 1, "name": name, "inputs": {"x": 1}, "tags": [], "env": {}, "meta": {}},
              ts_ns=ts_ns)
    st.append("span_start", tid, "s1", {"parent_id": None, "kind": "action" if unfinished_action else "custom",
                                        "name": "a", "inputs": {"b": st.put_blob(b"payload")}, "input_hash": "h",
                                        "meta": {"tier": 2}}, ts_ns=ts_ns + 1)
    if not unfinished_action:
        st.append("span_end", tid, "s1", {"dur_ns": 5, "status": "ok", "outputs": 1}, ts_ns=ts_ns + 2)
    st.append("trace_end", tid, None, {"dur_ns": 10, "status": "ok"}, ts_ns=ts_ns + 3)
    for l in labels:
        st.label(tid, l)


class TestRecall(unittest.TestCase):
    def test_item_id_and_overlap(self):
        self.assertEqual(item_id({"id": 7, "text": "x"}), "7")
        self.assertEqual(item_id("abc"), "abc")
        self.assertEqual(item_id({"zzz": 1}), '{"zzz":1}')
        m = overlap_metrics(["a", "b", "c"], ["b", "c", "d"])
        self.assertAlmostEqual(m["jaccard"], 0.5)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertEqual(m["top1_hit"], 0.0)
        self.assertEqual(overlap_metrics([], [])["jaccard"], 1.0)
        self.assertAlmostEqual(overlap_metrics(["a", "b"], ["a", "x"], k=1)["jaccard"], 1.0)  # truncated to top-1

    def test_recall_call_records_graph_version(self):
        st = Store()
        with Recorder(st).trace("t") as rt:
            out = recall_call(rt, lambda q, k, scope: [q] * k, "hello", 2, graph_version="g42", scope="work")
        s = st.load_trace(rt.trace_id).spans[0]
        self.assertEqual((out, s.kind, s.meta["graph_version"], s.inputs), (["hello", "hello"], "recall", "g42",
                                                                           {"q": "hello", "k": 2, "scope": "work"}))

    def _baseline(self, retr):
        st = Store()
        for q in ("alpha", "beta", "gamma"):
            with Recorder(st).trace("turn", {"q": q}) as rt:
                rt.set_output({"hits": recall_call(rt, retr, q, 3)})
        return st

    def test_overlap_scorer_with_live_recall_in_replay(self):
        base = lambda q, k: [{"id": f"{q}-{i}"} for i in range(k)]
        st = self._baseline(base)

        def agent_with(retr):
            def agent(rt, inputs):
                return {"hits": recall_call(rt, retr, inputs["q"], 3)}
            agent.__name__ = "a"
            return agent

        sc = [RecallOverlapScorer(min_jaccard=0.6)]
        same = run_suite(st, agent_with(base), live=["recall"], scorers=sc)
        self.assertEqual(same.exit_code(), 0)
        worse = run_suite(st, agent_with(lambda q, k: [{"id": f"{q}-0"}, {"id": "zz1"}, {"id": "zz2"}]),
                          live=["recall"], scorers=sc)
        self.assertEqual((worse.verdict, worse.exit_code()), ("fail", 1))
        self.assertIn("jaccard", worse.failures("regression")[0])
        frozen = run_suite(st, agent_with(lambda q, k: []), live=[], scorers=sc)  # recall frozen: bad retriever never runs
        self.assertEqual(frozen.exit_code(), 0)

    def test_recall_eval(self):
        base = lambda q, k: [{"id": f"{q}-{i}"} for i in range(k)]
        st = self._baseline(base)
        ok = recall_eval(st, base, min_jaccard=0.9)
        self.assertEqual((len(ok.cases), ok.mean_jaccard, ok.passed), (3, 1.0, True))
        half = recall_eval(st, lambda q, k: base(q, k)[:1] + [{"id": "x"}, {"id": "y"}], min_jaccard=0.9)
        self.assertFalse(half.passed)
        self.assertAlmostEqual(half.mean_jaccard, 0.2)
        self.assertEqual(len(half.worst(2)), 2)
        trunc = recall_eval(st, base, k=1)
        self.assertEqual(trunc.mean_jaccard, 1.0)
        boom = recall_eval(st, lambda q, k: 1 / 0)
        self.assertEqual((boom.errors, boom.passed), (3, False))
        self.assertEqual(len(recall_eval(st, base, name="nope").cases), 0)
        self.assertIn("FAIL", recall_eval(st, base, name="nope").summary())  # no cases is not a pass


class TestRisk(unittest.TestCase):
    def test_fail_closed_and_clipping(self):
        f = fail_closed(lambda **kw: 1 / 0)
        self.assertEqual(f()["risk"], 1.0)
        self.assertEqual(f()["label"], "scorer_error")
        self.assertEqual(fail_closed(lambda **kw: 7)()["risk"], 1.0)
        self.assertEqual(fail_closed(lambda **kw: -3)()["risk"], 0.0)
        self.assertEqual(fail_closed(lambda **kw: float("nan"))()["risk"], 1.0)
        self.assertEqual(fail_closed(lambda **kw: "garbage")()["risk"], 1.0)
        self.assertEqual(fail_closed(lambda **kw: {"risk": 0.3, "label": "x"})(), {"risk": 0.3, "label": "x"})

    def test_adapter(self):
        seen = []
        ad = RiskAdapter(lambda feats: seen.append(feats) or {"risk": 0.7, "reasons": ["r"]},
                         {"app.install": lambda a: {"package": a["pkg"]}, "net.*": lambda a: {"host": a["h"]}},
                         default_risk=0.1)
        self.assertEqual(ad(action="app.install", args={"pkg": "com.x"})["risk"], 0.7)
        self.assertEqual(seen[-1], {"package": "com.x"})
        self.assertEqual(ad(action="net.fetch", args={"h": "evil.example"})["risk"], 0.7)
        self.assertEqual(ad(action="timer.set", args={})["label"], "unscored")
        self.assertEqual(ad(action="timer.set", args={})["risk"], 0.1)
        bad = RiskAdapter(lambda feats: 0.2, {"a": lambda args: args["missing"]})  # extractor itself fails
        self.assertEqual(bad(action="a", args={})["risk"], 1.0)


class TestRecover(unittest.TestCase):
    def test_recover_flags_and_marks_once(self):
        st = Store()
        rt = Recorder(st).trace("crashy", {"x": 1})
        rt.start()
        rt._begin("action", "shell.exec", {"cmd": "pm trim-caches"}, {"tier": 3, "intent": "clean"})  # process "dies" here
        rt2 = Recorder(st).trace("fine")
        with rt2:
            rt2.call("custom", "c", {}, lambda: 1)
        r = recover(st, stale_s=0)
        self.assertEqual((len(r.orphaned), r.clean, r.marked), (1, False, []))
        self.assertIn("outcome unknown", r.lines()[0])
        self.assertIn("no undo information", r.lines()[0])
        r = recover(st, stale_s=0, mark=True)
        self.assertEqual(r.marked, [rt.trace_id])
        self.assertEqual(r.closed, [rt.trace_id])
        self.assertEqual(st.load_trace(rt.trace_id).status, "error")
        self.assertIn(INCIDENT_LABEL, st.load_trace(rt.trace_id).label_set())
        self.assertEqual(recover(st, stale_s=0, mark=True).marked, [])  # idempotent
        self.assertTrue(recover(Store(), stale_s=0).clean)
        self.assertFalse(recover(st, stale_s=3600).clean)  # closed by recovery: now a permanent, recorded orphan
        fresh = Store()
        live = Recorder(fresh).trace("live")
        live.start()
        live._begin("action", "x", {}, {"tier": 1})
        self.assertTrue(recover(fresh, stale_s=3600).clean)  # a fresh in-flight span is not yet an orphan


class TestCurate(unittest.TestCase):
    def setUp(self):
        self.st = Store()
        demo.record_baseline(self.st, label="")  # recorded, nothing labelled yet

    def test_prioritises_risky_and_diverse(self):
        r = curate(self.st, max_total=3, per_signature=1, label="gold", dry_run=True)
        self.assertFalse(r.applied)
        self.assertEqual(self.st.list_traces(label="gold"), [])
        sigs = {signature(self.st.load_trace(t)) for t in r.selected}
        flat = {e for s in sigs for e in s}
        self.assertIn(("act", "message.send", 2), flat)
        self.assertIn(("act", "shell.exec", 3), flat)
        self.assertIn(("declined",), flat)  # the declined install
        self.assertEqual(len(r.groups), 7)
        r = curate(self.st, max_total=100, per_signature=1, label="gold")
        self.assertEqual((len(r.selected), r.applied), (7, True))  # 8 turns, 2 share a signature
        self.assertEqual(len(self.st.list_traces(label="gold")), 7)
        again = curate(self.st, max_total=100, per_signature=5, label="gold")
        self.assertEqual(len(again.selected), 1)  # already-labelled traces are not picked twice

    def test_replay_and_failed_traces_are_excluded(self):
        bad = Recorder(self.st).trace("boom")
        try:
            with bad:
                raise RuntimeError("x")
        except RuntimeError:
            pass
        r = curate(self.st, max_total=100, per_signature=10, label="g2", dry_run=True)
        self.assertNotIn(bad.trace_id, r.selected)
        r = curate(self.st, max_total=100, per_signature=10, label="g2", include_failed=True, dry_run=True)
        self.assertIn(bad.trace_id, r.selected)


class TestAnalytics(unittest.TestCase):
    def test_parse_window(self):
        now = 100 * DAY
        self.assertEqual(parse_window("7d", now), (93 * DAY, now + 1))
        self.assertEqual(parse_window("14d..7d", now), (86 * DAY, 93 * DAY))
        for bad in ("abc", "7x", "1d..2d"):
            with self.assertRaises(ValueError):
                parse_window(bad, now)

    def test_window_stats_on_demo(self):
        st = Store()
        demo.record_baseline(st)
        w = window_stats(st, 0, time.time_ns() + DAY)
        self.assertEqual((w.traces, w.trace_errors, w.orphans), (8, 0, 0))
        self.assertEqual(w.decisions, {"allow": 4, "confirm": 3})
        self.assertEqual((w.confirms_asked, w.confirms_approved), (3, 2))
        self.assertAlmostEqual(w.approval_rate, 2 / 3)
        self.assertEqual(w.actions, {"settings.set": 2, "screen.read": 1, "message.send": 1, "timer.set": 1, "shell.exec": 1})
        self.assertEqual(w.tiers, {"0": 1, "1": 3, "2": 1, "3": 1})
        self.assertGreater(w.latency["llm/planner"]["p50"], 5)
        self.assertEqual(w.pre_tts["n"], 8)
        self.assertEqual(window_stats(st, 0, 1).traces, 0)

    def _w(self, **kw):
        w = WindowStats(0, 1, traces=kw.pop("traces", 100))
        for k, v in kw.items():
            setattr(w, k, v)
        return w

    def test_compare_flags_drift(self):
        base = self._w(decisions={"allow": 60, "confirm": 30, "deny": 10}, actions={"a": 5}, tiers={"1": 90, "3": 10},
                       confirms_asked=20, confirms_approved=18,
                       latency={"llm/p": {"n": 9, "p50": 100, "p95": 200}}, pre_tts={"n": 9, "p50": 300, "p95": 400})
        cand = self._w(decisions={"allow": 85, "confirm": 10, "deny": 5}, actions={"a": 5, "new.act": 3},
                       tiers={"1": 70, "3": 30}, confirms_asked=20, confirms_approved=9, orphans=2, trace_errors=10,
                       latency={"llm/p": {"n": 9, "p50": 100, "p95": 400}}, pre_tts={"n": 9, "p50": 300, "p95": 900})
        f = compare(base, cand)
        by = {x.metric: x for x in f}
        self.assertEqual(by["orphans"].level, "alert")
        self.assertEqual(by["error_rate"].level, "alert")
        self.assertEqual(by["decision.allow"].level, "alert")      # allow rate rising = looser in practice
        self.assertEqual(by["decision.confirm"].level, "warn")
        self.assertEqual(by["new_action"].level, "warn")
        self.assertEqual(by["confirm_approval"].level, "warn")
        self.assertEqual(by["tier3_share"].level, "warn")
        self.assertEqual(by["latency.llm/planner" if False else "latency.llm/p"].level, "warn")
        self.assertEqual(by["latency.pre_tts"].level, "alert")
        self.assertEqual([x.level for x in f], sorted((x.level for x in f), key=["alert", "warn", "info"].index))

    def test_compare_quiet_and_low_volume(self):
        a = self._w(decisions={"allow": 10}, tiers={"1": 10}, actions={"x": 1})
        self.assertEqual(compare(a, copy.deepcopy(a)), [])
        low = compare(self._w(traces=2), self._w(traces=3))
        self.assertEqual([x.metric for x in low], ["volume"])


class TestRotation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.old = Store(os.path.join(self.d, "old.db"))
        now = time.time_ns()
        old_trace(self.old, "t_old_plain", now - 90 * DAY)
        old_trace(self.old, "t_old_golden", now - 80 * DAY, labels=["golden"])
        old_trace(self.old, "t_old_incident", now - 70 * DAY, labels=["incident:orphaned_action"])
        old_trace(self.old, "t_old_orphan", now - 60 * DAY, unfinished_action=True)
        old_trace(self.old, "t_recent", now - 2 * DAY)
        self.now = now

    def tearDown(self):
        self.tmp.cleanup()

    def test_rotate_keeps_recent_golden_incident_and_unfinished(self):
        info = self.old.rotate_to(os.path.join(self.d, "new.db"), keep_s=30 * 86400, now_ns=self.now)
        self.assertEqual((info["kept_traces"], info["dropped_traces"]), (4, 1))
        new = Store(os.path.join(self.d, "new.db"))
        ids = {h.trace_id for h in new.list_traces()}
        self.assertEqual(ids, {"t_old_golden", "t_old_incident", "t_old_orphan", "t_recent"})
        rep = new.verify()
        self.assertTrue(rep.ok, rep.problems)
        self.assertEqual(rep.rotation["prev_head"], self.old.head())
        self.assertEqual(new.key, self.old.key)
        self.assertTrue(os.path.exists(os.path.join(self.d, "new.db.key")))
        self.assertEqual(new.rehydrate(new.load_trace("t_recent").spans[0].inputs["b"]), b"payload")  # blobs came along
        self.assertEqual(new.load_trace("t_old_golden").label_set(), {"golden"})
        self.assertEqual(len(self.old.list_traces()), 5)  # old store untouched
        self.assertTrue(self.old.verify().ok)

    def test_rotate_guards(self):
        with self.assertRaises(ValueError):
            self.old.rotate_to(self.old.path)
        target = os.path.join(self.d, "new.db")
        self.old.rotate_to(target, keep_s=1e9, now_ns=self.now)
        with self.assertRaises(ValueError):
            self.old.rotate_to(target, keep_s=1e9, now_ns=self.now)  # target no longer empty
        # a mismatched key file must not be adopted silently
        other = os.path.join(self.d, "other.db")
        with open(other + ".key", "w") as f:
            f.write("ab" * 32)
        with self.assertRaises(ValueError):
            self.old.rotate_to(other)


class TestGate(unittest.TestCase):
    def test_defaults_and_overrides(self):
        g = load_gate(None)
        self.assertEqual((g.min_pass, g.on_miss), (1.0, "positional"))
        self.assertEqual([s.name for s in g.scorers][:4], ["gating", "tier_integrity", "permission_loosening", "new_risky_proposal"])
        g = load_gate({"min_pass": 0.9, "live": ["llm", "action", "recall"], "latency": {"max_pre_tts_ms": 500},
                       "text_similarity": {"min_ratio": 0.5, "field": "text"}, "disable": ["divergence"],
                       "select": {"label": "golden"}})
        self.assertEqual((g.min_pass, g.live, g.select), (0.9, ["llm", "recall"], {"label": "golden"}))
        names = [s.name for s in g.scorers]
        self.assertNotIn("divergence", names)
        lat = [s for s in g.scorers if s.name == "latency"][0]
        self.assertEqual(lat.max_pre_tts_ms, 500)
        ts = [s for s in g.scorers if s.name == "text_similarity"][0]
        self.assertEqual((ts.field_name, ts.min_ratio), ("text", 0.5))

    def test_rejects_typos_and_safety_disable(self):
        for bad in ({"latncy": {}}, {"latency": {"bogus": 1}}, {"min_pass": 1.5}, {"on_miss": "guess"},
                    {"live": "llm"}, {"disable": ["nope"]}, {"disable": ["gating"]},
                    {"disable": ["permission_loosening"]}, [1, 2]):
            with self.assertRaises(GateError, msg=str(bad)):
                load_gate(bad)
        load_gate({"disable": ["gating"], "allow_disable_safety": True})  # explicit opt-out is allowed


class TestHtml(unittest.TestCase):
    def test_trace_view_and_escaping(self):
        st = Store()
        evil = '<script>alert(1)</script>"><img src=x onerror=alert(1)>'
        with Recorder(st).trace(evil, {"text": evil}) as rt:
            rt.call(Kind.LLM, evil, {"prompt": evil}, lambda prompt: {"reply": prompt})
            rt.call(Kind.ACTION, "fs.write", {"path": "/x"}, lambda path: {"status": "ok"}, meta={"tier": 2})
        html_out = render_trace_html(st.load_trace(rt.trace_id))
        self.assertNotIn("<script>", html_out)
        self.assertNotIn("<img src=x", html_out)
        self.assertIn("&lt;script&gt;", html_out)
        self.assertIn("Content-Security-Policy", html_out)
        self.assertIn("k-action", html_out)
        self.assertIn(">T2<", html_out)

    def test_replay_trace_gets_sequential_layout(self):
        st = Store()
        demo.record_baseline(st, n=1)
        base = st.load_trace(st.list_traces()[0].trace_id)
        cs = st.clone_empty()
        rt = TraceRuntime.for_replay(base, st, cs)
        with rt:
            demo.baseline(rt, rt.inputs)
        out = render_trace_html(cs.load_trace(rt.trace_id))
        self.assertIn("layout: sequential (replay)", out)
        self.assertIn("frozen", out)

    def test_report_html(self):
        st = Store()
        demo.record_baseline(st)
        rep = run_suite(st, demo.loose_policy, label="loose<script>")
        h = render_report_html(rep)
        self.assertIn("SAFETY FAIL", h)
        self.assertIn("permission_loosening", h)
        self.assertIn("message.send", h)
        self.assertNotIn("<script>", h)
        self.assertIn("loose&lt;script&gt;", h)
        self.assertIn("baseline actions", h)


class TestOpsCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = cls.tmp.name
        cls.path = os.path.join(cls.d, "demo.db")
        cls.st = Store(cls.path)
        cls.ids = demo.record_baseline(cls.st)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cli(self, *args, env=None):
        e = {**os.environ, **(env or {})}
        p = subprocess.run([sys.executable, "-m", "jarvis_trace", *args], cwd=ROOT, capture_output=True, text=True, env=e)
        return p.returncode, p.stdout, p.stderr

    def j(self, name, obj):
        path = os.path.join(self.d, name)
        with open(path, "w") as f:
            json.dump(obj, f)
        return path

    def test_status_pause_resume(self):
        pf = os.path.join(self.d, "PAUSE")
        c, out, _ = self.cli("status", "--store", self.path, "--file", pf)
        self.assertEqual(c, 0)
        self.assertIn("traces     8", out)
        self.assertIn("paused     no", out)
        c, out, _ = self.cli("pause", "--reason", "maintenance", "--file", pf)
        self.assertIn("PAUSED", out)
        self.assertIn("paused     YES: maintenance", self.cli("status", "--store", self.path, "--file", pf)[1])
        self.assertIn("resumed", self.cli("resume", "--file", pf)[1])
        self.assertIn("was not paused", self.cli("resume", "--file", pf)[1])

    def test_policy_tools(self):
        good = self.j("pol.json", DEFAULT_POLICY)
        c, out, _ = self.cli("policy-lint", good, "--strict")
        self.assertEqual(c, 0, out)
        bad = copy.deepcopy(DEFAULT_POLICY)
        bad["tier_decisions"]["2"] = "allow"
        c, out, _ = self.cli("policy-lint", self.j("bad.json", bad), "--strict")
        self.assertEqual(c, 1)
        self.assertIn("WARNING", out)
        self.assertEqual(self.cli("policy-lint", self.j("broken.json", {"rules": [{"id": "x"}]}))[0], 1)
        cases = self.j("cases.json", [{"action": "message.send", "expect": {"decision": "confirm"}},
                                      {"action": "shell.exec", "args": {"cmd": "rm -rf /"}, "expect": {"decision": "deny"}}])
        self.assertEqual(self.cli("policy-test", good, cases)[0], 0)
        failing = self.j("cases2.json", [{"action": "message.send", "expect": {"decision": "allow"}}])
        self.assertEqual(self.cli("policy-test", good, failing)[0], 1)
        c, out, _ = self.cli("policy-diff", "--old", good, "--new", self.j("bad.json", bad), "--use-store", "--store", self.path)
        self.assertEqual(c, 2)
        self.assertIn("LOOSENED", out)
        self.assertIn("message.send", out)
        self.assertEqual(self.cli("policy-diff", "--old", good, "--new", good, "--use-store", "--store", self.path)[0], 0)
        self.assertEqual(self.cli("policy-diff", "--old", good, "--new", good)[0], 1)  # nothing to compare
        c, out, _ = self.cli("policy-explain", good, "--action", "shell.exec", "--args", '{"cmd": "mkfs /dev/x"}')
        self.assertIn("MATCH", out)
        self.assertIn("deny", out)

    def test_check_drift_curate_recover_recall_eval(self):
        c, out, _ = self.cli("check", "--store", self.path)
        self.assertEqual((c, "0 violation" in out), (0, True), out)
        c, out, _ = self.cli("drift", "--store", self.path, "--min-traces", "1")
        self.assertEqual(c, 0, out)
        self.assertIn("candidate 8 traces", out)
        c, out, _ = self.cli("drift", "--store", self.path, "--json")
        self.assertEqual(json.loads(out)["candidate"]["traces"], 8)
        c, out, _ = self.cli("curate", "--store", self.path, "--dry-run", "--label", "g9", "--per-signature", "1")
        self.assertIn("would select 7", out)
        c, out, _ = self.cli("curate", "--store", self.path, "--dry-run", "--label", "g9")  # default: 2 per signature
        self.assertIn("would select 8", out)
        c, out, _ = self.cli("recover", "--store", self.path)
        self.assertEqual((c, "clean" in out), (0, True))
        c, out, _ = self.cli("recall-eval", "--retriever", "jarvis_trace.demo:_recall", "--min-overlap", "0.9",
                             "--store", self.path)
        self.assertEqual(c, 0, out)
        self.assertIn("PASS", out)
        empty = os.path.join(self.d, "empty_retriever.py")
        with open(empty, "w") as f:
            f.write("def r(q, k):\n    return []\n")
        c, out, _ = self.cli("recall-eval", "--retriever", empty + ":r", "--min-overlap", "0.9", "--store", self.path)
        self.assertEqual(c, 1, out)

    def test_rotate_and_verify_prev(self):
        new = os.path.join(self.d, "rot.db")
        c, out, _ = self.cli("rotate", "--to", new, "--keep", "30d", "--store", self.path)
        self.assertEqual(c, 0, out)
        self.assertIn("kept 8 traces", out)
        c, out, _ = self.cli("verify", "--store", new, "--prev", self.path)
        self.assertEqual(c, 0, out)
        self.assertIn("head matches", out)
        c, out, _ = self.cli("verify", "--store", new, "--prev", os.path.join(self.d, "rot.db"))  # wrong "previous"
        self.assertEqual(c, 1)
        self.assertIn("rotated", self.cli("status", "--store", new)[1])

    def test_html_outputs_and_gate_file(self):
        tid = self.ids[2]
        out_html = os.path.join(self.d, "t.html")
        self.assertEqual(self.cli("show", tid, "--html", out_html, "--store", self.path)[0], 0)
        with open(out_html) as f:
            self.assertIn("message.send", f.read())
        rep_html = os.path.join(self.d, "r.html")
        c, out, _ = self.cli("replay", "--agent", "jarvis_trace.demo:loose_policy", "--html", rep_html, "--store", self.path)
        self.assertEqual(c, 2)
        with open(rep_html) as f:
            self.assertIn("SAFETY FAIL", f.read())
        # gate: a loose latency budget makes the slow model pass, a tight one fails it
        ok = self.j("gate_ok.json", {"latency": {"max_slowdown": None, "max_pre_tts_ms": 5000}})
        self.assertEqual(self.cli("replay", "--agent", "jarvis_trace.demo:slow_llm", "--gate", ok, "--store", self.path)[0], 0)
        tight = self.j("gate_tight.json", {"latency": {"max_slowdown": None, "max_pre_tts_ms": 100}})
        self.assertEqual(self.cli("replay", "--agent", "jarvis_trace.demo:slow_llm", "--gate", tight, "--store", self.path)[0], 1)
        # min_pass from the gate file, overridden by the flag
        gate = self.j("gate_mp.json", {"min_pass": 0.8})
        self.assertEqual(self.cli("replay", "--agent", "jarvis_trace.demo:crashy_llm", "--gate", gate, "--store", self.path)[0], 0)
        self.assertEqual(self.cli("replay", "--agent", "jarvis_trace.demo:crashy_llm", "--gate", gate, "--min-pass", "1.0",
                                  "--store", self.path)[0], 1)
        bad = self.j("gate_bad.json", {"disable": ["gating"]})
        c, out, err = self.cli("replay", "--agent", "jarvis_trace.demo:baseline", "--gate", bad, "--store", self.path)
        self.assertEqual(c, 1)
        self.assertIn("refusing to disable", err)

    def test_undo_command_through_executor_factory(self):
        root = os.path.join(self.d, "undo_root")
        os.makedirs(root)
        factory = os.path.join(self.d, "wiring.py")
        with open(factory, "w") as f:
            f.write(
                "import os\n"
                "from jarvis_trace import Executor, PolicyEngine, ScriptedConfirmer\n"
                "from jarvis_trace.builtin_actions import build_registry\n"
                "def make_executor(store):\n"
                "    root = os.environ['JT_ROOT']\n"
                "    return Executor(build_registry([root], os.path.join(root, 'kv.json')), PolicyEngine(),\n"
                "                    ScriptedConfirmer({'*': True}))\n")
        env = {"JT_ROOT": root}
        import importlib.util
        spec = importlib.util.spec_from_file_location("wiring_t", factory)
        mod = importlib.util.module_from_spec(spec)
        os.environ["JT_ROOT"] = root
        try:
            spec.loader.exec_module(mod)
            st = Store(os.path.join(self.d, "undo.db"))
            ex = mod.make_executor(st)
            f1 = os.path.join(root, "n.txt")
            with Recorder(st).trace("t") as rt:
                ex.run(rt, "fs.write", {"path": f1, "content": "hi"}, intent="note")
                ex.run(rt, "kv.set", {"key": "a", "value": 1}, intent="set")
        finally:
            del os.environ["JT_ROOT"]
        self.assertTrue(os.path.exists(f1))
        c, out, _ = self.cli("undo", rt.trace_id, "--store", st.path, "--executor", factory + ":make_executor", env=env)
        self.assertEqual(c, 0, out)
        self.assertIn("planned", out)
        self.assertIn("dry run", out)
        self.assertTrue(os.path.exists(f1))
        self.assertEqual(self.cli("undo", rt.trace_id, "--store", st.path, "--apply")[0], 1)  # needs --executor
        c, out, _ = self.cli("undo", rt.trace_id, "--store", st.path, "--apply", "--executor", factory + ":make_executor", env=env)
        self.assertEqual(c, 0, out)
        self.assertFalse(os.path.exists(f1))
        c, out, err = self.cli("undo", rt.trace_id, "--store", st.path, "--apply", "--executor", factory + ":make_executor", env=env)
        self.assertEqual(c, 1)
        self.assertIn("already labeled 'undone'", err)


if __name__ == "__main__":
    unittest.main()
