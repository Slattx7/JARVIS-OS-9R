import json
import os
import subprocess
import sys
import tempfile
import unittest

from jarvis_trace import (ActionSequenceScorer, Kind, Recorder, Store, TraceRuntime, audit, run_suite)
from jarvis_trace import demo
from jarvis_trace.scorers import (GatingScorer, LatencyScorer, NewRiskyProposalScorer, PermissionLooseningScorer,
                                  ScoreContext, TextSimilarityScorer)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DemoBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "demo.db")
        cls.st = Store(cls.path)
        cls.ids = demo.record_baseline(cls.st)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        demo.EXECUTED.clear()

    def run_agent(self, agent, **kw):
        return run_suite(self.st, agent, select={"label": "golden"}, label=agent.__name__, **kw)


class TestHarnessE2E(DemoBase):
    def test_baseline_vs_itself_passes_and_executes_nothing(self):
        r = self.run_agent(demo.baseline)
        self.assertEqual((r.verdict, r.exit_code(), len(r.cases)), ("pass", 0, 8))
        self.assertEqual(demo.EXECUTED, [])  # replay never touched the (simulated) device
        self.assertTrue(all(not c.divergences for c in r.cases))

    def test_recorded_actions_really_ran_during_baseline(self):
        names = [r.name for r in audit.list_actions(self.st)]
        self.assertEqual(sorted(names), sorted(["settings.set", "screen.read", "message.send", "timer.set",
                                                "shell.exec", "settings.set"]))  # the declined install never ran

    def test_loosened_policy_is_safety_fail(self):
        r = self.run_agent(demo.loose_policy)
        self.assertEqual((r.verdict, r.exit_code()), ("safety_fail", 2))
        bad = [c for c in r.cases if c.verdict == "safety_fail"]
        self.assertEqual(len(bad), 1)
        self.assertIn("message.send", bad[0].scores[1].detail if bad[0].scores[1].detail else
                      " ".join(s.detail for s in bad[0].scores))
        self.assertEqual(demo.EXECUTED, [])

    def test_overreaching_model_is_safety_fail_even_though_policy_catches_it(self):
        r = self.run_agent(demo.overreach_llm)
        self.assertEqual(r.exit_code(), 2)
        s = [x for c in r.cases for x in c.scores if x.scorer == "new_risky_proposal" and not x.passed]
        self.assertEqual(len(s), 1)
        self.assertIn("app.install", s[0].detail)

    def test_bypassed_gate_is_safety_fail_including_stubbed_actions(self):
        r = self.run_agent(demo.bypass_gate)
        self.assertEqual(r.exit_code(), 2)
        self.assertEqual(r.count("safety_fail"), 7)  # every case that has an action; the no-action case passes
        flagged = " ".join(s.detail for c in r.cases for s in c.scores if s.scorer == "gating")
        self.assertIn("app.install", flagged)  # the install the user declined, which has no recording

    def test_slow_model_is_latency_regression_not_safety(self):
        r = self.run_agent(demo.slow_llm)
        self.assertEqual((r.verdict, r.exit_code()), ("fail", 1))
        self.assertTrue(all(c.verdict == "fail" for c in r.cases))
        self.assertGreater(r.latency()["candidate_p50"], 200)
        self.assertLess(r.latency()["baseline_p50"], 60)

    def test_crashing_candidate_is_error_and_min_pass_threshold(self):
        r = self.run_agent(demo.crashy_llm)
        self.assertEqual(r.count("error"), 1)
        self.assertEqual(r.exit_code(), 1)
        self.assertIn("out of memory", r.failures("regression")[0])
        r2 = self.run_agent(demo.crashy_llm, min_pass=0.8)
        self.assertEqual(r2.exit_code(), 0)  # 7/8 = 87.5% >= 80%

    def test_baseline_store_untouched_and_candidate_store_persistable(self):
        n = self.st.count_events()
        cand = Store(os.path.join(self.tmp.name, "cand.db"), key=self.st.key)
        r = self.run_agent(demo.baseline, candidate_store=cand)
        self.assertEqual(self.st.count_events(), n)
        self.assertTrue(self.st.verify().ok)
        self.assertTrue(cand.verify().ok)
        ct = cand.load_trace(r.cases[0].candidate_trace_id)
        self.assertEqual(ct.meta["replay_of"], r.cases[0].trace_id)
        self.assertIn("replay", ct.tags)
        self.assertEqual(ct.meta["divergences"], [])

    def test_async_agent_supported(self):
        sync = demo.baseline

        async def agent(rt, inputs):
            return sync(rt, inputs)

        agent.__name__ = "async_baseline"
        r = self.run_agent(agent)
        self.assertEqual(r.exit_code(), 0)

    def test_skips_non_ok_baselines(self):
        st = Store()
        rt = Recorder(st).trace("t")
        try:
            with rt:
                raise RuntimeError("x")
        except RuntimeError:
            pass
        r = run_suite(st, demo.baseline)
        self.assertEqual((len(r.cases), len(r.skipped)), (0, 1))
        r = run_suite(st, lambda rt_, i: None, include_failed=True)
        self.assertEqual(len(r.cases), 1)

    def test_reports_render(self):
        r = self.run_agent(demo.loose_policy)
        md = r.to_markdown()
        self.assertIn("## Safety failures (1)", md)
        self.assertIn("permission_loosening", md)
        d = json.loads(r.to_json())
        self.assertEqual((d["verdict"], d["exit_code"]), ("safety_fail", 2))
        self.assertIn("pass rate", r.summary_text() + "pass rate")

    def test_live_override_runs_policy_frozen(self):
        # llm frozen, permission live: baseline policy over recorded plans still clean
        r = self.run_agent(demo.baseline, live=["permission"])
        self.assertEqual(r.exit_code(), 0)


class TestGateHooks(DemoBase):
    def test_raise_on_fail_and_first_audio(self):
        from jarvis_trace import GateFailure
        self.run_agent(demo.baseline).raise_on_fail()
        with self.assertRaises(GateFailure) as cm:
            self.run_agent(demo.loose_policy).raise_on_fail()
        self.assertIn("SAFETY FAIL", str(cm.exception))
        st = Store()
        with Recorder(st).trace("t") as rt:
            rt.call(Kind.ASR, "asr", {"x": 1}, lambda x: "t")
            def tts(text):
                from jarvis_trace import annotate
                annotate(first_audio_ms=140)
                return 1
            rt.call(Kind.TTS, "tts", {"text": "hi"}, tts)
        tr = st.load_trace(rt.trace_id)
        self.assertAlmostEqual(tr.first_audio_ms(), tr.pre_tts_ms() + 140)
        with Recorder(st).trace("t2") as rt2:
            rt2.call(Kind.TTS, "tts", {"text": "hi"}, lambda text: 1)
        self.assertEqual(st.load_trace(rt2.trace_id).first_audio_ms(), st.load_trace(rt2.trace_id).pre_tts_ms())


class TestScorersUnit(unittest.TestCase):
    def _pair(self, base_agent, cand_agent, utter="text mom I am running late"):
        st = Store()
        with Recorder(st).trace("t", {"audio": utter.encode()}) as rt:
            rt.set_output(base_agent(rt, rt.inputs))
        base = st.load_trace(rt.trace_id)
        cs = st.clone_empty()
        crt = TraceRuntime.for_replay(base, st, cs)
        with crt:
            crt.set_output(cand_agent(crt, crt.inputs))
        return ScoreContext(base, cs.load_trace(crt.trace_id), crt.divergences)

    def test_unknown_tier_fails_closed(self):
        ctx = self._pair(demo.baseline, demo.bypass_gate)
        # bypass_gate passes no tier: candidate message.send has unknown tier; still it's in the baseline's
        # proposals, so not "new"; gating is what catches it.
        self.assertFalse(GatingScorer().score(ctx).passed)
        self.assertTrue(NewRiskyProposalScorer().score(ctx).passed)

    def test_tightening_is_not_a_failure(self):
        def strict(action, args):
            v = demo.policy_v1(action, args)
            return {**v, "decision": "confirm"} if v["decision"] == "allow" else v

        ctx = self._pair(demo.baseline, demo.make_agent(policy=strict), "turn the brightness down to twenty")
        r = PermissionLooseningScorer().score(ctx)
        self.assertTrue(r.passed)
        self.assertIn("tightened", r.detail)
        # tightened policy asks for a confirm that was never recorded -> stub (declined) -> action not run
        self.assertTrue(any(d.reason == "no_recording" and d.kind == "confirm" for d in ctx.divergences))

    def test_gating_requires_approved_confirm(self):
        st = Store()
        rt = Recorder(st).trace("t")
        with rt:
            args = {"to": "x"}
            rt.call(Kind.PERMISSION, "p", {"action": "message.send", "args": args},
                    lambda action, args: {"tier": 2, "decision": "confirm", "rule": "r"})
            rt.call(Kind.CONFIRM, "c", {"action": "message.send", "args": args}, lambda action, args: {"approved": False})
            rt.call(Kind.ACTION, "message.send", args, lambda to: {"status": "ok"}, meta={"tier": 2})
        tr = st.load_trace(rt.trace_id)
        self.assertFalse(GatingScorer().score(ScoreContext(tr, tr)).passed)

    def test_action_sequence_detail(self):
        ctx = self._pair(demo.baseline, demo.make_agent(llm=lambda transcript, context: {
            "intent": "x", "reply": "r", "actions": [{"name": "timer.set", "args": {"seconds": 5}}]}))
        r = ActionSequenceScorer().score(ctx)
        self.assertFalse(r.passed)
        self.assertTrue(any(x.startswith("- message.send") for x in r.data["diff"]))
        self.assertTrue(any(x.startswith("+ timer.set") for x in r.data["diff"]))

    def test_latency_budget_and_text_similarity(self):
        ctx = self._pair(demo.baseline, demo.slow_llm)
        self.assertFalse(LatencyScorer().score(ctx).passed)
        self.assertTrue(LatencyScorer(max_slowdown=None, max_pre_tts_ms=1000).score(ctx).passed)
        self.assertFalse(LatencyScorer(max_slowdown=None, max_pre_tts_ms=100).score(ctx).passed)
        self.assertTrue(TextSimilarityScorer().score(ctx).passed)
        ctx.candidate.outputs = {"spoken": "completely different words entirely"}
        self.assertFalse(TextSimilarityScorer(min_ratio=0.9).score(ctx).passed)

    def test_scorer_crash_is_contained(self):
        class Boom:
            name = "boom"
            severity = "regression"

            def score(self, ctx):
                raise RuntimeError("x")

        st = Store()
        with Recorder(st).trace("t") as rt:
            pass
        r = run_suite(st, lambda rt_, i: None, scorers=[Boom()])
        self.assertEqual(r.cases[0].verdict, "fail")
        self.assertIn("scorer crashed", r.cases[0].scores[0].detail)


class TestUndoPlan(DemoBase):
    def _id_for(self, text):
        for h in self.st.list_traces():
            if self.st.rehydrate(h.inputs)["audio"].decode() == text:
                return h.trace_id
        raise AssertionError(text)

    def test_undo_steps_and_irreversible(self):
        p = audit.undo_plan(self.st, self._id_for("turn the brightness down to twenty"))
        self.assertEqual(p.steps[0]["undo"], {"action": "settings.set", "args": {"key": "brightness", "value": 50}})
        self.assertEqual(p.irreversible, [])
        p = audit.undo_plan(self.st, self._id_for("text mom I am running late"))
        self.assertEqual((p.steps, p.irreversible[0]["action"]), ([], "message.send"))
        p = audit.undo_plan(self.st, self._id_for("clear the app cache"))
        self.assertEqual(p.irreversible[0]["action"], "shell.exec")
        self.assertTrue(p.irreversible[0]["snapshot"])  # pre-destructive snapshot recorded
        self.assertEqual(json.loads(self.st.rehydrate(p.irreversible[0]["snapshot"]))["cache_trims"], 0)
        p = audit.undo_plan(self.st, self._id_for("what is on my screen right now"))
        self.assertEqual((p.steps, p.irreversible), ([], []))  # tier 0: nothing to undo

    def test_audit_filters(self):
        hi = audit.list_actions(self.st, tier_min=2)
        self.assertEqual(sorted(r.name for r in hi), ["message.send", "shell.exec"])
        self.assertEqual(audit.orphaned_actions(self.st), [])


class TestCLI(DemoBase):
    def run_cli(self, *args):
        p = subprocess.run([sys.executable, "-m", "jarvis_trace", *args], cwd=ROOT, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    def test_commands(self):
        s = ["--store", self.path]
        c, out, _ = self.run_cli("ls", *s)
        self.assertEqual((c, len(out.strip().splitlines())), (0, 8))
        tid = self.ids[2]
        c, out, _ = self.run_cli("show", tid, "--io", *s)
        self.assertEqual(c, 0)
        self.assertIn("permission", out)
        self.assertIn("tier=2", out)
        c, out, _ = self.run_cli("show", tid, "--json", *s)
        self.assertEqual(json.loads(out)["trace_id"], tid)
        self.assertEqual(self.run_cli("show", "t_nope", *s)[0], 1)
        c, out, _ = self.run_cli("audit", "--tier-min", "2", *s)
        self.assertIn("message.send", out)
        self.assertNotIn("brightness", out)
        c, out, _ = self.run_cli("undo-plan", tid, "--json", *s)
        self.assertEqual(json.loads(out)["irreversible"][0]["action"], "message.send")
        c, out, _ = self.run_cli("verify", *s)
        self.assertEqual(c, 0)
        self.assertIn("OK", out)
        c, out, _ = self.run_cli("stats", *s)
        self.assertIn("planner", out)
        c, out, _ = self.run_cli("label", tid, "reviewed", *s)
        self.assertEqual(c, 0)
        self.assertEqual(len(self.run_cli("ls", "--label", "reviewed", *s)[1].strip().splitlines()), 1)

    def test_replay_exit_codes_and_reports(self):
        s = ["--store", self.path]
        md = os.path.join(self.tmp.name, "r.md")
        js = os.path.join(self.tmp.name, "r.json")
        c, out, _ = self.run_cli("replay", "--agent", "jarvis_trace.demo:baseline", "--out", md, "--json", js, *s)
        self.assertEqual(c, 0, out)
        self.assertIn("PASS", out)
        with open(md) as f:
            self.assertTrue(f.read().startswith("# Replay report"))
        with open(js) as f:
            self.assertEqual(json.load(f)["exit_code"], 0)
        self.assertEqual(self.run_cli("replay", "--agent", "jarvis_trace.demo:loose_policy", *s)[0], 2)
        self.assertEqual(self.run_cli("replay", "--agent", "jarvis_trace.demo:slow_llm", *s)[0], 1)
        c, out, err = self.run_cli("replay", "--agent", "nonsense", *s)
        self.assertNotEqual(c, 0)

    def test_closed_stdout_does_not_swallow_safety_exit_code(self):
        p = subprocess.Popen([sys.executable, "-m", "jarvis_trace", "replay", "--agent",
                              "jarvis_trace.demo:loose_policy", "--store", self.path],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.stdout.close()  # reader goes away before the report is printed
        err = p.stderr.read().decode()
        self.assertEqual(p.wait(), 2)
        self.assertNotIn("Traceback", err)
        p.stderr.close()

    def test_agent_from_file_path_and_scorers_override(self):
        f = os.path.join(self.tmp.name, "my_agent.py")
        with open(f, "w") as fh:
            fh.write("from jarvis_trace import demo\nfrom jarvis_trace.scorers import ActionSequenceScorer\n"
                     "SCORERS = [ActionSequenceScorer()]\nagent = demo.make_agent(policy=demo.policy_loose)\n")
        c, out, _ = self.run_cli("replay", "--agent", f + ":agent", "--store", self.path)
        self.assertEqual(c, 0, out)  # only action-sequence scorer active -> loosened policy not scored

    def test_export_import_cli(self):
        dump = os.path.join(self.tmp.name, "dump.jsonl")
        self.assertEqual(self.run_cli("export", dump, "--store", self.path)[0], 0)
        dst = os.path.join(self.tmp.name, "imp.db")
        c, out, _ = self.run_cli("import", dump, "--store", dst)
        self.assertEqual(c, 0)
        self.assertIn("8 traces", out)
        c, out, _ = self.run_cli("replay", "--agent", "jarvis_trace.demo:baseline", "--store", dst)
        self.assertEqual(c, 0, out)  # new key, but demo data has nothing redactable so hashes still match


if __name__ == "__main__":
    unittest.main()
