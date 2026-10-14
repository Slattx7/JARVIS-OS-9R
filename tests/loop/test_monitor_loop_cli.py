import contextlib, io, json, sys, unittest
from pathlib import Path

from common import O, T, tmpdir, write_jsonl
from jarvis_loop import GateConfig, Loop, LoopConfig, Outcome, Registry, Trace
from jarvis_loop.cli import main as cli
from jarvis_loop.demo import _live_traces, run_demo
from jarvis_loop.factory import build_loop
from jarvis_loop.monitor import MonitorConfig, evaluate
from jarvis_loop.sim import DEMO_TIERS, SimReplayer, make_demo_traces
from jarvis_loop.trainer import DryRunTrainer


def TR(model, n, succ=1.0, viol=0, t0=1000.0):
    k = round(n * succ)
    return [Trace.from_dict({"id": f"{model}-{i}", "task": f"t{i}", "model": model, "ts": t0 + i,
                             "success": i < k, "violations": 1 if i < viol else 0}) for i in range(n)]


class TestMonitor(unittest.TestCase):
    def test_violation_rules(self):
        self.assertEqual(evaluate(TR("n", 50, viol=1), []).action, "rollback")                 # no baseline: any violation
        self.assertEqual(evaluate(TR("n", 50, viol=1), TR("p", 100, viol=10)).action, "ok")    # below prev rate
        self.assertEqual(evaluate(TR("n", 50, viol=5), TR("p", 100, viol=1)).action, "rollback")

    def test_success_rules(self):
        self.assertEqual(evaluate(TR("n", 60, 0.6), TR("p", 100, 0.9)).action, "rollback")
        self.assertEqual(evaluate(TR("n", 40, 0.85), TR("p", 100, 0.90)).action, "ok")         # within noise
        self.assertEqual(evaluate(TR("n", 10), TR("p", 100)).action, "insufficient")

    def test_window_uses_recent(self):
        old_bad = TR("n", 100, 0.0, t0=0)
        recent_good = TR("n", 60, 1.0, t0=5000)
        self.assertEqual(evaluate(old_bad + recent_good, TR("p", 100, 0.9), MonitorConfig(window=60)).action, "ok")


def mk(tmp, n=700, init=True, trainer=None, replayer=None, replay_params=None, **over):
    tp = tmp / "traces.jsonl"
    make_demo_traces(tp, n=n, seed=1, model="v0")
    params = {"skill": {"v0": 0.55, "v0001": 0.80, "v0002": 0.90}, "violators": []}
    params.update(replay_params or {})
    kw = dict(root=str(tmp / "data"), traces=str(tp), base_model="base", tiers=DEMO_TIERS, min_train=50,
              min_new_examples=20, max_cycles_per_day=10, trainer={"kind": "dry"},
              replay={"kind": "sim", "params": params})
    kw.update(over)
    cfg = LoopConfig(**kw)
    loop = Loop(cfg, trainer or DryRunTrainer(), replayer or SimReplayer(**params))
    if init:
        loop.init("v0")
    return cfg, loop, tp


class Scripted:
    def __init__(self, fn):
        self.fn, self.calls = fn, {}

    def prepare(self, ref):
        pass

    def replay(self, ref, t):
        self.calls[ref.version] = self.calls.get(ref.version, 0) + 1
        return self.fn(ref.version, t)


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.d, self.p = tmpdir()

    def tearDown(self):
        self.d.cleanup()

    def test_promote_cycle_and_artifacts(self):
        cfg, loop, tp = mk(self.p)
        r = loop.cycle()
        self.assertEqual((r.status, r.version, loop.reg.current()), ("promoted", "v0001", "v0001"))
        run = Path(cfg.runs_dir) / "v0001"
        self.assertTrue((run / "data" / "train.jsonl").exists() and (run / "report.json").exists())
        self.assertEqual(json.loads((run / "report.json").read_text())["status"], "promoted")
        self.assertGreater(json.loads(Path(cfg.state_path).read_text())["last_train_count"], 100)
        self.assertTrue(any(e["event"] == "cycle" and e["decision"] == "promoted" for e in loop.reg.history()))
        r2 = loop.cycle()                                                # no new data -> nothing to do
        self.assertEqual(r2.status, "skipped")
        self.assertIn("new examples", r2.reason)
        self.assertEqual(loop.reg.versions(), ["v0", "v0001"])
        self.assertEqual(loop.status()["current"], "v0001")

    def test_violating_candidate_rejected(self):
        cfg, loop, tp = mk(self.p, replay_params={"violators": ["v0002"]})
        self.assertEqual(loop.cycle().status, "promoted")
        make_demo_traces(tp, n=300, seed=2, model="v0001")
        r = loop.cycle()
        self.assertEqual((r.status, loop.reg.current()), ("rejected", "v0001"))
        self.assertIn("violation", r.reason)
        self.assertIn("v0002", loop.reg.versions())                       # archived, never deleted

    def test_equal_candidate_held(self):
        cfg, loop, tp = mk(self.p, replay_params={"skill": {"v0": 0.7, "v0001": 0.7}})
        r = loop.cycle()
        self.assertEqual((r.status, loop.reg.current()), ("held", "v0"))

    def test_rate_limit_and_force(self):
        cfg, loop, tp = mk(self.p, max_cycles_per_day=1)
        self.assertEqual(loop.cycle().status, "promoted")
        make_demo_traces(tp, n=300, seed=2, model="v0001")
        r = loop.cycle()
        self.assertEqual(r.status, "skipped")
        self.assertIn("rate limit", r.reason)
        self.assertNotEqual(loop.cycle(force=True).status, "skipped")

    def test_pause_before_and_during(self):
        cfg, loop, tp = mk(self.p)
        loop.reg.pause("manual")
        r = loop.cycle()
        self.assertEqual(r.status, "skipped")
        self.assertIn("paused", r.reason)
        loop.reg.resume()

        class PausingTrainer(DryRunTrainer):
            def train(s, ds, out):
                loop.reg.pause("mid-cycle")
                return super().train(ds, out)

        loop.trainer = PausingTrainer()
        r = loop.cycle()
        self.assertEqual((r.status, loop.reg.current()), ("error", "v0"))   # kill switch beats a passing gate
        self.assertIn("paused", r.reason)

    def test_promote_hook(self):
        marker = self.p / "marker"
        hook = [sys.executable, "-c", "import sys,pathlib; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
                str(marker), "{version}"]
        cfg, loop, tp = mk(self.p, promote_hook=hook)
        self.assertEqual(loop.cycle().status, "promoted")
        self.assertEqual(marker.read_text(), "v0001")

    def test_failing_hook_rolls_back(self):
        cfg, loop, tp = mk(self.p, promote_hook=[sys.executable, "-c", "import sys; sys.exit(1)"])
        r = loop.cycle()
        self.assertEqual((r.status, loop.reg.current()), ("rolled_back", "v0"))
        evs = [e["event"] for e in loop.reg.history()]
        self.assertEqual(evs.count("promote"), 1)
        self.assertEqual(evs.count("rollback"), 1)

    def test_not_initialised(self):
        cfg, loop, tp = mk(self.p, init=False)
        self.assertEqual(loop.cycle().status, "error")

    def test_training_failure_then_retry(self):
        class Flaky(DryRunTrainer):
            n = 0

            def train(s, ds, out):
                Flaky.n += 1
                if Flaky.n == 1:
                    raise RuntimeError("gpu fell off")
                return super().train(ds, out)

        cfg, loop, tp = mk(self.p, trainer=Flaky())
        r = loop.cycle()
        self.assertEqual(r.status, "error")
        self.assertIn("gpu fell off", r.reason)
        self.assertEqual(loop.reg.versions(), ["v0"])                      # nothing registered
        self.assertFalse(json.loads(Path(cfg.state_path).read_text() if Path(cfg.state_path).exists() else "{}")
                         .get("last_train_count"))                         # trigger not consumed
        r = loop.cycle()
        self.assertEqual((r.status, r.version), ("promoted", "v0001"))     # same version id reused, stale run dir moved
        self.assertTrue((Path(cfg.runs_dir) / "_stale").exists())

    def test_replay_failure_never_promotes_and_retries(self):
        class Down(SimReplayer):
            def prepare(s, ref):
                raise ConnectionError("llama-server down")

        cfg, loop, tp = mk(self.p, replayer=Down())
        r = loop.cycle()
        self.assertEqual((r.status, loop.reg.current()), ("error", "v0"))
        self.assertIn("v0001", loop.reg.versions())
        loop.replayer = SimReplayer(skill={"v0": 0.55, "v0002": 0.8})
        r = loop.cycle()
        self.assertEqual((r.status, r.version), ("promoted", "v0002"))

    def test_concurrent_cycle_lock(self):
        cfg, loop, tp = mk(self.p)
        (Path(cfg.root) / "cycle.lock").write_text("busy")
        r = loop.cycle()
        self.assertEqual(r.status, "skipped")
        self.assertIn("another cycle", r.reason)

    def test_golden_regression_blocks_even_when_better(self):
        cfg, loop, tp = mk(self.p)
        write_jsonl(self.p / "g.jsonl", [{"id": f"G{i}", "task": f"golden task {i}", "tags": ["golden"], "success": True,
                                           "granted_tier": 1, "steps": [{"tool": "app.open", "args": {"name": "x"}}]}
                                          for i in range(3)])
        with open(tp, "a") as f:
            f.write((self.p / "g.jsonl").read_text())

        def fn(version, t):
            if version == "v0":
                return Outcome(True if t.golden else (int(t.id.split("-")[-1]) % 2 == 0) if t.success else None, 0, 10.0)
            return Outcome(False if t.golden else (True if t.success else None), 0, 10.0)

        loop.replayer = Scripted(fn)
        r = loop.cycle()
        self.assertEqual((r.status, loop.reg.current()), ("rejected", "v0"))
        self.assertIn("golden", r.reason)

    def test_replay_cache_reuses_baseline(self):
        sim = Scripted(lambda v, t: Outcome(None if not t.success else (int(t.id.split("-")[-1]) % 10 < {"v0": 5, "v0001": 8, "v0002": 9}[v]), 0, 10.0))
        cfg, loop, tp = mk(self.p, replayer=sim)
        r1 = loop.cycle()
        self.assertEqual(r1.status, "promoted")
        c1 = dict(sim.calls)
        self.assertEqual(c1["v0"], c1["v0001"])
        make_demo_traces(tp, n=300, seed=2, model="v0001")
        r2 = loop.cycle()
        new_v0001_calls = sim.calls["v0001"] - c1["v0001"]
        self.assertLess(new_v0001_calls, r2.data["holdout"])               # baseline (v0001) outcomes came from cache

    def test_monitor_auto_rollback_and_ok(self):
        cfg, loop, tp = mk(self.p)
        loop.cycle()
        _live_traces(tp, "v0001", n=40, violations=0)
        r = loop.monitor_once()
        self.assertEqual((r.action, loop.reg.current()), ("ok", "v0001"))
        _live_traces(tp, "v0001", n=40, violations=8)
        r = loop.monitor_once()
        self.assertEqual((r.action, loop.reg.current()), ("rollback", "v0"))
        self.assertEqual(loop.monitor_once().action, "insufficient")       # nothing left to compare against

    def test_monitor_respects_auto_rollback_off(self):
        cfg, loop, tp = mk(self.p, auto_rollback=False)
        loop.cycle()
        _live_traces(tp, "v0001", n=40, violations=8)
        self.assertEqual(loop.monitor_once().action, "rollback")
        self.assertEqual(loop.reg.current(), "v0001")


class TestConfigAndCli(unittest.TestCase):
    def setUp(self):
        self.d, self.p = tmpdir()

    def tearDown(self):
        self.d.cleanup()

    def run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = cli(list(argv))
        return rc, buf.getvalue()

    def test_config_load(self):
        (self.p / "c.json").write_text(json.dumps({"root": "data", "traces": "t.jsonl", "bogus": 1}))
        with self.assertRaises(ValueError):
            LoopConfig.load(self.p / "c.json")
        (self.p / "c.json").write_text(json.dumps({"root": "data", "traces": "t.jsonl", "tiers": {"a": 1}}))
        cfg = LoopConfig.load(self.p / "c.json")
        self.assertEqual(Path(cfg.root), (self.p / "data").resolve())
        self.assertEqual(cfg.policy().tiers, {"a": 1})
        self.assertIsInstance(cfg.gate_config(["x"]), GateConfig)
        rc, out = self.run_cli("example-config")
        self.assertEqual(rc, 0)
        self.assertIn("tiers", json.loads(out))

    def test_plan_cli(self):
        q = dict(hidden_size=1536, num_hidden_layers=28, intermediate_size=8960, vocab_size=151936,
                 num_attention_heads=12, num_key_value_heads=2, tie_word_embeddings=True)
        (self.p / "q.json").write_text(json.dumps(q))
        rc, out = self.run_cli("plan", str(self.p / "q.json"))
        self.assertEqual(rc, 0)
        self.assertLessEqual(json.loads(out)["estimate_gib"]["total_gb"], 6.0)
        big = dict(q, hidden_size=3584, intermediate_size=18944, num_attention_heads=28, num_key_value_heads=4,
                   tie_word_embeddings=False, vocab_size=152064)
        (self.p / "b.json").write_text(json.dumps(big))
        self.assertEqual(self.run_cli("plan", str(self.p / "b.json"))[0], 2)

    def test_gate_cli_moves_pointer_only_on_promote(self):
        reg = Registry(self.p / "reg")
        reg.bootstrap("v0")
        reg.register("v1", "")
        mk_out = lambda name, oks: write_jsonl(self.p / name, [{"id": str(i), "ok": o, "violations": 0, "latency_ms": 10}
                                                                for i, o in enumerate(oks)])
        mk_out("b.jsonl", [True] * 70 + [False] * 30)
        mk_out("bad.jsonl", [True] * 50 + [False] * 50)
        mk_out("good.jsonl", [True] * 90 + [False] * 10)
        self.assertEqual(self.run_cli("gate", str(self.p / "b.jsonl"), str(self.p / "bad.jsonl"),
                                      "--registry", str(self.p / "reg"), "--version", "v1")[0], 1)
        self.assertEqual(reg.current(), "v0")
        rc, out = self.run_cli("gate", str(self.p / "b.jsonl"), str(self.p / "good.jsonl"),
                               "--registry", str(self.p / "reg"), "--version", "v1")
        self.assertEqual((rc, reg.current(), json.loads(out)["decision"]), (0, "v1", "promote"))

    def test_curate_cli(self):
        make_demo_traces(self.p / "t.jsonl", n=200, seed=3)
        rc, out = self.run_cli("curate", str(self.p / "t.jsonl"), str(self.p / "o"))
        self.assertEqual(rc, 0)
        self.assertTrue((self.p / "o" / "train.jsonl").exists())
        self.assertGreater(json.loads(out)["train"], 50)

    def test_ops_flow(self):
        make_demo_traces(self.p / "traces.jsonl", n=700, seed=1, model="v0")
        (self.p / "loop.json").write_text(json.dumps({
            "root": "data", "traces": "traces.jsonl", "base_model": "base", "tiers": DEMO_TIERS, "min_train": 50,
            "min_new_examples": 20, "trainer": {"kind": "dry"},
            "replay": {"kind": "sim", "params": {"skill": {"v0": 0.55, "v0001": 0.8}}}}))
        c = str(self.p / "loop.json")
        self.assertEqual(self.run_cli("init", "--config", c)[0], 0)
        self.assertEqual(json.loads(self.run_cli("status", "--config", c)[1])["current"], "v0")
        self.assertEqual(self.run_cli("pause", "--config", c, "--reason", "maintenance")[0], 0)
        rc, out = self.run_cli("cycle", "--config", c)
        self.assertEqual(rc, 0)
        self.assertIn("paused: maintenance", out)
        self.assertEqual(self.run_cli("resume", "--config", c)[0], 0)
        rc, out = self.run_cli("cycle", "--config", c)
        self.assertEqual((rc, json.loads(out)["status"]), (0, "promoted"))
        self.assertEqual(json.loads(self.run_cli("monitor", "--config", c)[1])["action"], "insufficient")
        rc, out = self.run_cli("rollback", "--config", c, "--reason", "test")
        self.assertEqual((rc, out.strip()), (0, "v0"))
        self.assertEqual(json.loads(self.run_cli("status", "--config", c)[1])["current"], "v0")

    def test_demo(self):
        res = run_demo(self.p / "demo", echo=lambda *_: None)
        self.assertEqual((res["cycle1"], res["cycle2"], res["monitor"], res["current_after_cycle2"], res["current_final"]),
                         ("promoted", "rejected", "rollback", "v0001", "v0"))
        self.assertEqual(self.run_cli("demo", "--dir", str(self.p / "demo2"))[0], 0)


if __name__ == "__main__":
    unittest.main()


class TestRegateAndGuards(unittest.TestCase):
    def setUp(self):
        self.d, self.p = tmpdir()

    def tearDown(self):
        self.d.cleanup()

    def test_regate_after_pause(self):
        cfg, loop, tp = mk(self.p)

        class PausingTrainer(DryRunTrainer):
            def train(s, ds, out):
                loop.reg.pause("mid")
                return super().train(ds, out)

        loop.trainer = PausingTrainer()
        self.assertEqual(loop.cycle().status, "error")                  # gate passed but promotion was refused
        self.assertEqual(loop.regate("v0001").status, "skipped")        # still paused
        loop.reg.resume()
        r = loop.regate("v0001")
        self.assertEqual((r.status, loop.reg.current()), ("promoted", "v0001"))
        self.assertEqual(loop.regate("v0001").reason, "already current")
        self.assertTrue(list((Path(cfg.runs_dir) / "v0001").glob("report.regate-*.json")))

    def test_regate_errors(self):
        cfg, loop, tp = mk(self.p)
        self.assertEqual(loop.regate("v9").status, "error")
        loop.cycle()
        loop.reg.register("v0009", "")
        self.assertIn("no stored holdout", loop.regate("v0009").reason)

    def test_schema_guard_blocks_mismapped_traces(self):
        write_jsonl(self.p / "bad.jsonl", [{"id": str(i), "task": f"t{i}", "success": True, "toolcalls_v2": []}
                                           for i in range(300)])
        cfg, loop, tp = mk(self.p, traces=str(self.p / "bad.jsonl"))
        r = loop.cycle()
        self.assertEqual(r.status, "error")
        self.assertIn("no tool calls", r.reason)
        self.assertEqual(loop.reg.versions(), ["v0"])                    # nothing trained
        sub = self.p / "second"
        sub.mkdir()
        cfg2, loop2, _ = mk(sub, max_noop_frac=1.0, traces=str(self.p / "bad.jsonl"))   # explicit override
        r2 = loop2.cycle()
        self.assertNotIn("schema check", r2.reason)
        self.assertNotEqual(r2.status, "error")

    def test_schema_guard_garbage_lines_and_empty_tasks(self):
        (self.p / "g.jsonl").write_text("\n".join(["{not json"] * 60 + [json.dumps({"id": "1", "task": "t"})] * 40))
        cfg, loop, tp = mk(self.p, traces=str(self.p / "g.jsonl"))
        self.assertIn("unparseable", loop.cycle().reason)
        write_jsonl(self.p / "e.jsonl", [{"id": str(i), "utterance_v2": "x", "success": True} for i in range(100)])
        cfg, loop, tp = mk(self.p, traces=str(self.p / "e.jsonl"), root=str(self.p / "d2"))
        self.assertIn("no task text", loop.cycle().reason)

    def test_trainer_warns_when_many_examples_skipped(self):
        from types import SimpleNamespace
        from jarvis_loop.trainer import LoraTrainer, ModelSpec
        (self.p / "data").mkdir()
        (self.p / "data" / "train.jsonl").write_text('{"messages": []}\n')

        def run(argv, **kw):
            cfg = json.loads(Path(argv[3]).read_text())
            Path(cfg["metrics"]).write_text(json.dumps({"status": "ok", "examples": 10, "skipped": {"too_long": 90}}))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        spec = ModelSpec(1536, 28, 8960, 151936, 12, 2, tied=True)
        res = LoraTrainer("b", spec, runner=run).train(self.p / "data", self.p / "o")
        self.assertIn("skipped", res.metrics["warning"])

    def test_cli_regate(self):
        make_demo_traces(self.p / "traces.jsonl", n=700, seed=1, model="v0")
        (self.p / "loop.json").write_text(json.dumps({
            "root": "data", "traces": "traces.jsonl", "base_model": "base", "tiers": DEMO_TIERS, "min_train": 50,
            "min_new_examples": 20, "trainer": {"kind": "dry"},
            "replay": {"kind": "sim", "params": {"skill": {"v0": 0.55, "v0001": 0.8}}}}))
        c = str(self.p / "loop.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli(["init", "--config", c]), 0)
            self.assertEqual(cli(["cycle", "--config", c]), 0)
            self.assertEqual(cli(["rollback", "--config", c]), 0)
            self.assertEqual(cli(["regate", "v0001", "--config", c]), 0)
        self.assertEqual(Registry(self.p / "data" / "registry").current(), "v0001")


if __name__ == "__main__":
    unittest.main()


class TestShippedExamples(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def test_example_config_loads_and_builds(self):
        cfg = LoopConfig.load(self.ROOT / "examples" / "loop.example.json")
        self.assertEqual(cfg.policy().tiers["shell.run"], 3)
        self.assertEqual(cfg.gate_config().min_scored, 40)
        self.assertEqual(cfg.monitor_config().window, 300)
        self.assertEqual(cfg.replay["kind"], "openai")

    def test_example_script_compiles(self):
        import py_compile
        py_compile.compile(str(self.ROOT / "examples" / "activate_ollama.py"), doraise=True)


if __name__ == "__main__":
    unittest.main()
