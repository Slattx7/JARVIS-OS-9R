import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from jarvis_trace import (ActionRegistry, ActionSpec, Executor, PolicyEngine, Recorder, ScriptedConfirmer, Store,
                          diff_policies)
from jarvis_trace.policy import DEFAULT_POLICY, run_cases

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples", "jarvis_wiring.py")


def load_example():
    spec = importlib.util.spec_from_file_location("jarvis_wiring_example", EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestExample(unittest.TestCase):
    def test_end_to_end_demo(self):
        mod = load_example()
        with tempfile.TemporaryDirectory() as d:
            out = mod.run_demo(os.path.realpath(d), verbose=False)
        self.assertEqual(out["violations"], [])
        self.assertEqual(out["paused_status"], "denied")
        self.assertFalse(out["note_exists_after_undo"])
        self.assertEqual((out["same_exit"], out["loose_exit"]), (0, 2))
        self.assertEqual(out["loosened"], ["fs.delete"])  # shell rm -rf stays denied: deny rules are not loosened
        self.assertEqual(out["live_outputs"]["wipe the data"], "wiping: denied")
        self.assertEqual(out["live_outputs"]["what does bob like"], "bob prefers email")
        self.assertTrue(out["store_ok"])

    def test_cli_entrypoints_of_the_example(self):
        mod = load_example()
        root = os.path.realpath(tempfile.mkdtemp())
        try:
            mod.run_demo(root, verbose=False)
            env = {**os.environ, "JARVIS_EXAMPLE_ROOT": root}
            store = os.path.join(root, "traces.db")
            p = subprocess.run([sys.executable, "-m", "jarvis_trace", "replay", "--agent", EX + ":agent", "--label", "golden",
                                "--store", store], cwd=ROOT, capture_output=True, text=True, env=env)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("6 cases", p.stdout)
            p = subprocess.run([sys.executable, "-m", "jarvis_trace", "check", "--store", store], cwd=ROOT,
                               capture_output=True, text=True, env=env)
            self.assertEqual(p.returncode, 0, p.stdout)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestShippedConfigs(unittest.TestCase):
    def test_example_configs_stay_valid(self):
        import json
        from jarvis_trace import load_gate
        pol = os.path.join(ROOT, "examples", "policy.json")
        e = PolicyEngine.from_file(pol)
        self.assertEqual(e.lint(), [])
        with open(os.path.join(ROOT, "examples", "policy_cases.json")) as f:
            self.assertEqual(run_cases(e, json.load(f)), [])
        g = load_gate(os.path.join(ROOT, "examples", "gate.json"))
        self.assertEqual((g.min_pass, g.live, g.select["label"]), (0.95, ["llm", "permission"], "golden"))
        self.assertTrue(os.access(os.path.join(ROOT, "examples", "ci_gate.sh"), os.X_OK))


class TestBytesArgsRegression(unittest.TestCase):
    """Raw bytes in action args used to crash canonical() in policy-diff and the executor's default confirm text."""

    def test_executor_default_describe_and_diff_with_bytes_args(self):
        reg = ActionRegistry()
        reg.register(ActionSpec("blob.put", lambda data: {"status": "ok", "n": len(data)}, 2, {"data": bytes}))  # no describe=
        ex = Executor(reg, PolicyEngine(), ScriptedConfirmer({"*": True}))
        st = Store()
        with Recorder(st).trace("t") as rt:
            r = ex.run(rt, "blob.put", {"data": b"\x00\x01binary"})
        self.assertEqual((r.status, r.result["n"]), ("ok", 8))
        cases = [{"action": "blob.put", "args": {"data": b"\x00\x01"}}, {"action": "blob.put", "args": {"data": b"\x00\x01"}}]
        loose = dict(DEFAULT_POLICY)
        loose["tier_decisions"] = {**DEFAULT_POLICY["tier_decisions"], "3": "allow"}  # blob.put is unmapped -> tier 3
        d = diff_policies(PolicyEngine(), PolicyEngine(loose), cases)
        self.assertEqual((d.n, len(d.loosened), d.loosened[0]["count"]), (1, 1, 2))
        self.assertIn("__blob__", str(d.loosened[0]["args"]))
        fails = run_cases(PolicyEngine(), [{"action": "blob.put", "args": {"data": b"x"}, "expect": {"decision": "allow"}}])
        self.assertEqual(len(fails), 1)


if __name__ == "__main__":
    unittest.main()
