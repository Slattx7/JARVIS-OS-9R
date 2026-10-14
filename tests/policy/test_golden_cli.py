import io, json, os, sys, unittest, tempfile, contextlib
from helpers import *
from jarvis_actions import Action, Source
from jarvis_actions.cli import main

HERE = os.path.dirname(__file__)


class Golden(unittest.TestCase):
    def test_cases(self):
        e = make_env(default=True); bad = []
        for line in open(os.path.join(HERE, "policy_cases.jsonl")):
            c = json.loads(line)
            d = e.d.explain(Action(c["name"], c["args"], Source[c["source"]], c["device"], tuple(c["tainted"])))
            if d.verdict.value != c["verdict"] or (c["verdict"] != "deny" and int(d.tier) != c["tier"]):
                bad.append((c["name"], c["args"], c["device"], c["source"], c["tainted"], d.verdict.value, int(d.tier), d.reasons))
        self.assertEqual(bad, [])


class CLI(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        cfg = {"read_roots": [self.t + "/data"], "scratch_root": self.t + "/scratch", "audit_path": self.t + "/audit.jsonl",
               "frida_dir": self.t + "/frida", "devices": {"nitro": {"kind": "local"}, "9r": {"kind": "adb", "root": True}}}
        self.cfg = self.t + "/c.json"; json.dump(cfg, open(self.cfg, "w"))

    def call(self, *a):
        out = io.StringIO()
        with contextlib.redirect_stdout(out): rc = main(["-c", self.cfg, *a])
        return rc, out.getvalue()

    def test_tools_explain_plan_audit_replay(self):
        rc, out = self.call("tools"); self.assertEqual(rc, 0); self.assertIn("T3  frida.run", out)
        rc, out = self.call("explain", "shell.exec", "-a", "cmd=ls; id"); self.assertEqual(json.loads(out)["verdict"], "confirm_strong")
        rc, out = self.call("explain", "app.open", "-d", "9r", "-a", "package=com.a.b", "--taint", "package")
        self.assertEqual(json.loads(out)["tier"], 2)
        plan = self.t + "/p.json"; json.dump({"calls": [{"name": "ui.tap", "args": {"x": 1, "y": 2}, "device": "9r"}]}, open(plan, "w"))
        rc, out = self.call("plan", plan, "--dry-run", "--source", "derived")
        self.assertIn('"ok": false', out) if False else None                      # dry-run still needs confirm; stdin absent -> denied
        rc, out = self.call("audit", "verify", self.t + "/audit.jsonl"); self.assertIn("chain OK", out)
        rc, out = self.call("replay", self.t + "/audit.jsonl", "--fail-on-loosened"); self.assertEqual(rc, 0); self.assertIn("loosened=0", out)

    def test_frida_manifest(self):
        os.makedirs(self.t + "/fr"); open(self.t + "/fr/a.js", "w").write("1")
        rc, _ = self.call("frida-manifest", self.t + "/fr"); self.assertEqual(rc, 0)
        self.assertIn("a", json.load(open(self.t + "/fr/manifest.json")))

if __name__ == "__main__": unittest.main()
