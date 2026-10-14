import copy
import json
import os
import tempfile
import unittest

from jarvis_trace import (AuditUnavailable, Recorder, Store, audit, killswitch, run_suite)
from jarvis_trace.actions import ActionRegistry, ActionSpec, Executor, UndoRunner, validate_args, ArgError
from jarvis_trace.builtin_actions import build_registry
from jarvis_trace.confirm import (ApproveAll, CallbackConfirmer, CLIConfirmer, DenyAll, ScriptedConfirmer)
from jarvis_trace.invariants import check_trace, live_invariants
from jarvis_trace.policy import (DEFAULT_POLICY, PolicyEngine, PolicyError, cases_from_store, diff_policies,
                                 run_cases, validate_policy)


def rd(path):
    with open(path) as f:
        return f.read()


def rj(path):
    with open(path) as f:
        return json.load(f)


def pol(**over):
    p = copy.deepcopy(DEFAULT_POLICY)
    p.update(over)
    return PolicyEngine(p)


class TestPolicy(unittest.TestCase):
    def test_default_table(self):
        e = PolicyEngine()
        got = {a: (e.check(a)["tier"], e.check(a)["decision"]) for a in
               ("screen.read", "settings.set", "message.send", "fs.delete", "mystery.thing")}
        self.assertEqual(got, {"screen.read": (0, "allow"), "settings.set": (1, "allow"),
                               "message.send": (2, "confirm"), "fs.delete": (3, "confirm"),
                               "mystery.thing": (3, "confirm")})  # unknown action fails closed
        self.assertEqual(e.check("mystery.thing")["rule"], "default")

    def test_rules_only_tighten(self):
        p = copy.deepcopy(DEFAULT_POLICY)
        p["rules"].append({"id": "try-loosen", "action": "message.send", "decision": "allow"})
        e = PolicyEngine(p)
        self.assertEqual(e.check("message.send")["decision"], "confirm")
        self.assertTrue(any("try-loosen" in w for w in e.lint()))
        # raising a tier also pulls the decision up to what that tier requires
        p2 = copy.deepcopy(DEFAULT_POLICY)
        p2["rules"].append({"id": "bump", "action": "kv.set", "when": {"arg": "key", "eq": "wifi"}, "tier": 3})
        r = PolicyEngine(p2).check("kv.set", {"key": "wifi", "value": 1})
        self.assertEqual((r["tier"], r["decision"], r["rule"]), (3, "confirm", "bump"))
        self.assertEqual(PolicyEngine(p2).check("kv.set", {"key": "x", "value": 1})["tier"], 1)

    def test_context_rules(self):
        e = PolicyEngine()
        self.assertEqual(e.check("message.send", {}, {"recent": {"message.send": 5}})["decision"], "deny")
        self.assertEqual(e.check("message.send", {}, {"recent": {"message.send": 4}})["decision"], "confirm")
        self.assertEqual(e.check("settings.set", {}, {"paused": True})["rule"], "paused")
        self.assertEqual(e.check("screen.read", {}, {"paused": True})["decision"], "allow")  # tier 0 unaffected
        self.assertEqual(e.check("settings.set", {}, {"risk": {"risk": 0.6}})["decision"], "confirm")
        self.assertEqual(e.check("settings.set", {}, {"risk": {"risk": 0.9}})["decision"], "deny")
        self.assertEqual(e.check("message.send", {}, {"attended": False, "hour": 3})["decision"], "deny")
        self.assertEqual(e.check("message.send", {}, {"attended": True, "hour": 3})["decision"], "confirm")

    def test_args_and_missing_value_semantics(self):
        e = PolicyEngine()
        self.assertEqual(e.check("shell.exec", {"cmd": "rm -rf /"})["decision"], "deny")
        self.assertEqual(e.check("shell.exec", {"cmd": "ls; dd if=/dev/zero of=/dev/sda"})["decision"], "deny")
        self.assertEqual(e.check("shell.exec", {"cmd": "pm trim-caches 1G"})["decision"], "confirm")
        p = copy.deepcopy(DEFAULT_POLICY)
        p["rules"].append({"id": "scope", "action": "fs.*", "when": {"arg": "path", "not_under": ["/tmp/j"]},
                           "decision": "deny"})
        e = PolicyEngine(p)
        for path in ("/tmp/j/a", "/tmp/j"):
            self.assertEqual(e.check("fs.write", {"path": path})["decision"], "confirm", path)
        for path in ("/tmp/j/../etc/passwd", "/tmp/jx/a", "relative/x", "/etc/x", None):
            self.assertEqual(e.check("fs.write", {"path": path})["decision"], "deny", path)
        self.assertEqual(e.check("fs.write", {})["decision"], "deny")  # missing arg: negative op fires

    def test_validation_rejects_bad_policies(self):
        bad = [
            {"rules": [{"id": "a", "decision": "maybe"}]},
            {"rules": [{"id": "a", "decision": "deny", "when": {"arg": "x"}}]},
            {"rules": [{"id": "a", "decision": "deny", "when": {"arg": "x", "eq": 1, "ne": 2}}]},
            {"rules": [{"id": "a", "decision": "deny", "when": {"arg": "x", "matches": "("}}]},
            {"rules": [{"id": "a", "decision": "deny"}, {"id": "a", "decision": "deny"}]},
            {"rules": [{"id": "a"}]},
            {"tiers": {"x": 7}},
            {"tier_decisions": {"9": "allow"}},
            {"bogus": 1},
        ]
        for b in bad:
            with self.assertRaises(PolicyError, msg=str(b)):
                validate_policy(b)

    def test_policy_id_and_purity(self):
        a, b = PolicyEngine(), PolicyEngine()
        self.assertEqual(a.policy_id, b.policy_id)
        p = copy.deepcopy(DEFAULT_POLICY)
        p["tiers"]["kv.set"] = 2
        self.assertNotEqual(PolicyEngine(p).policy_id, a.policy_id)
        args, ctx = {"cmd": "echo"}, {"recent": {}}
        self.assertEqual(a.check("shell.exec", args, ctx), a.check("shell.exec", args, ctx))

    def test_lint_and_explain(self):
        p = copy.deepcopy(DEFAULT_POLICY)
        p["tier_decisions"]["2"] = "allow"
        p["rules"] = [r for r in p["rules"] if r["id"] != "paused"]
        w = " ".join(PolicyEngine(p).lint())
        self.assertIn("tier 2", w)
        self.assertIn("ctx.paused", w)
        self.assertEqual(PolicyEngine().lint(), [])
        text = "\n".join(PolicyEngine().explain("shell.exec", {"cmd": "mkfs /dev/x"}))
        self.assertIn("MATCH -> decision=deny", text)
        self.assertIn("deciding: shell-destructive", text)

    def test_run_cases_and_diff(self):
        e = PolicyEngine()
        fails = run_cases(e, [{"action": "message.send", "expect": {"decision": "confirm", "tier": 2}},
                              {"action": "message.send", "expect": {"decision": "allow"}}])
        self.assertEqual(len(fails), 1)
        loose = copy.deepcopy(DEFAULT_POLICY)
        loose["tier_decisions"]["2"] = "allow"
        tight = copy.deepcopy(DEFAULT_POLICY)
        tight["tiers"]["kv.set"] = 2
        cases = [{"action": "message.send"}, {"action": "message.send"}, {"action": "kv.set", "args": {"key": "a"}},
                 {"action": "screen.read"}]
        d = diff_policies(e, PolicyEngine(loose), cases)
        self.assertEqual((d.n, len(d.loosened), d.loosened[0]["count"]), (3, 1, 2))  # deduped
        d = diff_policies(e, PolicyEngine(tight), cases)
        self.assertEqual((len(d.loosened), len(d.tightened)), (0, 1))
        self.assertEqual(diff_policies(e, PolicyEngine(), cases).n, 3)

    def test_context_window_pause_and_clock(self):
        t = [1000.0]
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "PAUSE")
            e = PolicyEngine(window_s=60, pause_file=pf, clock=lambda: t[0])
            e.note_executed("message.send")
            e.note_executed("message.send")
            self.assertEqual(e.context()["recent"], {"message.send": 2})
            t[0] += 61
            self.assertEqual(e.context()["recent"], {})
            self.assertFalse(e.context()["paused"])
            killswitch.engage(pf, "testing")
            c = e.context(extra={"attended": False})
            self.assertEqual((c["paused"], c["pause_reason"], c["attended"]), (True, "testing", False))


class TestKillSwitchAndConfirmers(unittest.TestCase):
    def test_killswitch(self):
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "sub", "PAUSE")
            self.assertIsNone(killswitch.state(pf))
            killswitch.engage(pf, "because", who="me")
            s = killswitch.state(pf)
            self.assertEqual((s["reason"], s["who"]), ("because", "me"))
            self.assertTrue(killswitch.release(pf))
            self.assertFalse(killswitch.release(pf))
            with open(pf, "w") as f:
                f.write("{not json")
            self.assertTrue(killswitch.state(pf)["corrupt"])  # fail closed: corrupt file still pauses
            os.environ["JARVIS_PAUSE_FILE"] = pf
            try:
                self.assertEqual(killswitch.default_path(), pf)
            finally:
                del os.environ["JARVIS_PAUSE_FILE"]

    def test_confirmers(self):
        s = ScriptedConfirmer({"fs.*": True, "message.send": False}, default=False)
        self.assertTrue(s.ask("fs.write", {}, 2, "", "")["approved"])
        self.assertFalse(s.ask("message.send", {}, 2, "", "")["approved"])
        self.assertFalse(s.ask("other", {}, 2, "", "")["approved"])
        self.assertEqual(len(s.asked), 3)
        self.assertFalse(DenyAll().ask("a", {}, 1, "", "")["approved"])
        self.assertTrue(ApproveAll().ask("a", {}, 1, "", "")["approved"])
        self.assertFalse(CallbackConfirmer(lambda *a: "yes").ask("a", {}, 1, "", "")["approved"])  # truthy != True
        self.assertTrue(CallbackConfirmer(lambda *a: True).ask("a", {}, 1, "", "")["approved"])
        r = CallbackConfirmer(lambda *a: 1 / 0).ask("a", {}, 1, "", "")
        self.assertFalse(r["approved"])
        self.assertIn("ZeroDivisionError", r["error"])
        out = []
        self.assertTrue(CLIConfirmer(lambda p: "Y", out.append).ask("a", {}, 2, "why", "do it")["approved"])
        self.assertFalse(CLIConfirmer(lambda p: "", out.append).ask("a", {}, 2, "", "do it")["approved"])
        def eof(_):
            raise EOFError
        self.assertFalse(CLIConfirmer(eof, out.append).ask("a", {}, 2, "", "do it")["approved"])


class ExecBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.realpath(self.tmp.name)
        self.kv = os.path.join(self.dir, "kv.json")
        self.reg = build_registry([self.dir], self.kv)
        p = copy.deepcopy(DEFAULT_POLICY)
        p["rules"].append({"id": "fs-scope", "action": "fs.*", "when": {"arg": "path", "not_under": [self.dir]},
                           "decision": "deny"})
        self.policy = p
        self.engine = PolicyEngine(p)
        self.conf = ScriptedConfirmer({"fs.*": True, "kv.*": True}, default=False)
        self.ex = Executor(self.reg, self.engine, self.conf)
        self.store = Store(os.path.join(self.dir, "t.db"))
        self.rec = Recorder(self.store)
        self.f = os.path.join(self.dir, "a.txt")

    def tearDown(self):
        self.tmp.cleanup()

    def spans(self, rt, kind):
        return self.store.load_trace(rt.trace_id).spans_of(kind)


class TestExecutor(ExecBase):
    def test_flow_statuses_and_effects(self):
        with self.rec.trace("t") as rt:
            r = [self.ex.run(rt, a, args, intent="test") for a, args in [
                ("kv.set", {"key": "brightness", "value": 20}),
                ("fs.write", {"path": self.f, "content": "hello"}),
                ("fs.write", {"path": self.f, "content": "world"}),
                ("shell.exec", {"cmd": "rm -rf /"}),
                ("shell.exec", {"cmd": "echo hi"}),
                ("fs.write", {"path": "/etc/x", "content": "y"}),
                ("fs.write", {"path": self.f, "bogus": 1}),
                ("nope.act", {}),
                ("fs.delete", {"path": self.f}),
            ]]
        self.assertEqual([x.status for x in r], ["ok", "ok", "ok", "denied", "declined", "denied", "invalid_args",
                                                 "unknown_action", "ok"])
        self.assertEqual(r[3].rule, "shell-destructive")
        self.assertEqual(r[5].rule, "fs-scope")
        self.assertFalse(os.path.exists(self.f))
        self.assertEqual(rj(self.kv), {"brightness": 20})
        names = [s.name for s in self.spans(rt, "action")]
        self.assertEqual(names, ["kv.set", "fs.write", "fs.write", "fs.delete"])  # denied/declined/invalid never reach action
        self.assertEqual(len(self.spans(rt, "confirm")), 4)  # 3 fs confirms + the declined shell.exec
        self.assertEqual(len(self.spans(rt, "permission")), 7)  # 9 requests - unknown action - invalid args
        self.assertEqual(len(self.spans(rt, "context")), 7)

    def test_executor_traces_satisfy_all_live_invariants(self):
        with self.rec.trace("t") as rt:
            self.ex.run(rt, "kv.set", {"key": "k", "value": 1})
            self.ex.run(rt, "fs.write", {"path": self.f, "content": "x"})
            self.ex.run(rt, "fs.delete", {"path": self.f})
            self.ex.run(rt, "fs.read", {"path": self.dir + "/nothing"})  # tier 0, errors in handler
        v = check_trace(self.store.load_trace(rt.trace_id), live_invariants())
        self.assertEqual([x.detail for x in v], [])

    def test_audit_rows_carry_tier_snapshot_undo(self):
        with self.rec.trace("t") as rt:
            self.ex.run(rt, "fs.write", {"path": self.f, "content": "v1"}, intent="seed")
            self.ex.run(rt, "fs.write", {"path": self.f, "content": "v2"}, intent="edit")
            self.ex.run(rt, "fs.delete", {"path": self.f}, intent="cleanup")
        rows = audit.list_actions(self.store, trace_id=rt.trace_id)
        self.assertEqual([(r.name, r.tier, r.outcome, r.intent) for r in rows],
                         [("fs.write", 2, "ok", "seed"), ("fs.write", 2, "ok", "edit"), ("fs.delete", 3, "ok", "cleanup")])
        self.assertIsNone(rows[0].snapshot)  # file did not exist: nothing to snapshot, undo is a remove
        self.assertEqual(rows[0].undo["action"], "fs.remove")
        self.assertEqual(self.store.rehydrate(rows[1].snapshot), b"v1")
        self.assertEqual(self.store.rehydrate(rows[2].snapshot), b"v2")
        self.assertEqual(rows[2].undo["action"], "fs.restore")

    def test_undo_roundtrip_restores_state(self):
        with self.rec.trace("t") as rt:
            self.ex.run(rt, "kv.set", {"key": "b", "value": 20})
            self.ex.run(rt, "fs.write", {"path": self.f, "content": "hello"})
            self.ex.run(rt, "fs.write", {"path": self.f, "content": "world"})
            self.ex.run(rt, "fs.delete", {"path": self.f})
        undo_ex = Executor(self.reg, self.engine, ScriptedConfirmer({"*": True}))
        ur = UndoRunner(self.store, undo_ex)
        dry = ur.run(rt.trace_id)
        self.assertEqual((dry["applied"], [s["status"] for s in dry["steps"]]), (False, ["planned"] * 4))
        self.assertFalse(os.path.exists(self.f))  # dry run touched nothing
        res = ur.run(rt.trace_id, apply=True)
        self.assertEqual([s["status"] for s in res["steps"]], ["ok"] * 4, res)
        self.assertFalse(os.path.exists(self.f))                 # created by the first write -> removed again
        self.assertEqual(rj(self.kv), {})           # kv key did not exist before -> unset
        undo_tr = self.store.load_trace(res["undo_trace_id"])
        self.assertIn("undo", undo_tr.tags)
        self.assertEqual(undo_tr.meta["undoes"], rt.trace_id)
        self.assertIn("undone", self.store.load_trace(rt.trace_id).label_set())
        with self.assertRaises(RuntimeError):
            ur.run(rt.trace_id, apply=True)                      # not twice
        self.assertTrue(self.store.verify().ok)

    def test_undo_refuses_to_remove_a_file_that_changed_since(self):
        with self.rec.trace("t") as rt:
            self.ex.run(rt, "fs.write", {"path": self.f, "content": "mine"})
        with open(self.f, "w") as fh:
            fh.write("someone else edited this")
        res = UndoRunner(self.store, Executor(self.reg, self.engine, ScriptedConfirmer({"*": True}))).run(
            rt.trace_id, apply=True)
        self.assertEqual(res["steps"][0]["status"], "error")
        self.assertIn("sha mismatch", res["steps"][0]["error"])
        self.assertTrue(os.path.exists(self.f))

    def test_undo_specs_are_stored_exact_not_redacted(self):
        # regression: the redactor used to mangle hex digests / emails inside recorded undo specs
        with self.rec.trace("t") as rt:
            self.ex.run(rt, "kv.set", {"key": "contact", "value": "alice@example.com"})
            self.ex.run(rt, "kv.set", {"key": "contact", "value": "bob@example.com"})
            self.ex.run(rt, "fs.write", {"path": self.f, "content": "hello"})
        rows = audit.list_actions(self.store, trace_id=rt.trace_id)
        self.assertEqual(rows[1].undo, {"action": "kv.set", "args": {"key": "contact", "value": "alice@example.com"}})
        self.assertEqual(len(rows[2].undo["args"]["expect_sha256"]), 64)
        raw = "\n".join(r[0] for r in self.store._q("SELECT body FROM events WHERE type='span_start'"))
        self.assertNotIn("alice@example.com", raw)  # inputs are still redacted at rest...
        res = UndoRunner(self.store, Executor(self.reg, self.engine, ScriptedConfirmer({"*": True}))).run(
            rt.trace_id, apply=True)
        self.assertEqual([s["status"] for s in res["steps"]], ["ok"] * 3)  # ...but the undo specs still execute
        self.assertEqual(rj(self.kv), {})

    def test_hash_keys_are_not_redacted(self):
        with self.rec.trace("t") as rt:
            self.ex.run(rt, "fs.write", {"path": self.f, "content": "hello"})
        out = self.spans(rt, "action")[0].outputs
        self.assertEqual(out["sha256"], "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")

    def test_dry_run_changes_nothing_and_is_not_counted(self):
        with self.rec.trace("t") as rt:
            r = self.ex.run(rt, "fs.write", {"path": self.f, "content": "x"}, dry_run=True)
        self.assertEqual(r.status, "dry_run")
        self.assertFalse(os.path.exists(self.f))
        self.assertEqual(audit.list_actions(self.store)[0].outcome, "dry_run")
        self.assertEqual(self.engine.context()["recent"], {})
        self.assertEqual(check_trace(self.store.load_trace(rt.trace_id), live_invariants()), [])

    def test_pause_switch_blocks_everything_above_tier_zero(self):
        pf = os.path.join(self.dir, "PAUSE")
        eng = PolicyEngine(self.policy, pause_file=pf)
        ex = Executor(self.reg, eng, ScriptedConfirmer({"*": True}))
        killswitch.engage(pf, "maintenance")
        with self.rec.trace("t") as rt:
            a = ex.run(rt, "kv.set", {"key": "k", "value": 1})
            b = ex.run(rt, "fs.write", {"path": self.f, "content": "x"})
            c = ex.run(rt, "fs.read", {"path": self.f})
        self.assertEqual((a.status, a.rule, b.status, b.rule), ("denied", "paused", "denied", "paused"))
        self.assertEqual(c.status, "error")  # tier 0 is allowed through (file just doesn't exist)
        self.assertFalse(os.path.exists(self.kv))
        killswitch.release(pf)
        with self.rec.trace("t2") as rt:
            self.assertEqual(ex.run(rt, "kv.set", {"key": "k", "value": 1}).status, "ok")

    def test_rate_limit_via_recorded_context(self):
        reg = ActionRegistry()
        sent = []
        reg.register(ActionSpec("message.send", lambda to: sent.append(to) or {"status": "ok"}, 2, {"to": str}))
        ex = Executor(reg, PolicyEngine(), ScriptedConfirmer({"*": True}))
        with self.rec.trace("t") as rt:
            res = [ex.run(rt, "message.send", {"to": f"p{i}"}).status for i in range(7)]
        self.assertEqual(res, ["ok"] * 5 + ["denied"] * 2)
        self.assertEqual(len(sent), 5)
        ctxs = [s.outputs for s in self.spans(rt, "context")]
        self.assertEqual(ctxs[5]["recent"], {"message.send": 5})  # the state that caused the denial is in the log

    def test_risk_hook_feeds_policy(self):
        scores = {"kv.set": 0.6, "kv.unset": 0.9}
        ex = Executor(self.reg, self.engine, ScriptedConfirmer({"kv.set": True}),
                      risk_fn=lambda action, args: {"risk": scores.get(action, 0.0), "label": "stub"})
        with self.rec.trace("t") as rt:
            a = ex.run(rt, "kv.set", {"key": "k", "value": 1})        # tier1 allow -> risk 0.6 -> confirm (approved)
            b = ex.run(rt, "kv.unset", {"key": "k"})                   # risk 0.9 -> deny
            c = ex.run(rt, "kv.get", {"key": "k"})                     # risk 0 -> allow
        self.assertEqual((a.status, a.rule, b.status, b.rule, c.status), ("ok", "risk-confirm", "denied", "risk-deny", "ok"))
        self.assertEqual(len(self.spans(rt, "risk")), 3)

    def test_audit_failure_fails_closed_through_executor(self):
        orig = self.store.append

        def flaky(typ, tid, sid, body, **kw):
            if typ == "span_start" and body.get("kind") == "action":
                raise OSError("disk full")
            return orig(typ, tid, sid, body, **kw)

        with self.rec.trace("t") as rt:
            self.store.append = flaky
            with self.assertRaises(AuditUnavailable):
                self.ex.run(rt, "fs.write", {"path": self.f, "content": "x"})
        self.assertFalse(os.path.exists(self.f))  # the write never happened

    def test_arg_validation(self):
        spec = self.reg.get("fs.write")
        validate_args(spec, {"path": "/x", "content": "c"})
        validate_args(spec, {"path": "/x", "content": b"c", "create_only": True})
        for bad in ({"path": "/x"}, {"path": 5, "content": "c"}, {"path": "/x", "content": "c", "create_only": 1},
                    {"path": "/x", "content": "c", "extra": 1}):
            with self.assertRaises(ArgError):
                validate_args(spec, bad)

    def test_handler_exception_is_recorded_as_outcome(self):
        with self.rec.trace("t") as rt:
            r = self.ex.run(rt, "fs.write", {"path": self.f, "content": "x", "create_only": True})
            r2 = self.ex.run(rt, "fs.write", {"path": self.f, "content": "y", "create_only": True})
        self.assertEqual((r.status, r2.status), ("ok", "error"))
        self.assertIn("FileExistsError", r2.error)
        self.assertEqual(rd(self.f), "x")
        self.assertEqual([a.outcome for a in audit.list_actions(self.store, trace_id=rt.trace_id)], ["ok", "error"])

    def test_builtin_shell_and_fs_confinement(self):
        with self.rec.trace("t") as rt:
            a = self.ex.run(rt, "shell.read", {"cmd": "echo hello"})
            b = self.ex.run(rt, "shell.read", {"cmd": "echo hi; whoami"})
            c = self.ex.run(rt, "shell.read", {"cmd": "rm file"})
            d = self.ex.run(rt, "shell.read", {"cmd": "cat /etc/hostname"})
            link = os.path.join(self.dir, "escape")
            os.symlink("/etc", link)
            e = self.ex.run(rt, "fs.read", {"path": link + "/hostname"})
        self.assertEqual((a.status, a.result["stdout"].strip()), ("ok", "hello"))
        self.assertEqual((b.status, c.status, d.status, e.status), ("error", "error", "error", "error"))
        self.assertIn("metacharacters", b.error)
        self.assertIn("allowlist", c.error)
        self.assertIn("outside", d.error)
        self.assertIn("outside", e.error)  # symlink escape caught by realpath


class TestReplayThroughExecutor(ExecBase):
    def setUp(self):
        super().setUp()
        self.calls = {"n": 0}
        f = self.f

        self.plans = {
            "save a note": [("fs.write", {"path": f, "content": "note"})],
            "dim it": [("kv.set", {"key": "brightness", "value": 20})],
            "clean up": [("fs.delete", {"path": f})],
            "chat": [],
        }

        def make_agent(ex):
            def agent(rt, inputs):
                plan = rt.call("llm", "planner", {"text": inputs["text"]}, lambda text: self.plans[text])
                res = [ex.run(rt, a, args, intent=inputs["text"]).status for a, args in plan]
                return {"spoken": ", ".join(res) or "ok"}
            return agent

        self.make_agent = make_agent
        base = make_agent(self.ex)
        for text in ("save a note", "dim it", "clean up", "chat"):
            with self.rec.trace("turn", {"text": text}) as rt:
                rt.set_output(base(rt, rt.inputs))
            self.store.label(rt.trace_id, "golden")
        self.fs_state = (rd(f) if os.path.exists(f) else None, rd(self.kv))

    def replay(self, policy, confirmer=None):
        ex = Executor(self.reg, PolicyEngine(policy), confirmer or ScriptedConfirmer({"*": True}))
        agent = self.make_agent(ex)
        agent.__name__ = "cand"
        return run_suite(self.store, agent, select={"label": "golden"})

    def test_same_policy_replays_clean_and_touches_nothing(self):
        r = self.replay(self.policy)
        self.assertEqual((r.exit_code(), len(r.cases)), (0, 4), r.summary_text())
        after = (rd(self.f) if os.path.exists(self.f) else None, rd(self.kv))
        self.assertEqual(after, self.fs_state)

    def test_loosened_policy_is_a_safety_fail(self):
        p = copy.deepcopy(self.policy)
        p["tier_decisions"]["2"] = "allow"
        r = self.replay(p)
        self.assertEqual(r.exit_code(), 2, r.summary_text())
        self.assertIn("permission_loosening", r.summary_text())
        self.assertEqual(self.fs_state[1], rd(self.kv))

    def test_tightened_policy_is_a_regression_not_safety(self):
        p = copy.deepcopy(self.policy)
        p["rules"].append({"id": "no-writes", "action": "fs.write", "decision": "deny"})
        r = self.replay(p)
        self.assertEqual((r.verdict, r.exit_code()), ("fail", 1), r.summary_text())
        self.assertIn("action_sequence", r.summary_text())

    def test_new_action_in_candidate_is_stubbed_and_unapproved(self):
        # candidate plans an extra destructive action never seen in the baseline
        self.plans["chat"] = [("fs.delete", {"path": self.f})]
        r = self.replay(self.policy)
        case = [c for c in r.cases if c.name == "turn" and any(
            d["reason"] == "no_recording" for d in c.divergences)]
        self.assertTrue(case)
        self.assertEqual(r.exit_code(), 2, r.summary_text())  # new tier>=2 proposal
        self.assertEqual(self.fs_state[0], rd(self.f) if os.path.exists(self.f) else None)


class TestRecordedPolicyCases(unittest.TestCase):
    def test_cases_from_store_and_policy_diff_end_to_end(self):
        from jarvis_trace import demo
        st = Store()
        demo.record_baseline(st)
        cases = list(cases_from_store(st))
        self.assertEqual(len(cases), 7)  # one recorded check per proposed action (brightness is proposed twice)
        self.assertEqual({c["action"] for c in cases},
                         {"settings.set", "screen.read", "message.send", "timer.set", "shell.exec", "app.install"})
        loose = copy.deepcopy(DEFAULT_POLICY)
        loose["tier_decisions"]["2"] = "allow"
        d = diff_policies(PolicyEngine(), PolicyEngine(loose), cases)
        self.assertEqual({r["action"] for r in d.loosened}, {"message.send", "app.install"})


if __name__ == "__main__":
    unittest.main()
