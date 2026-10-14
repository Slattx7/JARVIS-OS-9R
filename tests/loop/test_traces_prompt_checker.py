import json, unittest

from common import O, T, tmpdir, write_jsonl
from jarvis_loop import Trace, load_traces
from jarvis_loop.checker import Checker, ToolPolicy, match
from jarvis_loop.prompt import assistant_text, build_messages, extract_calls


class TestTraces(unittest.TestCase):
    def test_aliases_and_defaults(self):
        t = Trace.from_dict({"trace_id": "a", "utterance": "Hi  There", "actions": [
            {"name": "x", "arguments": {"k": 1}}, "y"], "outcome": {"success": True}, "tier": 2,
            "timestamp": "2026-01-01T00:00:00Z", "labels": ["golden"], "recalled": ["m1", "m2"]})
        self.assertEqual(t.id, "a")
        self.assertEqual(t.norm_task, "hi there")
        self.assertEqual(t.calls, [{"tool": "x", "args": {"k": 1}}, {"tool": "y", "args": {}}])
        self.assertTrue(t.success and t.golden)
        self.assertEqual(t.granted_tier, 2)
        self.assertEqual(t.context, "m1\nm2")
        self.assertGreater(t.ts, 1.7e9)

    def test_outcome_string_and_violation_counting(self):
        t = Trace.from_dict({"id": "b", "task": "t", "outcome": "failure",
                             "steps": [{"tool": "a", "denied": True}, {"tool": "b", "tier_violation": True}, {"tool": "c"}]})
        self.assertFalse(t.success)
        self.assertEqual(t.violations, 2)
        self.assertEqual(Trace.from_dict({"id": "c", "task": "t", "violations": ["x", "y", "z"]}).violations, 3)

    def test_key_ignores_case_ws_not_args(self):
        a = Trace.from_dict({"id": "1", "task": "Open  X", "steps": [{"tool": "t", "args": {"a": 1}}]})
        b = Trace.from_dict({"id": "2", "task": "open x", "steps": [{"tool": "t", "args": {"a": 1}}]})
        c = Trace.from_dict({"id": "3", "task": "open x", "steps": [{"tool": "t", "args": {"a": 2}}]})
        self.assertEqual(a.key, b.key)
        self.assertNotEqual(a.key, c.key)

    def test_loader_skips_bad_lines_and_strict(self):
        d, p = tmpdir()
        with d:
            f = p / "t.jsonl"
            f.write_text('{"id":"1","task":"a","success":true}\nnot json\n[1,2]\n\n')
            st = {}
            got = list(load_traces(f, stats=st))
            self.assertEqual((len(got), st["bad_lines"], st["loaded"]), (1, 2, 1))
            with self.assertRaises(ValueError):
                list(load_traces(f, strict=True))

    def test_loader_directory(self):
        d, p = tmpdir()
        with d:
            write_jsonl(p / "a.jsonl", [{"id": "1", "task": "x"}])
            write_jsonl(p / "b.jsonl", [{"id": "2", "task": "y"}])
            self.assertEqual([t.id for t in load_traces(p)], ["1", "2"])


class TestPrompt(unittest.TestCase):
    def test_extract_variants(self):
        ex = extract_calls
        self.assertEqual(ex('[{"tool":"a","args":{"x":1}}]'), [{"tool": "a", "args": {"x": 1}}])
        self.assertEqual(ex('Sure!\n```json\n[{"tool": "a"}]\n```'), [{"tool": "a", "args": {}}])
        self.assertEqual(ex("[]"), [])
        self.assertEqual(ex('{"calls":[{"tool":"a","args":{}}]}'), [{"tool": "a", "args": {}}])
        self.assertEqual(ex('ids [1,2] then [{"tool":"b","args":{}}]'), [{"tool": "b", "args": {}}])
        for bad in ("no idea", "[1,2,3]", '[{"name":"a"}]', "", None, 5):
            self.assertIsNone(ex(bad), bad)

    def test_messages_and_roundtrip(self):
        m = build_messages(" open x ", "ctx line", 2)
        self.assertEqual([x["role"] for x in m], ["system", "user"])
        self.assertIn("granted_tier: 2", m[1]["content"])
        self.assertIn("context:\nctx line", m[1]["content"])
        self.assertTrue(m[1]["content"].endswith("request: open x"))
        self.assertNotIn("granted_tier", build_messages("x")[1]["content"])
        calls = [{"tool": "a", "args": {"k": [1, 2]}}]
        self.assertEqual(extract_calls(assistant_text(calls)), calls)


class TestChecker(unittest.TestCase):
    def setUp(self):
        self.pol = ToolPolicy({"read": 0, "light": 1, "shell": 3}, default_grant=1, deny=frozenset({"rm"}))

    def test_violations(self):
        sh, rm, zz = [{"tool": "shell", "args": {}}], [{"tool": "rm", "args": {}}], [{"tool": "zzz", "args": {}}]
        self.assertEqual(self.pol.violations(sh, None), 1)
        self.assertEqual(self.pol.violations(sh, 3), 0)
        self.assertEqual(self.pol.violations(rm, 3), 1)          # denylist beats any grant
        self.assertEqual(self.pol.violations(zz, 0), 0)          # unknown tool = miss, not a permission violation
        self.assertEqual(self.pol.violations(sh + sh, 1), 2)

    def test_match_modes(self):
        g = [{"tool": "a", "args": {"x": 1}}]
        extra = [{"tool": "a", "args": {"x": 1, "force": True}}]
        wrong = [{"tool": "a", "args": {"x": 2}}]
        self.assertTrue(match(g, g, "exact"))
        self.assertFalse(match(g, extra, "exact"))
        self.assertTrue(match(g, extra, "subset"))
        self.assertFalse(match(g, wrong, "subset"))
        self.assertTrue(match(g, wrong, "tool_only"))
        self.assertFalse(match(g, g + g))
        with self.assertRaises(ValueError):
            Checker(self.pol, "fuzzy")

    def test_check(self):
        ck = Checker(self.pol)
        good = T("1", "t", tools=("light",), granted_tier=1)
        self.assertEqual(ck.check(good, good.calls), (True, 0))
        self.assertEqual(ck.check(good, None), (False, 0))
        self.assertEqual(ck.check(good, [{"tool": "light", "args": {"on": True}}]), (False, 0))
        self.assertEqual(ck.check(good, [{"tool": "shell", "args": {}}]), (False, 1))
        failed = T("2", "t", ok=False, granted_tier=0)
        self.assertEqual(ck.check(failed, [{"tool": "light", "args": {}}]), (None, 1))   # no gold: safety probe only
        noop = T("3", "t", tools=())
        self.assertEqual(ck.check(noop, []), (True, 0))


if __name__ == "__main__":
    unittest.main()
