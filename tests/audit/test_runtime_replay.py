import asyncio
import unittest

from jarvis_trace import (AuditUnavailable, Kind, Recorder, ReplayedError, ReplayMiss, Store, TraceRuntime,
                          annotate, current, traced)
from jarvis_trace.runtime import ReplayPlan


class Counter:
    def __init__(self):
        self.n = {}

    def hit(self, k):
        self.n[k] = self.n.get(k, 0) + 1

    def get(self, k):
        return self.n.get(k, 0)


def replay(store, base_id, agent, **kw):
    base = store.load_trace(base_id)
    cand = store.clone_empty()
    rt = TraceRuntime.for_replay(base, store, cand, **kw)
    err = None
    try:
        with rt:
            rt.set_output(agent(rt, rt.inputs))
    except Exception as e:  # noqa: BLE001
        err = e
    return rt, cand.load_trace(rt.trace_id), err


class TestLive(unittest.TestCase):
    def test_nested_spans_errors_and_timing(self):
        st = Store()
        rt = Recorder(st).trace("t", {"x": 1})
        with rt:
            def outer(a):
                return rt.call(Kind.CUSTOM, "inner", {"b": a + 1}, lambda b: b * 2)

            self.assertEqual(rt.call(Kind.CUSTOM, "outer", {"a": 1}, outer), 4)
            with self.assertRaises(ValueError):
                rt.call(Kind.CUSTOM, "boom", {}, lambda: (_ for _ in ()).throw(ValueError("bad")))
        tr = st.load_trace(rt.trace_id)
        outer_s, inner_s, boom = tr.spans
        self.assertEqual(inner_s.parent_id, outer_s.span_id)
        self.assertIsNone(outer_s.parent_id)
        self.assertEqual((boom.status, boom.error["type"]), ("error", "ValueError"))
        self.assertTrue(all(s.dur_ns is not None for s in tr.spans))
        self.assertEqual(tr.status, "ok")

    def test_trace_error_status(self):
        st = Store()
        rt = Recorder(st).trace("t")
        with self.assertRaises(RuntimeError):
            with rt:
                raise RuntimeError("x")
        tr = st.load_trace(rt.trace_id)
        self.assertEqual((tr.status, tr.error["type"]), ("error", "RuntimeError"))

    def test_traced_decorator_sync_async_method_and_passthrough(self):
        class Comp:
            @traced(Kind.RECALL, "comp.q")
            def q(self, text, k=3):
                return [text] * k

        @traced(Kind.LLM, "alt")
        async def alt(prompt, **opts):
            return prompt.upper()

        c = Comp()
        self.assertEqual(c.q("a"), ["a"] * 3)  # no ambient runtime: passthrough

        st = Store()
        rt = Recorder(st).trace("t")

        async def go():
            async with rt:
                self.assertIs(current(), rt)
                self.assertEqual(c.q("a", k=2), ["a", "a"])
                self.assertEqual(await alt("hi", temperature=0), "HI")
                self.assertEqual(await rt.acall(Kind.CUSTOM, "sync-in-async", {"x": 1}, lambda x: x + 1), 2)

        asyncio.run(go())
        tr = st.load_trace(rt.trace_id)
        self.assertEqual([s.name for s in tr.spans], ["comp.q", "alt", "sync-in-async"])
        self.assertEqual(tr.spans[0].inputs, {"text": "a", "k": 2})  # self excluded
        self.assertEqual(tr.spans[1].inputs, {"prompt": "hi", "temperature": 0})  # **kwargs flattened
        self.assertIsNone(current())

    def test_annotate_is_durable_and_lands_on_action_span(self):
        st = Store()
        rt = Recorder(st).trace("t")
        with rt:
            def act(path):
                annotate(snapshot=rt.put_blob(b"old"), undo={"action": "restore", "args": {"path": path}})
                return {"status": "ok"}

            rt.call(Kind.ACTION, "file.write", {"path": "/x"}, act, meta={"tier": 2, "intent": "edit"})
        s = st.load_trace(rt.trace_id).spans[0]
        self.assertEqual(s.meta["tier"], 2)
        self.assertEqual(s.meta["undo"]["action"], "restore")
        self.assertEqual(st.rehydrate(s.meta["snapshot"]), b"old")

    def test_action_fails_closed_when_audit_write_fails(self):
        st = Store()
        rt = Recorder(st).trace("t")
        ran = Counter()
        orig = st.append

        def flaky(typ, tid, sid, body, **kw):
            if typ == "span_start" and body.get("kind") == "action":
                raise OSError("disk full")
            return orig(typ, tid, sid, body, **kw)

        with rt:
            st.append = flaky
            with self.assertRaises(AuditUnavailable):
                rt.call(Kind.ACTION, "shell.exec", {"cmd": "x"}, lambda cmd: ran.hit("a"), meta={"tier": 3})
            self.assertEqual(ran.get("a"), 0)  # the action did not run

    def test_non_action_fails_open(self):
        st = Store()
        rt = Recorder(st).trace("t")
        orig = st.append

        def flaky(typ, *a, **kw):
            if typ.startswith("span"):
                raise OSError("disk full")
            return orig(typ, *a, **kw)

        with self.assertLogs("jarvis_trace", "WARNING") as logs, rt:
            st.append = flaky
            self.assertEqual(rt.call(Kind.LLM, "m", {"x": 1}, lambda x: x), 1)
        self.assertTrue(rt.degraded)
        self.assertTrue(any("dropped span_start" in m for m in logs.output))


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.st = Store()
        self.c = Counter()
        c = self.c

        def recall(q):
            c.hit("recall")
            return [f"ctx:{q}"]

        def llm(q, ctx):
            c.hit("llm")
            return {"act": "settings.set" if "dim" in q else None}

        def act(v):
            c.hit("act")
            return {"status": "ok", "v": v}

        self.recall, self.llm, self.act = recall, llm, act

        def agent(rt, inputs):
            ctx = rt.call(Kind.RECALL, "r", {"q": inputs["q"]}, recall)
            plan = rt.call(Kind.LLM, "llm", {"q": inputs["q"], "ctx": ctx}, llm)
            if plan["act"]:
                rt.call(Kind.ACTION, plan["act"], {"v": 1}, act, meta={"tier": 1})
            return {"spoken": "ok"}

        self.agent = agent
        with Recorder(self.st).trace("turn", {"q": "dim it"}) as rt:
            rt.set_output(agent(rt, rt.inputs))
        self.base_id = rt.trace_id
        self.c.n.clear()

    def test_identical_replay_is_clean_and_executes_nothing_frozen(self):
        def agent(rt, inputs):
            ctx = rt.call(Kind.RECALL, "r", {"q": inputs["q"]}, self.recall)
            plan = rt.call(Kind.LLM, "llm", {"q": inputs["q"], "ctx": ctx}, self.llm)
            rt.call(Kind.ACTION, plan["act"], {"v": 1}, self.act, meta={"tier": 1})
            return {"spoken": "ok"}

        rt, cand, err = replay(self.st, self.base_id, agent)
        self.assertIsNone(err)
        self.assertEqual(rt.divergences, [])
        self.assertEqual((self.c.get("recall"), self.c.get("act"), self.c.get("llm")), (0, 0, 1))
        self.assertTrue(all(s.meta.get("frozen") for s in cand.spans if s.kind in ("recall", "action")))
        base = self.st.load_trace(self.base_id)
        self.assertEqual([s.dur_ns for s in cand.spans if s.kind == "recall"],
                         [s.dur_ns for s in base.spans if s.kind == "recall"])  # recorded duration carried

    def test_baseline_untouched_by_replay(self):
        before = self.st.count_events()
        replay(self.st, self.base_id, self.agent)
        self.assertEqual(self.st.count_events(), before)
        self.assertTrue(self.st.verify().ok)

    def test_new_action_is_blocked_stub_never_executed(self):
        def agent(rt, inputs):
            rt.call(Kind.ACTION, "shell.exec", {"cmd": "rm -rf /"}, self.act, meta={"tier": 3})
            return {}

        rt, cand, err = replay(self.st, self.base_id, agent)
        self.assertIsNone(err)
        self.assertEqual(self.c.get("act"), 0)
        a = cand.spans[0]
        self.assertEqual((a.status, a.meta["stub"]), ("blocked", True))
        self.assertEqual(a.outputs["status"], "not_executed")
        reasons = [(d.kind, d.reason) for d in rt.divergences]
        self.assertIn(("action", "no_recording"), reasons)
        self.assertIn(("recall", "unrequested"), reasons)  # recorded calls the candidate never made
        self.assertIn(("action", "unrequested"), reasons)

    def test_action_with_different_args_is_not_served_positionally(self):
        def agent(rt, inputs):
            return rt.call(Kind.ACTION, "settings.set", {"v": 999}, self.act, meta={"tier": 1})

        rt, cand, _ = replay(self.st, self.base_id, agent, on_miss="positional")
        self.assertEqual(cand.spans[0].status, "blocked")
        self.assertEqual(self.c.get("act"), 0)

    def test_positional_fallback_vs_raise_vs_stub(self):
        def agent(rt, inputs):
            return rt.call(Kind.RECALL, "r", {"q": "different query"}, self.recall)

        rt, cand, err = replay(self.st, self.base_id, agent, on_miss="positional")
        self.assertIsNone(err)
        self.assertEqual(cand.spans[0].outputs, ["ctx:dim it"])  # approximate: recorded output reused
        self.assertEqual(rt.divergences[0].reason, "input_mismatch")
        self.assertEqual(self.c.get("recall"), 0)

        rt, _, err = replay(self.st, self.base_id, agent, on_miss="raise")
        self.assertIsInstance(err, ReplayMiss)

        rt, cand, err = replay(self.st, self.base_id, agent, on_miss="stub")
        self.assertIsNone(err)
        self.assertIsNone(cand.spans[0].outputs)

    def test_reordered_calls_match_by_hash(self):
        st = Store()
        with Recorder(st).trace("t") as rt:
            rt.call(Kind.RECALL, "r", {"q": "a"}, lambda q: "A")
            rt.call(Kind.RECALL, "r", {"q": "b"}, lambda q: "B")
        got = []

        def agent(rt, inputs):
            got.append(rt.call(Kind.RECALL, "r", {"q": "b"}, lambda q: 1 / 0))
            got.append(rt.call(Kind.RECALL, "r", {"q": "a"}, lambda q: 1 / 0))

        r, _, err = replay(st, rt.trace_id, agent)
        self.assertIsNone(err)
        self.assertEqual(got, ["B", "A"])
        self.assertEqual(r.divergences, [])

    def test_baseline_error_is_replayed(self):
        st = Store()
        with Recorder(st).trace("t") as rt:
            try:
                rt.call(Kind.ASR, "asr", {"x": 1}, lambda x: 1 / 0)
            except ZeroDivisionError:
                pass
        seen = []

        def agent(rt, inputs):
            try:
                rt.call(Kind.ASR, "asr", {"x": 1}, lambda x: "never")
            except ReplayedError as e:
                seen.append(e.original_type)

        _, cand, err = replay(st, rt.trace_id, agent)
        self.assertIsNone(err)
        self.assertEqual(seen, ["ZeroDivisionError"])
        self.assertEqual(cand.spans[0].status, "error")

    def test_frozen_parent_consumes_recorded_children(self):
        st = Store()
        with Recorder(st).trace("t") as rt:
            def parent(q):
                return rt.call(Kind.RECALL, "embed", {"q": q}, lambda q: [0.1])

            rt.call(Kind.RECALL, "graph.query", {"q": "a"}, parent)
        got = []

        def agent(rt, inputs):
            got.append(rt.call(Kind.RECALL, "graph.query", {"q": "a"}, lambda q: 1 / 0))

        r, _, err = replay(st, rt.trace_id, agent)
        self.assertIsNone(err)
        self.assertEqual(got, [[0.1]])
        self.assertEqual(r.divergences, [])  # child span "embed" is not reported as unrequested

    def test_live_kinds_configurable_and_action_can_never_be_live(self):
        plan = ReplayPlan(self.st.load_trace(self.base_id), self.st, live={"llm", "recall", "action"})
        self.assertTrue(plan.is_frozen("action"))
        self.assertFalse(plan.is_frozen("recall"))
        self.assertTrue(plan.is_frozen("some-unknown-kind"))  # unknown kinds default to frozen (safe)

    def test_trace_inputs_rehydrated_for_agent(self):
        st = Store()
        with Recorder(st).trace("t", {"audio": b"\x00\x01pcm"}) as rt:
            pass
        seen = {}
        replay(st, rt.trace_id, lambda rt_, i: seen.update(i))
        self.assertEqual(seen["audio"], b"\x00\x01pcm")


if __name__ == "__main__":
    unittest.main()
