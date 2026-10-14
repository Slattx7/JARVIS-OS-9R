import json, os, time, unittest

from common import O, T, tmpdir
from jarvis_loop import GateConfig, Outcome, Registry, Verdict, curate, gate
from jarvis_loop.curate import is_holdout, to_sft, write_dataset
from jarvis_loop.prompt import extract_calls
from jarvis_loop.registry import digest_path
from jarvis_loop.util import file_lock


def train_task(prefix="q"):
    return next(f"{prefix}{i}" for i in range(9999) if not is_holdout(T("x", f"{prefix}{i}")))


class TestCurate(unittest.TestCase):
    def test_filters_and_no_leak(self):
        ts = [T(f"t{i}", f"task {i}") for i in range(400)] + [T("f", "x", ok=False), T("v", "y", viol=1), T("d", "task 3")]
        c = curate(ts)
        self.assertFalse({t.norm_task for t in c.train} & {t.norm_task for t in c.holdout})
        self.assertTrue(all(t.violations == 0 and t.success for t in c.train))
        self.assertTrue(0.05 < len(c.holdout) / 400 < 0.30)

    def test_dup_and_cap(self):
        a = train_task()
        c = curate([T("1", a), T("2", a)])
        self.assertEqual((len(c.train), c.dropped["duplicate"]), (1, 1))
        b = train_task("w")
        five = [T(str(i), b, tools=("a",)) for i in range(5)]
        for i, t in enumerate(five):
            t.steps = [{"tool": "a", "args": {"n": i}}]            # distinct demos of the same task
        c = curate(five, max_per_task=3)
        self.assertEqual((len(c.train), c.dropped["capped"]), (3, 2))

    def test_holdout_keeps_failures_train_drops_them(self):
        ho = next(f"h{i}" for i in range(9999) if is_holdout(T("x", f"h{i}")))
        c = curate([T("a", ho, ok=False), T("b", train_task("z"), ok=False)])
        self.assertEqual((len(c.holdout), len(c.train), c.dropped["failed"]), (1, 0, 1))

    def test_golden_pins_whole_task(self):
        tk = train_task("gold")
        g = T("G", tk, tags=["golden"])
        c = curate([g, T("H", tk, tools=("b",))] + [T(f"n{i}", f"n{i}") for i in range(100)])
        self.assertEqual(c.golden, ["G"])
        self.assertTrue({"G", "H"} <= {t.id for t in c.holdout})
        self.assertNotIn(tk, {t.norm_task for t in c.train})

    def test_holdout_cap_preserves_golden(self):
        ts = [T(f"t{i}", f"task {i}") for i in range(1000)] + [T(f"g{i}", f"gold {i}", tags=["golden"]) for i in range(3)]
        c = curate(ts, holdout_frac=0.5, max_holdout=50)
        self.assertEqual(len(c.holdout), 50)
        self.assertTrue({"g0", "g1", "g2"} <= {t.id for t in c.holdout})
        self.assertGreater(c.dropped["holdout_capped"], 0)

    def test_sft_matches_replay_prompt_and_writes(self):
        t = T("1", "open x", tools=("app",), context="c", granted_tier=1)
        r = to_sft(t)
        self.assertEqual([m["role"] for m in r["messages"]], ["system", "user", "assistant"])
        self.assertEqual(extract_calls(r["messages"][2]["content"]), t.calls)
        d, p = tmpdir()
        with d:
            c = curate([T(f"t{i}", f"task {i}") for i in range(300)])
            stats = write_dataset(c, p / "out")
            for f in ("train.jsonl", "holdout.jsonl", "golden.json", "stats.json"):
                self.assertTrue((p / "out" / f).exists())
            self.assertEqual(len((p / "out" / "train.jsonl").read_text().splitlines()), stats["train"])


class TestGate(unittest.TestCase):
    def test_promote_hold_reject_basics(self):
        self.assertEqual(gate(O([1] * 70 + [0] * 30), O([1] * 85 + [0] * 15)).decision, "promote")
        self.assertEqual(gate(O([1] * 50), O([1] * 50)).decision, "hold")
        self.assertEqual(gate(O([1] * 50 + [0] * 50), O([1] * 100, viol={3: 1})).decision, "reject")
        self.assertEqual(gate(O([1] * 5), O([1] * 4)).decision, "reject")
        self.assertEqual(gate({}, {}).decision, "reject")
        self.assertEqual(gate(O([1] * 50, lat=100), O([1] * 50, lat=70)).decision, "promote")

    def test_golden_and_significant_regressions(self):
        b = O([1] * 20 + [0] * 80)
        c = O([0] + [1] * 19 + [1] * 80)
        self.assertEqual(gate(b, c, GateConfig(golden=frozenset({"0"}))).decision, "reject")
        self.assertEqual(gate(O([1] * 100), O([0] * 20 + [1] * 80), GateConfig(max_drop=0.5)).decision, "reject")

    def test_none_ok_is_probe_only(self):
        b, c = O([True] * 40 + [None] * 10), O([True] * 40 + [None] * 10)
        self.assertEqual(gate(b, c).decision, "hold")
        c2 = O([True] * 40 + [None] * 10, viol={45: 1})          # violation on an unscored safety probe still rejects
        v = gate(b, c2)
        self.assertEqual(v.decision, "reject")
        self.assertIn("violation", v.reasons[0])

    def test_insufficient_evidence(self):
        v = gate(O([1] * 10), O([1] * 10))
        self.assertEqual(v.decision, "hold")
        self.assertIn("insufficient", v.reasons[0])
        self.assertEqual(gate(O([1] * 10), O([1] * 10, viol={0: 1})).decision, "reject")

    def test_slice_guard(self):
        base = O([1] * 40 + [0] * 60)
        cand_oks = [0] * 15 + [1] * 25 + [1] * 40 + [0] * 20            # slice a: 40 -> 25 ok; slice b: 0 -> 40 ok
        sl = {str(i): ("a" if i < 40 else "b") for i in range(100)}
        v = gate(base, O(cand_oks), slices=sl)
        self.assertEqual(v.decision, "reject")
        self.assertIn("slice 'a'", v.reasons[0])
        self.assertEqual(gate(base, O(cand_oks)).decision, "promote")   # without the guard the aggregate hides it

    def test_replay_errors_reject(self):
        v = gate(O([1] * 50), O([1] * 50, err={i: "boom" for i in range(10)}))
        self.assertEqual(v.decision, "reject")
        self.assertIn("replay errors", v.reasons[0])

    def test_gain_not_significant_holds(self):
        base = O([1] * 60 + [0] * 40)
        cand = O([0] + [1] * 59 + [1] * 3 + [0] * 37)                   # 1 regression, 3 improvements, +2pts
        v = gate(base, cand)
        self.assertEqual(v.decision, "hold")
        self.assertIn("not significant", v.reasons[0])
        self.assertIn("HOLD", v.report())


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.d, self.p = tmpdir()
        self.reg = Registry(self.p / "reg")
        self.ok = Verdict("promote", ["x"], {})

    def tearDown(self):
        self.d.cleanup()

    def test_multi_level_rollback(self):
        r = self.reg
        r.bootstrap("v0", "base-id")
        for v in ("v1", "v2", "v3"):
            r.register(v, f"/m/{v}")
        for v in ("v1", "v2", "v3"):
            r.promote(v, self.ok)
        self.assertEqual(r.current(), "v3")
        self.assertEqual([r.rollback(), r.rollback(), r.rollback()], ["v2", "v1", "v0"])
        self.assertEqual(r.cur.read_text(), "v0")
        with self.assertRaises(RuntimeError):
            r.rollback()
        self.assertEqual(r.stack(), ["v0"])

    def test_only_promote_verdict_moves_pointer(self):
        self.reg.bootstrap("v0")
        self.reg.register("v1", "")
        with self.assertRaises(PermissionError):
            self.reg.promote("v1", Verdict("hold", ["x"]))
        self.assertEqual(self.reg.current(), "v0")

    def test_integrity(self):
        art = self.p / "adapter"
        art.mkdir()
        (art / "w.bin").write_bytes(b"abc")
        self.reg.bootstrap("v0")
        self.reg.register("v1", str(art))
        d1 = digest_path(str(art))
        self.assertEqual(d1, digest_path(str(art)))
        (art / "w.bin").write_bytes(b"abd")
        self.assertNotEqual(d1, digest_path(str(art)))
        with self.assertRaises(ValueError):
            self.reg.promote("v1", self.ok)
        self.assertEqual(self.reg.current(), "v0")

    def test_rollback_target_tamper_detected(self):
        art = self.p / "a1"
        art.mkdir()
        (art / "w").write_bytes(b"1")
        self.reg.bootstrap("v0")
        self.reg.register("v1", str(art))
        self.reg.promote("v1", self.ok)
        self.reg.register("v2", "")
        self.reg.promote("v2", self.ok)
        (art / "w").write_bytes(b"2")
        with self.assertRaises(ValueError):
            self.reg.rollback()                                      # v1 is the target and was tampered with

    def test_pause_blocks_promote_not_rollback(self):
        r = self.reg
        r.bootstrap("v0")
        r.register("v1", "")
        r.promote("v1", self.ok)
        r.register("v2", "")
        r.pause("investigating")
        self.assertEqual(r.paused(), "investigating")
        with self.assertRaises(PermissionError):
            r.promote("v2", self.ok)
        self.assertEqual(r.rollback(), "v0")
        r.resume()
        self.assertIsNone(r.paused())
        r.promote("v2", self.ok)
        self.assertEqual(r.current(), "v2")

    def test_ledger_is_truth_current_file_self_heals(self):
        r = self.reg
        r.bootstrap("v0")
        r.register("v1", "")
        r.promote("v1", self.ok)
        r.cur.unlink()
        self.assertEqual(r.current(), "v1")
        r.register("v2", "")
        r.promote("v2", self.ok)
        self.assertEqual(r.cur.read_text(), "v2")

    def test_guards(self):
        self.reg.bootstrap("v0")
        with self.assertRaises(RuntimeError):
            self.reg.bootstrap("v9")
        with self.assertRaises(FileExistsError):
            self.reg.register("v0", "")
        with self.assertRaises(ValueError):
            self.reg.register("../evil", "")
        with self.assertRaises(KeyError):
            self.reg.promote("nope", self.ok)
        self.assertEqual(self.reg.ref("v0").version, "v0")

    def test_lock_busy_and_stale(self):
        lp = self.p / "x.lock"
        with file_lock(lp):
            with self.assertRaises(TimeoutError):
                with file_lock(lp, timeout=0.0):
                    pass
        lp.write_text("dead")
        old = time.time() - 1000
        os.utime(lp, (old, old))
        with file_lock(lp, timeout=0.0, stale=300):                  # stale lock stolen
            pass
        self.assertFalse(lp.exists())


if __name__ == "__main__":
    unittest.main()
