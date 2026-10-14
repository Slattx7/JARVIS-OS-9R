import dataclasses
import enum
import io
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from jarvis_trace import Kind, Recorder, Store, audit
from jarvis_trace.redact import Redactor
from jarvis_trace.schema import NormCtx, blob_ref, canonical, hash_normalized, normalize


def ctx(red=None, put=None):
    return NormCtx(redactor=red, put_blob=put)


class Color(enum.Enum):
    RED = "red"


@dataclasses.dataclass
class P:
    x: int
    y: bytes


class TestNormalize(unittest.TestCase):
    def test_canonical_key_order_stable(self):
        self.assertEqual(canonical({"b": 1, "a": 2}), canonical({"a": 2, "b": 1}))

    def test_bytes_become_blob_refs_hash_only_path_matches_store_path(self):
        st = Store()
        a = normalize({"audio": b"abc"}, st.norm_ctx())
        b = normalize({"audio": b"abc"}, st.norm_ctx(store_blobs=False))
        self.assertEqual(a, b)
        self.assertEqual(a["audio"]["__blob__"], blob_ref(b"abc"))
        self.assertEqual(st.blobs.get(blob_ref(b"abc")), b"abc")

    def test_misc_types(self):
        n = normalize({"c": Color.RED, "p": P(1, b"z"), "s": {3, 1, 2}, "t": (1, 2), "f": float("nan"),
                       "path": Path("/tmp/x"), 5: "intkey"}, ctx())
        self.assertEqual(n["c"], "red")
        self.assertEqual(n["s"], [1, 2, 3])
        self.assertEqual(n["t"], [1, 2])
        self.assertEqual(n["f"], {"__float__": "nan"})
        self.assertEqual(n["5"], "intkey")
        self.assertIn("__blob__", n["p"]["y"])
        canonical(n)  # must not raise (allow_nan=False)

    def test_surrogates_and_unknown_objects_do_not_crash(self):
        n = normalize({"s": "bad\ud800str", "o": object()}, ctx())
        canonical(n)
        n["s"].encode("utf-8")
        self.assertEqual(n["o"]["type"], "object")

    def test_long_string_externalized_and_rehydrates(self):
        st = Store(max_inline=100)
        n = normalize({"t": "x" * 500}, st.norm_ctx())
        self.assertEqual(n["t"]["as"], "str")
        self.assertEqual(st.rehydrate(n)["t"], "x" * 500)

    def test_depth_guard(self):
        d = cur = {}
        for _ in range(100):
            cur["n"] = {}
            cur = cur["n"]
        canonical(normalize(d, ctx()))


class TestRedact(unittest.TestCase):
    def setUp(self):
        self.r = Redactor(b"k" * 32)
        self.c = ctx(self.r)

    def test_patterns(self):
        s = ("mail bob@example.com or +91 98765 43210, token=abc123def456 "
             "Bearer abcdefghijklmnop1234 sk-live_abcdefghijklmnopqrstuv")
        out = self.r.scrub_str(s)
        for needle in ("bob@example.com", "98765", "abc123def456", "abcdefghijklmnop1234", "sk-live_abcdef"):
            self.assertNotIn(needle, out)
        self.assertIn("⟦email:", out)

    def test_plain_text_untouched(self):
        s = "turn the brightness down to twenty at 1728392039 seconds, max_tokens 512, version 3.12.3"
        self.assertEqual(self.r.scrub_str(s), s)

    def test_idempotent_and_deterministic(self):
        s = "reach me at a.b@c.io, password: hunter2hunter2, key 0123456789abcdef0123456789abcdef"
        once = self.r.scrub_str(s)
        self.assertEqual(self.r.scrub_str(once), once)
        self.assertEqual(self.r.scrub_str(s), once)
        other = Redactor(b"z" * 32).scrub_str(s)
        self.assertNotEqual(other, once)  # key-dependent

    def test_secret_keys_by_name_and_idempotent(self):
        n = normalize({"api_key": "xyz", "password": "p", "max_tokens": "512", "token_count": 7, "pin": 1234,
                       "nested": {"Authorization": "abc"}}, self.c)
        self.assertTrue(n["api_key"].startswith("⟦secret:"))
        self.assertTrue(n["password"].startswith("⟦secret:"))
        self.assertEqual(n["max_tokens"], "512")
        self.assertEqual(n["token_count"], 7)
        self.assertTrue(n["pin"].startswith("⟦secret:"))
        self.assertTrue(n["nested"]["Authorization"].startswith("⟦secret:"))
        self.assertEqual(normalize(n, self.c), n)

    def test_blob_refs_and_sha_prefix_survive(self):
        ref = blob_ref(b"data")
        n = normalize({"snap": {"__blob__": ref, "len": 4}, "s": ref}, self.c)
        self.assertEqual(n["snap"]["__blob__"], ref)
        self.assertEqual(n["s"], ref)  # hex after "sha256:" is not mangled

    def test_hash_stable_across_redaction_roundtrip(self):
        raw = {"q": "email bob@example.com please"}
        h1 = hash_normalized(normalize(raw, self.c))
        again = normalize(normalize(raw, self.c), self.c)  # replay feeds already-redacted data back in
        self.assertEqual(h1, hash_normalized(again))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, st, n=2):
        rec = Recorder(st)
        ids = []
        for i in range(n):
            with rec.trace("turn", {"i": i, "audio": b"pcm%d" % i}, tags=["t"]) as rt:
                rt.call(Kind.ASR, "asr", {"audio": b"pcm%d" % i}, lambda audio: "text")
                rt.set_output({"spoken": "ok"})
            ids.append(rt.trace_id)
        return ids

    def test_roundtrip_and_chain(self):
        st = Store(self.path)
        ids = self._record(st)
        tr = st.load_trace(ids[0])
        self.assertEqual(tr.status, "ok")
        self.assertEqual([s.kind for s in tr.spans], ["asr"])
        self.assertEqual(tr.spans[0].outputs, "text")
        self.assertEqual(tr.outputs, {"spoken": "ok"})
        r = st.verify()
        self.assertTrue(r.ok, r.problems)
        self.assertEqual(r.events, st.count_events())
        self.assertEqual(r.head, st.head())

    def test_append_only_triggers(self):
        st = Store(self.path)
        self._record(st, 1)
        with self.assertRaises(sqlite3.Error):
            st._conn.execute("UPDATE events SET body='{}'")
        with self.assertRaises(sqlite3.Error):
            st._conn.execute("DELETE FROM events")

    def test_tamper_detected(self):
        st = Store(self.path)
        self._record(st)
        st.close()
        con = sqlite3.connect(self.path)
        con.execute("DROP TRIGGER events_no_update")
        con.execute("UPDATE events SET body='{\"name\":\"evil\",\"tags\":[]}' WHERE seq=1")
        con.commit()
        con.close()
        r = Store(self.path).verify()
        self.assertFalse(r.ok)
        self.assertTrue(any("seq 1" in p for p in r.problems))

    def test_deleted_row_detected(self):
        st = Store(self.path)
        self._record(st)
        st.close()
        con = sqlite3.connect(self.path)
        con.execute("DROP TRIGGER events_no_delete")
        con.execute("DELETE FROM events WHERE seq=3")
        con.commit()
        con.close()
        self.assertFalse(Store(self.path).verify().ok)

    def test_blob_tamper_and_missing_detected(self):
        st = Store(self.path)
        self._record(st, 1)
        blobs = list(Path(self.path + ".blobs").rglob("*"))
        files = [b for b in blobs if b.is_file()]
        self.assertTrue(files)
        files[0].write_bytes(b"corrupted")
        r = st.verify()
        self.assertFalse(r.ok)
        self.assertTrue(any("corrupt blob" in p for p in r.problems))
        files[0].unlink()
        self.assertTrue(any("missing blob" in p for p in st.verify().problems))

    def test_two_handles_same_file_keep_chain_valid(self):
        a, b = Store(self.path), Store(self.path)
        self.assertEqual(a.key, b.key)
        ra, rb = Recorder(a), Recorder(b)

        def work(rec):
            for _ in range(15):
                with rec.trace("t") as rt:
                    rt.call(Kind.CUSTOM, "c", {"x": 1}, lambda x: x)

        ts = [threading.Thread(target=work, args=(r,)) for r in (ra, rb)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        r = a.verify()
        self.assertTrue(r.ok, r.problems)
        self.assertEqual(len(a.list_traces()), 30)

    def test_orphaned_action_surfaces_in_audit(self):
        st = Store(self.path)
        rt = Recorder(st).trace("crashy")
        rt.start()
        rt._begin("action", "shell.exec", {"cmd": "pm trim-caches"}, {"tier": 3, "intent": "clean"})  # never ended
        rt.finish()
        r = st.verify()
        self.assertEqual([u["kind"] for u in r.unfinished], ["action"])
        rows = audit.orphaned_actions(st)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].name, rows[0].tier, rows[0].outcome), ("shell.exec", 3, "unknown"))
        self.assertEqual(st.load_trace(rt.trace_id).spans[0].status, "orphaned")

    def test_labels_and_filters(self):
        st = Store(self.path)
        a, b = self._record(st)
        st.label(a, "golden")
        st.label(b, "golden")
        st.label(b, "golden", remove=True)
        self.assertEqual([t.trace_id for t in st.list_traces(label="golden")], [a])
        self.assertEqual(len(st.list_traces(tag="t")), 2)
        self.assertEqual(len(st.list_traces(tag="nope")), 0)
        self.assertEqual(len(st.list_traces(limit=1)), 1)
        with self.assertRaises(KeyError):
            st.label("t_missing", "x")

    def test_export_import_roundtrip_idempotent(self):
        st = Store(self.path)
        ids = self._record(st)
        buf = io.StringIO()
        st.export_jsonl(buf)
        dst = Store(os.path.join(self.tmp.name, "d.db"))
        buf.seek(0)
        ev, tr, bl = dst.import_jsonl(buf)
        self.assertEqual((tr, bl > 0), (2, True))
        self.assertTrue(dst.verify().ok)
        self.assertEqual(dst.load_trace(ids[0]).spans[0].input_hash, st.load_trace(ids[0]).spans[0].input_hash)
        self.assertEqual(dst.rehydrate(dst.load_trace(ids[0]).inputs)["audio"], b"pcm0")
        buf.seek(0)
        self.assertEqual(dst.import_jsonl(buf)[:2], (0, 0))  # idempotent per trace

    def test_redaction_at_rest(self):
        st = Store(self.path)
        rec = Recorder(st)
        with rec.trace("t", {"transcript": "email bob@example.com", "api_key": "hunter2"}) as rt:
            pass
        raw = "\n".join(r[0] for r in st._q("SELECT body FROM events"))
        self.assertNotIn("bob@example.com", raw)
        self.assertNotIn("hunter2", raw)


if __name__ == "__main__":
    unittest.main()
