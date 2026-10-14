import json, sys, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from common import O, T, tmpdir
from jarvis_loop import Outcome
from jarvis_loop.checker import Checker, ToolPolicy
from jarvis_loop.refs import ModelRef
from jarvis_loop.replay import (CommandActivator, CommandPropose, OpenAICompat, OutcomeCache, ProposalReplayer,
                                replay_all)
from jarvis_loop.trainer import (CommandTrainer, DoesNotFit, DryRunTrainer, LoraTrainer, ModelSpec, SEQ_LADDER,
                                 TrainError, estimate_vram, plan_training)
import jarvis_loop.train_lora as tl

QWEN15 = dict(hidden_size=1536, num_hidden_layers=28, intermediate_size=8960, vocab_size=151936,
              num_attention_heads=12, num_key_value_heads=2, tie_word_embeddings=True)
BIG7 = dict(hidden_size=3584, num_hidden_layers=28, intermediate_size=18944, vocab_size=152064,
            num_attention_heads=28, num_key_value_heads=4, tie_word_embeddings=False)
REF = ModelRef("v1", "/a/v1")


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.ck = Checker(ToolPolicy({"light": 1, "shell": 3}, default_grant=1))
        self.t = T("1", "lights", tools=("light",), granted_tier=1)

    def test_proposal_outcomes(self):
        mk = lambda fn: ProposalReplayer(fn, self.ck)
        good = json.dumps(self.t.calls)
        self.assertEqual(mk(lambda r, m: good).replay(REF, self.t).ok, True)
        self.assertEqual(mk(lambda r, m: "idk").replay(REF, self.t).ok, False)
        bad = mk(lambda r, m: '[{"tool":"shell","args":{}}]').replay(REF, self.t)
        self.assertEqual((bad.ok, bad.violations), (False, 1))

        def boom(r, m):
            raise TimeoutError("slow")
        o = mk(boom).replay(REF, self.t)
        self.assertEqual((o.ok, bool(o.error)), (False, True))
        failed = T("2", "x", ok=False)
        self.assertIsNone(mk(boom).replay(REF, failed).ok)               # no gold -> unscored even on error

    def test_replay_prompt_is_the_training_prompt(self):
        seen = []
        ProposalReplayer(lambda r, m: seen.append(m) or "[]", self.ck).replay(REF, self.t)
        from jarvis_loop.prompt import build_messages
        self.assertEqual(seen[0], build_messages("lights", "", 1))

    def test_openai_compat_against_stub_server(self):
        got = {}

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                got["path"] = self.path
                got["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                out = json.dumps({"choices": [{"message": {"content": '[{"tool":"light","args":{}}]'}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            c = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "jarvis:{version}", timeout=5)
            text = c(REF, [{"role": "user", "content": "hi"}])
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(got["path"], "/v1/chat/completions")
        self.assertEqual(got["body"]["model"], "jarvis:v1")
        self.assertEqual(got["body"]["temperature"], 0.0)
        self.assertIn("light", text)

    def test_command_propose_and_activator(self):
        cp = CommandPropose([sys.executable, "-c", "import sys,json; m=json.load(sys.stdin); print('[]' if m else 'x')"])
        self.assertEqual(cp(REF, [{"role": "user", "content": "a"}]).strip(), "[]")
        with self.assertRaises(RuntimeError):
            CommandPropose([sys.executable, "-c", "import sys; sys.exit(2)"])(REF, [])
        CommandActivator([sys.executable, "-c", "pass", "{version}"])(REF)
        with self.assertRaises(RuntimeError):
            CommandActivator([sys.executable, "-c", "import sys; sys.exit(1)"])(REF)

    def test_cache_and_replay_all(self):
        d, p = tmpdir()
        with d:
            calls = {"n": 0, "prep": 0}

            class R:
                def prepare(self, ref):
                    calls["prep"] += 1

                def replay(self, ref, t):
                    calls["n"] += 1
                    return Outcome(None if t.id == "e" else True, 0, 5.0, error="boom" if t.id == "e" else "")

            ts = [T(str(i), f"t{i}") for i in range(5)] + [T("e", "te")]
            cache = OutcomeCache(p / "c.json")
            out = replay_all(R(), REF, ts, cache)
            self.assertEqual((calls["n"], calls["prep"], len(out)), (6, 1, 6))
            out2 = replay_all(R(), REF, ts, OutcomeCache(p / "c.json"))     # reloaded from disk
            self.assertEqual(calls["n"], 7)                                  # only the errored trace re-ran (errors not cached)
            self.assertEqual(calls["prep"], 2)
            replay_all(R(), REF, ts[:5], OutcomeCache(p / "c.json"))
            self.assertEqual((calls["n"], calls["prep"]), (7, 2))            # fully cached: no prepare, no replay

    def test_workers_same_result(self):
        class R:
            def prepare(self, ref): pass
            def replay(self, ref, t): return Outcome(int(t.id) % 2 == 0, 0, 1.0)

        ts = [T(str(i), f"t{i}") for i in range(20)]
        self.assertEqual(replay_all(R(), REF, ts, workers=1), replay_all(R(), REF, ts, workers=4))


class TestSizing(unittest.TestCase):
    def test_spec_param_counts(self):
        s = ModelSpec.from_hf_config(QWEN15)
        self.assertTrue(1.50e9 < s.n_params() < 1.58e9, s.n_params())
        self.assertEqual(s.head_dim, 128)
        nested = ModelSpec.from_hf_config({"text_config": QWEN15})
        self.assertEqual(nested.n_params(), s.n_params())
        self.assertTrue(17e6 < s.lora_params(16) < 20e6, s.lora_params(16))

    def test_estimate_monotone_and_logit_trim(self):
        s = ModelSpec.from_hf_config(QWEN15)
        e = lambda **k: estimate_vram(s, **{"r": 16, "seq_len": 1024, "micro_batch": 1, **k})["total_gb"]
        self.assertLess(e(), e(seq_len=2048))
        self.assertLess(e(), e(micro_batch=4))
        self.assertLess(e(r=8), e(r=32))
        self.assertLess(e(), e(trim_logits=False, seq_len=1024))
        self.assertLess(e(), e(quantized=False))

    def test_plan_fits_6gb_and_rejects_7b(self):
        s = ModelSpec.from_hf_config(QWEN15)
        p = plan_training(s, 6.0)
        self.assertLessEqual(p.est_gb, 6.0 * 0.88)
        self.assertGreaterEqual(p.seq_len, 1024)
        self.assertGreaterEqual(p.micro_batch * p.grad_accum, 16)
        with self.assertRaises(DoesNotFit):
            plan_training(ModelSpec.from_hf_config(BIG7), 6.0)
        self.assertEqual(plan_training(ModelSpec.from_hf_config(BIG7), 24.0).r, 16)

    def test_shrink_ladder_terminates(self):
        s = ModelSpec.from_hf_config(QWEN15)
        p = plan_training(s, 6.0)
        seen, steps = [(p.micro_batch, p.seq_len, p.r)], 0
        while p is not None and steps < 50:
            p = p.shrunk(s, 16)
            steps += 1
            if p:
                seen.append((p.micro_batch, p.seq_len, p.r))
        self.assertIsNone(p)
        self.assertEqual(seen[-1][:2], (1, SEQ_LADDER[-1]))
        self.assertTrue(all(a != b for a, b in zip(seen, seen[1:])))


class TestTrainers(unittest.TestCase):
    def _ds(self, p, n=3):
        (p / "data").mkdir()
        (p / "data" / "train.jsonl").write_text("".join(json.dumps({"messages": []}) + "\n" for _ in range(n)))
        return p / "data"

    def test_dry_run_deterministic(self):
        d, p = tmpdir()
        with d:
            ds = self._ds(p)
            a = DryRunTrainer().train(ds, p / "o1")
            b = DryRunTrainer().train(ds, p / "o2")
            self.assertEqual((Path(a.artifact) / "adapter_model.safetensors").read_bytes(),
                             (Path(b.artifact) / "adapter_model.safetensors").read_bytes())
            self.assertEqual(a.metrics["examples"], 3)

    def _runner(self, script):
        calls = []

        def run(argv, **kw):
            cfg = json.loads(Path(argv[3]).read_text())
            calls.append(cfg)
            rc = script(len(calls), cfg)
            return SimpleNamespace(returncode=rc, stdout="out", stderr="err")
        return run, calls

    def test_lora_oom_retry_then_success(self):
        d, p = tmpdir()
        with d:
            ds = self._ds(p)

            def script(n, cfg):
                if n == 1:
                    return 3
                Path(cfg["out"]).mkdir(parents=True, exist_ok=True)
                Path(cfg["metrics"]).write_text(json.dumps({"status": "ok", "final_loss": 0.1}))
                return 0

            run, calls = self._runner(script)
            tr = LoraTrainer("base", ModelSpec.from_hf_config(QWEN15), runner=run)
            res = tr.train(ds, p / "out")
            self.assertEqual((res.metrics["attempts"], res.metrics["final_loss"]), (2, 0.1))
            a, b = calls
            self.assertTrue(b["micro_batch"] < a["micro_batch"] or b["seq_len"] < a["seq_len"] or b["r"] < a["r"])
            self.assertTrue(a["micro_batch"] * a["grad_accum"] >= 16 and b["micro_batch"] * b["grad_accum"] >= 16)
            self.assertTrue((p / "out" / "train_lora.py").exists())
            self.assertIn("attempt 2", (p / "out" / "train.log").read_text())

    def test_lora_failure_modes(self):
        d, p = tmpdir()
        with d:
            ds = self._ds(p)
            spec = ModelSpec.from_hf_config(QWEN15)
            run, calls = self._runner(lambda n, c: 3)
            with self.assertRaises(TrainError):
                LoraTrainer("b", spec, runner=run, max_attempts=3).train(ds, p / "o1")
            self.assertEqual(len(calls), 3)
            run, calls = self._runner(lambda n, c: 1)
            with self.assertRaises(TrainError):
                LoraTrainer("b", spec, runner=run).train(ds, p / "o2")
            self.assertEqual(len(calls), 1)                                  # non-OOM failure: no retry
            (p / "empty").mkdir()
            (p / "empty" / "train.jsonl").write_text("")
            with self.assertRaises(TrainError):
                LoraTrainer("b", spec, runner=run).train(p / "empty", p / "o3")

    def test_command_trainer(self):
        d, p = tmpdir()
        with d:
            ds = self._ds(p)
            seen = {}

            def run(argv, **kw):
                seen["argv"] = argv
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            r = CommandTrainer(["train", "--data", "{dataset}", "--out", "{out}"], runner=run).train(ds, p / "o")
            self.assertTrue(seen["argv"][2].endswith("train.jsonl"))
            self.assertEqual(r.artifact, str(p / "o"))
            with self.assertRaises(TrainError):
                CommandTrainer(["x"], runner=lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="no")).train(ds, p / "o2")


class FakeTok:
    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        s = "".join(f"<{m['role']}>{m['content']}</>" for m in msgs)
        if add_generation_prompt:
            s += "<assistant>"
        return [ord(c) for c in s]


class TestTrainScriptHelpers(unittest.TestCase):
    def test_compiles(self):
        import py_compile
        py_compile.compile(tl.__file__, doraise=True)

    def test_lr_schedule(self):
        base, total, warm = 1.0, 100, 5
        self.assertLess(tl.lr_at(0, total, warm, base), tl.lr_at(4, total, warm, base))
        self.assertAlmostEqual(tl.lr_at(4, total, warm, base), base)
        self.assertGreater(tl.lr_at(10, total, warm, base), tl.lr_at(60, total, warm, base))
        self.assertAlmostEqual(tl.lr_at(total, total, warm, base), 0.1 * base, places=6)
        self.assertEqual(tl.lr_at(3, 0, 0, 2e-4), 2e-4)

    def test_build_example_masks_prompt(self):
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "[]"}]
        (ids, lab), why = tl.build_example(FakeTok(), msgs, 512, 64)
        self.assertEqual(why, "ok")
        n_ans = len("[]</>")
        self.assertEqual(sum(1 for x in lab if x != -100), n_ans)
        self.assertTrue(all(x == -100 for x in lab[:-n_ans]))
        self.assertEqual(lab[-n_ans:], ids[-n_ans:])
        self.assertEqual(tl.build_example(FakeTok(), msgs, 5, 64)[1], "too_long")
        self.assertEqual(tl.build_example(FakeTok(), msgs, 512, 2)[1], "answer_too_long")

    def test_ids_handles_encodings(self):
        self.assertEqual(tl._ids([1, 2]), [1, 2])
        self.assertEqual(tl._ids({"input_ids": [[3, 4]]}), [3, 4])
        self.assertEqual(tl._ids(SimpleNamespace(input_ids=[5])), [5])

    def test_collate_and_window(self):
        batch = [([1, 2, 3], [-100, 2, 3]), ([4, 5], [-100, 5])]
        ids, lab, msk = tl.collate_left(batch, 0)
        self.assertEqual(ids, [[1, 2, 3], [0, 4, 5]])
        self.assertEqual(lab, [[-100, 2, 3], [-100, -100, 5]])
        self.assertEqual(msk, [[1, 1, 1], [0, 1, 1]])
        self.assertEqual(tl.window_k(lab, 3), 3)                           # max 2 answer tokens + 1
        self.assertEqual(tl.window_k([[-100, 7]], 2), 2)
        self.assertEqual(tl.window_k([[-100] * 9 + [7]], 4), 2)


if __name__ == "__main__":
    unittest.main()
