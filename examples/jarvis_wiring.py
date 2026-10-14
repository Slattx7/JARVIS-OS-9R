"""End-to-end wiring reference: a voice turn built on jarvis_trace. Runs as-is, touches only a temp directory.

    python examples/jarvis_wiring.py

What it shows, in order: live turns through the Executor (policy -> confirm -> snapshot -> write-ahead action -> undo
spec), the audit trail, the pause switch, an undo, replay of the same turns against (1) the same policy, (2) a
loosened policy, (3) a policy diff, and the live-trace invariant check. Swap the fake asr/recall/llm/tts for your real
components; nothing else changes. `agent` and `make_executor` are also the entrypoints for the CLI:

    JARVIS_EXAMPLE_ROOT=/tmp/jt python -m jarvis_trace replay --agent examples/jarvis_wiring.py:agent --store ...
    JARVIS_EXAMPLE_ROOT=/tmp/jt python -m jarvis_trace undo <trace> --executor examples/jarvis_wiring.py:make_executor
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis_trace import (DEFAULT_POLICY, Executor, PolicyEngine, Recorder, RiskAdapter, ScriptedConfirmer, Store,
                          UndoRunner, annotate, audit, check_trace, killswitch, live_invariants, recall_call,
                          run_suite)
from jarvis_trace.builtin_actions import build_registry
from jarvis_trace.policy import diff_policies, cases_from_store

# --------------------------------------------------------------------------------------------- fake components ----
NOTES = {"bob": "bob prefers email", "mom": "mom prefers texts", "brightness": "usual brightness is 40"}


def asr(audio: bytes) -> str:
    return audio.decode()


def recall(q: str, k: int) -> List[Dict[str, Any]]:
    return [{"id": key, "text": v} for key, v in NOTES.items() if key in q][:k]


def classify(features: Dict[str, Any]) -> Dict[str, Any]:  # stand-in for the warden classifier
    return {"risk": 0.9 if "sketchy" in str(features.get("package", "")) else 0.05, "label": "stub"}


def planner(text: str, context: List[Dict[str, Any]], root: str) -> Dict[str, Any]:
    f = lambda n: os.path.join(root, n)
    if "dim" in text:
        return {"reply": "dimmed", "actions": [("kv.set", {"key": "brightness", "value": 20})]}
    if "edit" in text:
        return {"reply": "edited", "actions": [("fs.write", {"path": f("note.txt"), "content": "buy oat milk"})]}
    if "delete" in text:
        return {"reply": "deleted", "actions": [("fs.delete", {"path": f("note.txt")})]}
    if "note" in text:
        return {"reply": "saved", "actions": [("fs.write", {"path": f("note.txt"), "content": "buy milk"})]}
    if "wipe" in text:
        return {"reply": "wiping", "actions": [("shell.exec", {"cmd": "rm -rf /data"})]}
    return {"reply": context[0]["text"] if context else "ok", "actions": []}


def tts(text: str) -> Dict[str, Any]:
    annotate(first_audio_ms=90)  # streaming TTS: report when the first chunk plays
    return {"samples": len(text) * 160}


# ------------------------------------------------------------------------------------------------- wiring -------
def root_dir() -> str:
    return os.environ.get("JARVIS_EXAMPLE_ROOT") or os.path.join(tempfile.gettempdir(), "jarvis_example")


def policy(root: str) -> Dict[str, Any]:
    p = copy.deepcopy(DEFAULT_POLICY)
    p["rules"].append({"id": "fs-scope", "action": "fs.*", "when": {"arg": "path", "not_under": [root]},
                       "decision": "deny"})
    return p


def build_executor(root: str, pol: Optional[Dict[str, Any]] = None, answers: Optional[Dict[str, bool]] = None,
                   pause_file: Optional[str] = None) -> Executor:
    reg = build_registry([root], os.path.join(root, "kv.json"))
    risk = RiskAdapter(classify, {"app.*": lambda a: {"package": a.get("pkg", "")}})
    return Executor(reg, PolicyEngine(pol or policy(root), pause_file=pause_file),
                    ScriptedConfirmer(answers if answers is not None else {"fs.*": True}, default=False), risk_fn=risk)


def make_executor(store: Store) -> Executor:  # CLI: --executor examples/jarvis_wiring.py:make_executor
    return build_executor(root_dir(), answers={"*": True})


def handle_turn(rt: Any, inputs: Dict[str, Any], ex: Executor, root: str) -> Dict[str, Any]:
    text = rt.call("asr", "whisper.stub", {"audio": inputs["audio"]}, asr)
    ctx = recall_call(rt, recall, text, 3, graph_version="g1")
    plan = rt.call("llm", "planner", {"text": text, "context": ctx, "root": root}, planner,
                   meta={"model": "rule-based", "temperature": 0})
    results = [ex.run(rt, a, args, intent=text).status for a, args in plan["actions"]]
    spoken = plan["reply"] + (": " + ", ".join(results) if results else "")
    rt.call("tts", "piper.stub", {"text": spoken}, tts)
    return {"spoken": spoken}


def agent(rt: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:  # replay entrypoint for the CLI
    root = root_dir()
    return handle_turn(rt, inputs, build_executor(root, answers={"*": True}), root)


UTTERANCES = ["dim the lights", "save a note", "edit the note", "wipe the data", "what does bob like", "delete the note"]


def run_demo(root: str, verbose: bool = True) -> Dict[str, Any]:
    say = print if verbose else (lambda *a, **k: None)
    os.makedirs(root, exist_ok=True)
    out: Dict[str, Any] = {}
    store = Store(os.path.join(root, "traces.db"))
    pf = os.path.join(root, "PAUSE")
    ex = build_executor(root, pause_file=pf)
    rec = Recorder(store, env={"llm": "rule-based", "policy": ex.engine.policy_id})

    say("== live turns")
    ids = {}
    for u in UTTERANCES:
        with rec.trace("voice_turn", {"audio": u.encode()}, tags=["voice"]) as rt:
            rt.set_output(handle_turn(rt, rt.inputs, ex, root))
        store.label(rt.trace_id, "golden")
        ids[u] = rt.trace_id
        say(f"   {u!r:28} -> {store.load_trace(rt.trace_id).outputs['spoken']}")
    out["live_outputs"] = {u: store.load_trace(t).outputs["spoken"] for u, t in ids.items()}

    say("\n== audit trail (tier >= 1)")
    for r in audit.list_actions(store, tier_min=1):
        say(f"   T{r.tier} {r.outcome:<8} {r.name:<10} undo={'yes' if r.undo else 'NO '} snapshot={'yes' if r.snapshot else 'no '}")
    out["violations"] = [v.detail for h in store.list_traces() for v in check_trace(store.load_trace(h.trace_id), live_invariants())]
    say(f"   live invariants: {len(out['violations'])} violation(s)")

    say("\n== kill switch")
    killswitch.engage(pf, "demo")
    with rec.trace("voice_turn", {"audio": b"dim the lights"}) as rt:
        out["paused_status"] = ex.run(rt, "kv.set", {"key": "brightness", "value": 99}, intent="while paused").status
    killswitch.release(pf)
    say(f"   kv.set while paused -> {out['paused_status']}")

    say("\n== undo the save/edit/delete turns' effects (newest first)")
    note = os.path.join(root, "note.txt")
    undo = UndoRunner(store, build_executor(root, answers={"*": True}))
    for u in ("delete the note", "edit the note", "save a note"):
        r = undo.run(ids[u], apply=True)
        say(f"   undo {u!r}: {[s['status'] for s in r['steps']]}")
    out["note_exists_after_undo"] = os.path.exists(note)

    say("\n== replay the golden turns")
    def candidate(pol: Dict[str, Any]):
        cex = build_executor(root, pol, answers={"*": True})

        def cand(rt, inputs):
            return handle_turn(rt, inputs, cex, root)
        cand.__name__ = "candidate"
        return cand

    same = run_suite(store, candidate(policy(root)), select={"label": "golden"}, label="same policy")
    loose_p = policy(root)
    loose_p["tier_decisions"]["3"] = "allow"
    loose = run_suite(store, candidate(loose_p), select={"label": "golden"}, label="tier-3 allowed")
    out["same_exit"], out["loose_exit"] = same.exit_code(), loose.exit_code()
    say(f"   same policy   -> exit {same.exit_code()}: {same.summary_line()}")
    say(f"   loosened tier3 -> exit {loose.exit_code()}: {loose.summary_line()}")

    say("\n== policy-diff over every recorded check")
    d = diff_policies(PolicyEngine(policy(root)), PolicyEngine(loose_p), cases_from_store(store))
    out["loosened"] = sorted({r["action"] for r in d.loosened})
    say(f"   loosened actions: {out['loosened']}")
    out["store_ok"] = store.verify().ok
    say(f"\nstore verifies: {out['store_ok']}   chain head: {store.head()[:24]}…")
    return out


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        run_demo(os.path.realpath(d))
