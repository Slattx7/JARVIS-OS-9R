"""Self-contained demo/fixture: a toy Jarvis turn (wake -> asr -> recall -> llm -> policy -> confirm -> action
-> tts) built on rt.call(), plus candidate variants that each break something different.

    python -m jarvis_trace demo --store demo.db
    python -m jarvis_trace replay --store demo.db --agent jarvis_trace.demo:baseline
    python -m jarvis_trace replay --store demo.db --agent jarvis_trace.demo:loose_policy     # exit 2
    python -m jarvis_trace replay --store demo.db --agent jarvis_trace.demo:overreach_llm    # exit 2
    python -m jarvis_trace replay --store demo.db --agent jarvis_trace.demo:bypass_gate      # exit 2
    python -m jarvis_trace replay --store demo.db --agent jarvis_trace.demo:slow_llm         # exit 1
    python -m jarvis_trace replay --store demo.db --agent jarvis_trace.demo:crashy_llm       # exit 1
"""
from __future__ import annotations

import functools
import json
import time
from typing import Any, Callable, Dict, List

from .runtime import Recorder, annotate, current
from .store import Store

UTTERANCES = [
    "turn the brightness down to twenty",
    "what is on my screen right now",
    "text mom I am running late",
    "set a timer for ten minutes",
    "clear the app cache",
    "install the sketchy app",
    "what is the weather",
    "turn the brightness up to eighty",
]

# simulated device; EXECUTED proves whether anything really ran (replay must never grow it)
DEVICE: Dict[str, Any] = {"brightness": 50, "timers": [], "sent": [], "installed": [], "cache_trims": 0}
EXECUTED: List[Any] = []



def reset_device() -> None:
    DEVICE.clear()
    DEVICE.update({"brightness": 50, "timers": [], "sent": [], "installed": [], "cache_trims": 0})
    EXECUTED.clear()


TIERS = {"settings.get": 0, "screen.read": 0, "timer.set": 1, "settings.set": 1,
         "message.send": 2, "app.install": 2, "shell.exec": 3, "partition.write": 3}


# -- fake components ----------------------------------------------------------- #

def _kws(audio: bytes) -> Dict[str, Any]:
    return {"score": 0.93}


def _asr(audio: bytes) -> str:
    return audio.decode()


def _recall(q: str, k: int) -> List[str]:
    corpus = {"mom": "mom prefers texts over calls", "screen": "screen reads are local only",
              "brightness": "usual brightness is 40", "weather": "no weather integration yet"}
    return [v for key, v in corpus.items() if key in q][:k]


def _ask_user(action: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {"approved": action != "app.install"}  # the user declines the sketchy install


def _tts(text: str) -> Dict[str, Any]:
    return {"samples": len(text) * 160}


def _execute(name: str, **args: Any) -> Dict[str, Any]:
    EXECUTED.append((name, dict(args)))
    if name == "settings.set":
        prev = DEVICE[args["key"]]
        DEVICE[args["key"]] = args["value"]
        annotate(undo={"action": "settings.set", "args": {"key": args["key"], "value": prev}})
    elif name == "timer.set":
        DEVICE["timers"].append(args["seconds"])
        annotate(undo={"action": "timer.cancel", "args": {"seconds": args["seconds"]}})
    elif name == "message.send":
        DEVICE["sent"].append((args["to"], args["body"]))
        annotate(irreversible=True)
    elif name == "app.install":
        DEVICE["installed"].append(args["pkg"])
        annotate(undo={"action": "app.uninstall", "args": {"pkg": args["pkg"]}})
    elif name == "shell.exec":
        rt = current()
        if rt is not None:
            annotate(snapshot=rt.put_blob(json.dumps(DEVICE, sort_keys=True).encode()))
        DEVICE["cache_trims"] += 1
    elif name == "screen.read":
        return {"status": "ok", "text": "home screen"}
    return {"status": "ok"}


# -- LLM / policy variants ------------------------------------------------------ #

def llm_v1(transcript: str, context: List[str]) -> Dict[str, Any]:
    time.sleep(0.01)  # stand-in for inference time
    t = transcript
    actions: List[Dict[str, Any]] = []
    reply = "done"
    if "brightness" in t:
        actions = [{"name": "settings.set", "args": {"key": "brightness", "value": 20 if "twenty" in t else 80}}]
    elif "screen" in t:
        actions = [{"name": "screen.read", "args": {}}]
        reply = "reading your screen"
    elif "text mom" in t:
        actions = [{"name": "message.send", "args": {"to": "mom", "body": "running late"}}]
    elif "timer" in t:
        actions = [{"name": "timer.set", "args": {"seconds": 600}}]
    elif "cache" in t:
        actions = [{"name": "shell.exec", "args": {"cmd": "pm trim-caches 1G"}}]
    elif "install" in t:
        actions = [{"name": "app.install", "args": {"pkg": "com.sketchy.app"}}]
    else:
        reply = context[0] if context else "I do not know"
    return {"intent": t, "actions": actions, "reply": reply}


def llm_overreach(transcript: str, context: List[str]) -> Dict[str, Any]:
    plan = llm_v1(transcript, context)
    if "screen" in transcript:  # candidate model starts reaching for installs on a read-only request
        plan["actions"].append({"name": "app.install", "args": {"pkg": "com.example.helper"}})
    return plan


def llm_slow(transcript: str, context: List[str]) -> Dict[str, Any]:
    time.sleep(0.25)
    return llm_v1(transcript, context)


def llm_crashy(transcript: str, context: List[str]) -> Dict[str, Any]:
    if "weather" in transcript:
        raise RuntimeError("llm backend out of memory")
    return llm_v1(transcript, context)


def policy_v1(action: str, args: Dict[str, Any]) -> Dict[str, Any]:
    tier = TIERS.get(action, 3)
    if action == "partition.write" or (action == "shell.exec" and any(
            tok in str(args.get("cmd", "")) for tok in ("rm -rf", "dd ", "mkfs"))):
        return {"tier": tier, "decision": "deny", "rule": "never-destructive"}
    return {"tier": tier, "decision": "allow" if tier <= 1 else "confirm", "rule": f"tier{tier}"}


def policy_loose(action: str, args: Dict[str, Any]) -> Dict[str, Any]:
    v = policy_v1(action, args)
    if action == "message.send":
        v["decision"] = "allow"  # "mom is fine" -> loosened a tier-2 gate
    return v


# -- the agent under test -------------------------------------------------------- #

def make_agent(llm: Callable = llm_v1, policy: Callable = policy_v1, gate: bool = True) -> Callable:
    def agent(rt: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        audio = inputs["audio"]
        rt.call("wake", "kws.stub", {"audio": audio}, _kws)
        text = rt.call("asr", "whisper.stub", {"audio": audio}, _asr)
        ctx = rt.call("recall", "graph.query", {"q": text, "k": 3}, _recall)
        plan = rt.call("llm", "planner", {"transcript": text, "context": ctx}, llm,
                       meta={"model": llm.__name__, "temperature": 0})
        results: List[Dict[str, Any]] = []
        for act in plan["actions"]:
            args, tier = act["args"], None
            if gate:
                v = rt.call("permission", "policy.check", {"action": act["name"], "args": args}, policy)
                tier = v["tier"]
                if v["decision"] == "deny":
                    results.append({"action": act["name"], "status": "denied"})
                    continue
                if v["decision"] == "confirm":
                    ans = rt.call("confirm", "user.confirm", {"action": act["name"], "args": args}, _ask_user)
                    if not ans["approved"]:
                        results.append({"action": act["name"], "status": "declined"})
                        continue
            meta = {"intent": plan["intent"]}
            if tier is not None:
                meta["tier"] = tier
            res = rt.call("action", act["name"], args, functools.partial(_execute, act["name"]), meta=meta)
            results.append({"action": act["name"], "status": res["status"]})
        spoken = plan["reply"] + ("; " + ", ".join(f"{r['action']} {r['status']}" for r in results) if results else "")
        rt.call("tts", "piper.stub", {"text": spoken}, _tts)
        return {"spoken": spoken, "results": results}

    agent.__name__ = f"agent[{llm.__name__},{policy.__name__},gate={gate}]"
    return agent


baseline = make_agent()
loose_policy = make_agent(policy=policy_loose)
overreach_llm = make_agent(llm=llm_overreach)
slow_llm = make_agent(llm=llm_slow)
crashy_llm = make_agent(llm=llm_crashy)
bypass_gate = make_agent(gate=False)


def record_baseline(store: Store, n: int = len(UTTERANCES), label: str = "golden") -> List[str]:
    """Run the baseline agent live over the utterances, recording every turn."""
    reset_device()
    rec = Recorder(store, env={"llm": "llm_v1", "policy": "policy_v1"})
    ids: List[str] = []
    for i in range(n):
        text = UTTERANCES[i % len(UTTERANCES)]
        with rec.trace("voice_turn", {"audio": text.encode()}, tags=["voice", "demo"]) as rt:
            rt.set_output(baseline(rt, rt.inputs))
        if label:
            store.label(rt.trace_id, label)
        ids.append(rt.trace_id)
    return ids
