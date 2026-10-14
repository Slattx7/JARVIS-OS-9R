from __future__ import annotations
import hashlib, json, random
from pathlib import Path

from .gate import Outcome
from .refs import ModelRef
from .traces import Trace

DEMO_TIERS = {"notes.search": 0, "home.light": 1, "app.open": 1, "shell.run": 3}


def _u(s: str) -> float:
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) / float(16 ** 12)


class SimReplayer:
    """Deterministic fake replayer for demos/tests. ok iff u(trace) < skill[version] with u shared across versions
    (paired design: a higher-skill version succeeds on a superset). `violators` emit tier violations on ~5% of traces."""

    def __init__(self, skill=None, default=0.6, violators=(), latency=None):
        self.skill, self.default = dict(skill or {}), default
        self.violators, self.latency = set(violators), dict(latency or {})

    def prepare(self, ref: ModelRef) -> None:
        pass

    def replay(self, ref: ModelRef, trace: Trace) -> Outcome:
        u = _u("t:" + trace.id)
        ok = None if not trace.success else (u < self.skill.get(ref.version, self.default))
        viol = 1 if ref.version in self.violators and _u("v:" + trace.id) < 0.05 else 0
        return Outcome(ok, viol, self.latency.get(ref.version, 100.0))


def make_demo_traces(path, n=600, seed=1, model="v0", start_ts=1.7e9) -> int:
    rng = random.Random(seed)
    rooms = ["kitchen", "bedroom", "office", "hall", "garage", "porch"]
    apps = ["browser", "terminal", "notes", "music", "camera"]
    topics = ["invoices", "router", "recipes", "frida", "vram", "taxes", "gym", "travel"]
    scripts = ["backup", "cleanup", "sync", "update"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for i in range(n):
            kind = rng.choice(["light", "app", "search", "shell"])
            if kind == "light":
                room, on = rng.choice(rooms), rng.random() < 0.5
                task, calls, tier = f"turn {'on' if on else 'off'} the {room} light {rng.randint(0, 40)}", \
                    [{"tool": "home.light", "args": {"room": room, "on": on}}], 1
            elif kind == "app":
                a = rng.choice(apps)
                task, calls, tier = f"open {a} {rng.randint(0, 40)}", [{"tool": "app.open", "args": {"name": a}}], 1
            elif kind == "search":
                tp = rng.choice(topics)
                task, calls, tier = f"find my notes about {tp} {rng.randint(0, 40)}", \
                    [{"tool": "notes.search", "args": {"q": tp}}], 0
            else:
                s = rng.choice(scripts)
                task, calls, tier = f"run the {s} script {rng.randint(0, 40)}", [{"tool": "shell.run", "args": {"script": s}}], 3
            granted = tier if rng.random() < 0.85 else max(0, tier - 2)
            r = rng.random()
            violation = granted < tier and r < 0.5
            success = (not violation) and r < 0.9
            rec = {"id": f"{model}-{seed}-{i}", "ts": start_ts + seed * 1e6 + i, "task": task, "granted_tier": granted,
                   "context": "", "model": model, "success": success, "latency_ms": 80 + rng.random() * 40,
                   "violations": 1 if violation else 0, "steps": calls if success or violation else calls[:0]}
            f.write(json.dumps(rec) + "\n")
    return n
