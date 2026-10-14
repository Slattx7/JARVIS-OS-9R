from __future__ import annotations
import json, tempfile, time
from pathlib import Path

from .config import LoopConfig
from .factory import build_loop
from .sim import DEMO_TIERS, make_demo_traces


def _live_traces(path, model: str, n: int, violations: int) -> None:
    now = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for i in range(n):
            bad = i < violations
            f.write(json.dumps({"id": f"live-{model}-{i}", "ts": now + i, "task": f"live task {i}", "model": model,
                                "success": not bad, "violations": 1 if bad else 0, "granted_tier": 1,
                                "steps": [] if bad else [{"tool": "notes.search", "args": {"q": "x"}}]}) + "\n")


def run_demo(workdir=None, echo=print) -> dict:
    """End-to-end on synthetic data with the dry trainer + simulated replayer:
    bootstrap -> cycle (promote) -> cycle with a violating candidate (reject) -> live regression -> auto-rollback."""
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="jarvis_loop_demo_"))
    tp = tmp / "traces.jsonl"
    make_demo_traces(tp, n=700, seed=1, model="v0")
    cfg = LoopConfig(
        root=str(tmp / "data"), traces=str(tp), base_model="demo-base", tiers=DEMO_TIERS, min_train=50,
        min_new_examples=20, max_cycles_per_day=10, trainer={"kind": "dry"},
        replay={"kind": "sim", "params": {"skill": {"v0": 0.55, "v0001": 0.80, "v0002": 0.90},
                                           "violators": ["v0002"]}})
    loop = build_loop(cfg)
    out: dict = {"dir": str(tmp)}
    loop.init("v0")
    echo("bootstrapped v0")

    r1 = loop.cycle()
    echo(f"cycle 1 -> {r1.status} {r1.version}: {r1.reason}")
    out["cycle1"] = r1.status

    make_demo_traces(tp, n=300, seed=2, model="v0001")
    r2 = loop.cycle()
    echo(f"cycle 2 -> {r2.status} {r2.version}: {r2.reason}")
    out["cycle2"] = r2.status
    out["current_after_cycle2"] = loop.reg.current()

    _live_traces(tp, "v0001", n=40, violations=6)
    m = loop.monitor_once()
    echo(f"monitor -> {m.action}: {'; '.join(m.reasons)}")
    out["monitor"] = m.action
    out["current_final"] = loop.reg.current()
    echo(f"current = {out['current_final']}   stack = {loop.reg.stack()}")
    return out
