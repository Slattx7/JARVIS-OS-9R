from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from .config import LoopConfig
from .curate import curate, write_dataset
from .gate import GateConfig, Outcome, gate
from .registry import Registry
from .traces import load_traces


def _outcomes(path) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            ok = d.get("ok")
            out[str(d["id"])] = Outcome(None if ok is None else bool(ok), int(d.get("violations", 0)),
                                        float(d.get("latency_ms", 0.0)), d.get("error", ""))
    return out


def _p(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("jarvis-loop", description="Trace-driven, replay-gated self-improvement loop.")
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp.add_parser("example-config", help="print a starter loop.json")
    p = sp.add_parser("init", help="bootstrap the baseline version (no gate)")
    p.add_argument("--config", required=True)
    p.add_argument("--version", default="v0")
    p = sp.add_parser("plan", help="VRAM-fit a QLoRA plan for a model's HF config.json")
    p.add_argument("model_config")
    p.add_argument("--vram", type=float, default=6.0)
    p.add_argument("--seq-cap", type=int, default=2048)
    p.add_argument("--no-quant", action="store_true")
    p = sp.add_parser("curate", help="traces -> train.jsonl + holdout")
    p.add_argument("traces")
    p.add_argument("out")
    p.add_argument("--holdout", type=float, default=0.15)
    p = sp.add_parser("gate", help="gate two outcome files ({id,ok,violations,latency_ms}); exit 0 iff promote")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--golden")
    p.add_argument("--registry")
    p.add_argument("--version")
    p = sp.add_parser("cycle", help="run one improvement cycle")
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("regate", help="re-run the gate for an existing version vs CURRENT (no retraining)")
    p.add_argument("version")
    p.add_argument("--config", required=True)
    for name, hlp in (("monitor", "post-deploy live check; auto-rollback"), ("status", "show registry state"),
                      ("resume", "lift the kill switch")):
        p = sp.add_parser(name, help=hlp)
        p.add_argument("--config", required=True)
    p = sp.add_parser("pause", help="kill switch: blocks promotion")
    p.add_argument("--config", required=True)
    p.add_argument("--reason", default="")
    p = sp.add_parser("rollback", help="move CURRENT back one promotion")
    p.add_argument("--config", required=True)
    p.add_argument("--reason", default="manual")
    p = sp.add_parser("demo", help="end-to-end run on synthetic data (no GPU)")
    p.add_argument("--dir")
    a = ap.parse_args(argv)

    if a.cmd == "example-config":
        print(LoopConfig(
            root="jarvis_loop_data", traces="traces/", base_model="Qwen/Qwen2.5-1.5B-Instruct",
            model_config="/path/to/base-model/config.json",
            tiers={"notes.search": 0, "home.light": 1, "app.open": 1, "shell.run": 3}).to_json())
        return 0
    if a.cmd == "demo":
        from .demo import run_demo
        res = run_demo(a.dir)
        return 0 if (res["cycle1"], res["cycle2"], res["monitor"]) == ("promoted", "rejected", "rollback") else 1
    if a.cmd == "plan":
        from .trainer import DoesNotFit, ModelSpec, estimate_vram, plan_training
        spec = ModelSpec.from_hf_config(a.model_config)
        try:
            plan = plan_training(spec, a.vram, seq_cap=a.seq_cap, quantized=not a.no_quant)
        except DoesNotFit as e:
            print(f"does not fit: {e}", file=sys.stderr)
            return 2
        est = estimate_vram(spec, plan.r, plan.seq_len, plan.micro_batch, quantized=not a.no_quant)
        _p({"params_b": round(spec.n_params() / 1e9, 3), "plan": plan.__dict__,
            "estimate_gib": {k: round(v, 2) for k, v in est.items()}, "note": "heuristic; trainer retries smaller on real OOM"})
        return 0
    if a.cmd == "curate":
        _p(write_dataset(curate(load_traces(a.traces), a.holdout), a.out))
        return 0
    if a.cmd == "gate":
        golden = frozenset(json.load(open(a.golden))) if a.golden else frozenset()
        v = gate(_outcomes(a.baseline), _outcomes(a.candidate), GateConfig(golden=golden))
        _p({"decision": v.decision, "reasons": v.reasons, "stats": v.stats})
        if a.registry and a.version and v.decision == "promote":
            Registry(a.registry).promote(a.version, v)
        return 0 if v.decision == "promote" else 1

    cfg = LoopConfig.load(a.config)
    if a.cmd == "init":
        from .loop import Loop
        _p(Loop(cfg, None, None).init(a.version))
        return 0
    if a.cmd == "cycle":
        from .factory import build_loop
        rep = build_loop(cfg).cycle(force=a.force)
        _p(rep.to_dict())
        return 1 if rep.status in ("error", "rolled_back") else 0
    if a.cmd == "regate":
        from .factory import build_loop
        rep = build_loop(cfg).regate(a.version)
        _p(rep.to_dict())
        return 1 if rep.status in ("error", "rolled_back") else 0
    from .loop import Loop
    loop = Loop(cfg, None, None)
    if a.cmd == "status":
        _p(loop.status())
        return 0
    if a.cmd == "monitor":
        r = loop.monitor_once()
        _p({"action": r.action, "reasons": r.reasons, "stats": r.stats})
        return 1 if r.action == "rollback" else 0
    if a.cmd == "pause":
        loop.reg.pause(a.reason)
        return 0
    if a.cmd == "resume":
        loop.reg.resume()
        return 0
    if a.cmd == "rollback":
        print(loop.reg.rollback(a.reason))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
