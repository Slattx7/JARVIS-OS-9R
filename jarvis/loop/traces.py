from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

# Adapter: maps whatever jarvis_trace emits onto Trace. Extend these alias lists, not the code.
ALIASES = {
    "id": ("id", "trace_id"),
    "task": ("task", "utterance", "input", "prompt"),
    "steps": ("steps", "actions", "tool_calls"),
    "model": ("model", "model_version"),
    "latency_ms": ("latency_ms", "latency"),
    "context": ("context", "recall", "recalled", "retrieved"),
    "ts": ("ts", "timestamp", "time", "created_at"),
    "granted_tier": ("granted_tier", "tier", "max_tier"),
    "tags": ("tags", "labels"),
}


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _pick(d: dict, key: str, default=None):
    for k in ALIASES[key]:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _ts(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _norm_step(s) -> dict:
    if isinstance(s, str):
        return {"tool": s, "args": {}}
    d = dict(s) if isinstance(s, dict) else {"tool": str(s)}
    d["tool"] = str(d.get("tool", d.get("name", d.get("action", ""))))
    args = d.get("args", d.get("arguments", d.get("params", {})))
    d["args"] = args if isinstance(args, dict) else {"value": args}
    return d


def _success(d: dict) -> bool:
    for k in ("success", "ok"):
        if k in d:
            return bool(d[k])
    o = d.get("outcome")
    if isinstance(o, dict):
        return bool(o.get("success", o.get("ok", False)))
    if isinstance(o, str):
        return o.strip().lower() in ("success", "ok", "done", "completed")
    return False


@dataclass
class Trace:
    id: str
    task: str
    steps: list
    success: bool
    violations: int = 0          # action-tier violations (attempted calls above the granted tier / denied)
    latency_ms: float = 0.0
    model: str = ""              # registry version that produced this trace (Jarvis must stamp this)
    context: str = ""            # recalled context fed to the planner
    granted_tier: Optional[int] = None
    ts: float = 0.0
    tags: tuple = ()
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def norm_task(self) -> str:
        return " ".join(self.task.lower().split())

    @property
    def calls(self) -> list:
        return [{"tool": s["tool"], "args": s["args"]} for s in self.steps]

    @property
    def key(self) -> str:
        return hashlib.sha1(f"{self.norm_task}::{canon(self.calls)}".encode()).hexdigest()

    @property
    def golden(self) -> bool:
        return "golden" in self.tags or self.raw.get("golden") is True

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        steps = [_norm_step(s) for s in (_pick(d, "steps", []) or [])]
        viol = d.get("violations")
        if isinstance(viol, (list, tuple)):
            viol = len(viol)
        if viol is None:
            viol = sum(1 for s in steps if s.get("tier_violation") or s.get("denied"))
        ctx = _pick(d, "context", "")
        if isinstance(ctx, (list, tuple)):
            ctx = "\n".join(str(x) for x in ctx)
        gt = _pick(d, "granted_tier")
        tags = _pick(d, "tags", ()) or ()
        return cls(
            id=str(_pick(d, "id", hashlib.sha1(canon(d).encode()).hexdigest()[:12])),
            task=str(_pick(d, "task", "")),
            steps=steps,
            success=_success(d),
            violations=int(viol),
            latency_ms=float(_pick(d, "latency_ms", 0.0) or 0.0),
            model=str(_pick(d, "model", "")),
            context=str(ctx),
            granted_tier=None if gt is None else int(gt),
            ts=_ts(_pick(d, "ts")),
            tags=tuple(str(t) for t in tags),
            raw=d,
        )


def load_traces(path, strict: bool = False, stats: Optional[dict] = None) -> Iterator[Trace]:
    """Stream traces from a .jsonl file or a directory of them. Bad lines are skipped (counted in stats)."""
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    st = stats if stats is not None else {}
    st.setdefault("bad_lines", 0)
    st.setdefault("loaded", 0)
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if not isinstance(d, dict):
                        raise ValueError("not an object")
                    t = Trace.from_dict(d)
                except Exception as e:
                    if strict:
                        raise ValueError(f"{fp}:{n}: {e}") from e
                    st["bad_lines"] += 1
                    continue
                st["loaded"] += 1
                yield t
