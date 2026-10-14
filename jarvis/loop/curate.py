from __future__ import annotations
import hashlib, json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .prompt import assistant_text, build_messages
from .traces import Trace


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def is_holdout(t: Trace, frac: float = 0.15, salt: str = "jl1") -> bool:
    # split on normalized TASK so identical tasks never straddle train/holdout
    return _h(f"{salt}:{t.norm_task}") % 10000 < int(frac * 10000)


@dataclass
class Curated:
    train: list = field(default_factory=list)
    holdout: list = field(default_factory=list)
    golden: list = field(default_factory=list)       # ids of pinned traces inside holdout
    dropped: dict = field(default_factory=lambda: Counter())


def curate(traces: Iterable[Trace], holdout_frac: float = 0.15, salt: str = "jl1",
           max_per_task: int = 3, max_holdout: int = 400) -> Curated:
    """Train = successful, violation-free, de-duplicated, per-task capped demos.
    Holdout = every trace whose TASK hashes into the holdout bucket (failures kept: they are replay/safety probes).
    Golden-tagged traces pin their whole task into holdout and never reach training."""
    traces = list(traces)
    golden_tasks = {t.norm_task for t in traces if t.golden}
    out = Curated()
    seen_hold_keys, seen_hold_ids, seen_train = set(), set(), set()
    per_task = Counter()
    for t in traces:
        if not t.task.strip():
            out.dropped["empty"] += 1
            continue
        if t.norm_task in golden_tasks or is_holdout(t, holdout_frac, salt):
            if t.id in seen_hold_ids or t.key in seen_hold_keys:
                out.dropped["duplicate"] += 1
                continue
            seen_hold_ids.add(t.id)
            seen_hold_keys.add(t.key)
            out.holdout.append(t)
            if t.golden:
                out.golden.append(t.id)
            continue
        if not t.success:
            out.dropped["failed"] += 1
            continue
        if t.violations > 0:
            out.dropped["violation"] += 1          # never learn from tier violations
            continue
        if t.key in seen_train:
            out.dropped["duplicate"] += 1
            continue
        if per_task[t.norm_task] >= max_per_task:
            out.dropped["capped"] += 1
            continue
        seen_train.add(t.key)
        per_task[t.norm_task] += 1
        out.train.append(t)
    if len(out.holdout) > max_holdout:
        pinned = [t for t in out.holdout if t.golden]
        rest = sorted((t for t in out.holdout if not t.golden), key=lambda t: _h(salt + t.key))
        keep = pinned + rest[: max(0, max_holdout - len(pinned))]
        out.dropped["holdout_capped"] += len(out.holdout) - len(keep)
        out.holdout = keep
    out.dropped = dict(out.dropped)
    for k in ("failed", "violation", "duplicate", "empty", "capped", "holdout_capped"):
        out.dropped.setdefault(k, 0)
    return out


def to_sft(t: Trace) -> dict:
    msgs = build_messages(t.task, t.context, t.granted_tier)
    msgs.append({"role": "assistant", "content": assistant_text(t.calls)})
    return {"messages": msgs, "meta": {"trace_id": t.id}}


def write_dataset(c: Curated, outdir) -> dict:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "train.jsonl", "w", encoding="utf-8") as f:
        for t in c.train:
            f.write(json.dumps(to_sft(t)) + "\n")
    with open(p / "holdout.jsonl", "w", encoding="utf-8") as f:
        for t in c.holdout:
            f.write(json.dumps(t.raw) + "\n")
    (p / "golden.json").write_text(json.dumps(sorted(c.golden)))
    stats = {"train": len(c.train), "holdout": len(c.holdout), "golden": len(c.golden), "dropped": c.dropped}
    (p / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
