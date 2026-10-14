"""Per-turn stage tracing -> JSONL, plus percentile reports. Latency work starts here."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# (metric, start_mark, end_mark)
STAGES = [
    ("stt_ms", "speech_end", "stt_done"),
    ("llm_ttft_ms", "stt_done", "llm_first_token"),
    ("first_sentence_ms", "llm_first_token", "first_sentence"),
    ("tts_ttfa_ms", "first_sentence", "tts_first_pcm"),
    ("out_buffer_ms", "tts_first_pcm", "audio_first_out"),
    ("e2e_ms", "speech_end", "audio_first_out"),
]
PERCEIVED_TARGET_MS = 1500.0


def derive_metrics(marks: dict[str, float], meta: dict) -> dict[str, float]:
    m: dict[str, float] = {}
    for name, a, b in STAGES:
        if a in marks and b in marks:
            m[name] = round((marks[b] - marks[a]) * 1000.0, 1)
    if "e2e_ms" in m and "endpoint_ms" in meta:
        m["perceived_ms"] = round(m["e2e_ms"] + meta["endpoint_ms"], 1)
    return m


class TurnTrace:
    def __init__(self, tracer: "Tracer", turn_id: int, kind: str = "user"):
        self.tracer, self.id, self.kind = tracer, turn_id, kind
        self.marks: dict[str, float] = {}
        self.meta: dict = {}
        self._done = False

    def mark(self, name: str, t: float | None = None) -> None:
        self.marks.setdefault(name, self.tracer.clock() if t is None else t)

    def finish(self, **meta) -> dict | None:
        if self._done:
            return None
        self._done = True
        self.mark("turn_end")
        self.meta.update(meta)
        return self.tracer.write(self)


class Tracer:
    def __init__(self, path: str | os.PathLike | None, log_text: bool = False, clock=time.monotonic):
        self.path = Path(os.path.expanduser(str(path))) if path else None
        self.log_text = log_text
        self.clock = clock
        self._n = 0
        self._lock = threading.Lock()
        self.records: list[dict] = []

    def start_turn(self, kind: str = "user") -> TurnTrace:
        with self._lock:
            self._n += 1
            return TurnTrace(self, self._n, kind)

    def write(self, t: TurnTrace) -> dict:
        meta = dict(t.meta)
        if "text" in meta and not self.log_text:
            meta["text_chars"] = len(meta.pop("text"))
        t0 = min(t.marks.values()) if t.marks else 0.0
        rec = {
            "id": t.id, "kind": t.kind, "ts": round(time.time(), 3),
            "marks_ms": {k: round((v - t0) * 1000.0, 1) for k, v in sorted(t.marks.items(), key=lambda kv: kv[1])},
            "metrics": derive_metrics(t.marks, meta), "meta": meta,
        }
        with self._lock:
            self.records.append(rec)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return rec


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        raise ValueError("empty")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stats(by: dict[str, list[float]]) -> dict:
    return {k: {"n": len(v), "p50": round(percentile(v, .5), 1), "p95": round(percentile(v, .95), 1),
                "max": round(max(v), 1)} for k, v in by.items()}


def summarize(records: list[dict]) -> dict:
    """LLM turns drive the headline stats/target; fastpath turns (no LLM) are reported separately."""
    by: dict[str, list[float]] = {}
    fast: dict[str, list[float]] = {}
    done = interrupted = fast_turns = 0
    for r in records:
        if r.get("kind") != "user":
            continue
        meta = r.get("meta", {})
        if meta.get("interrupted"):
            interrupted += 1
        is_fast = bool(meta.get("fast"))
        if is_fast:
            fast_turns += 1
        elif not meta.get("interrupted"):
            done += 1
        for k, v in r.get("metrics", {}).items():
            (fast if is_fast else by).setdefault(k, []).append(v)
    stats = _stats(by)
    p = stats.get("perceived_ms")
    return {"turns": done, "interrupted": interrupted, "fast_turns": fast_turns, "stats": stats,
            "fast_stats": _stats(fast),
            "target_perceived_p50_ms": PERCEIVED_TARGET_MS,
            "meets_target": (p["p50"] < PERCEIVED_TARGET_MS) if p else None}


def load_records(path: str | os.PathLike) -> list[dict]:
    out = []
    with open(os.path.expanduser(str(path))) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def format_report(summary: dict) -> str:
    lines = [f"llm_turns={summary['turns']} fast_turns={summary.get('fast_turns', 0)} interrupted={summary['interrupted']}",
             f"{'metric':<20}{'n':>5}{'p50':>9}{'p95':>9}{'max':>9}"]
    order = [st[0] for st in STAGES] + ["perceived_ms"]
    for k in order:
        st = summary["stats"].get(k)
        if st:
            lines.append(f"{k:<20}{st['n']:>5}{st['p50']:>9}{st['p95']:>9}{st['max']:>9}")
    mt = summary["meets_target"]
    if mt is not None:
        lines.append(f"perceived p50 target <{int(summary['target_perceived_p50_ms'])}ms: {'PASS' if mt else 'FAIL'}")
    fs = summary.get("fast_stats", {})
    if fs.get("perceived_ms"):
        st = fs["perceived_ms"]
        lines.append(f"fastpath (no LLM) perceived: n={st['n']} p50={st['p50']} p95={st['p95']} max={st['max']}")
    return "\n".join(lines)
