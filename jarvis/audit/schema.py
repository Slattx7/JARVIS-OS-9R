"""Trace schema: kinds, tiers, dataclasses, canonical JSON + normalization.

Stdlib only. Everything persisted goes through normalize() -> canonical().
Schema version 1.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import hashlib
import json
import math
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

SCHEMA_VERSION = 1

OK, ERROR, BLOCKED, RUNNING, ORPHANED = "ok", "error", "blocked", "running", "orphaned"


class Kind(str, enum.Enum):
    """Span kinds. `kind` is a free string on the wire; these are the known ones."""

    WAKE = "wake"
    ASR = "asr"
    RECALL = "recall"
    LLM = "llm"
    PERMISSION = "permission"  # policy.check: inputs {action,args} -> {tier,decision,rule}
    CONFIRM = "confirm"        # user confirmation round-trip: -> {approved: bool}
    ACTION = "action"          # device/tool execution. inputs == args dict, name == action id
    CONTEXT = "context"        # policy context snapshot (recent-action counts, pause flag, hour, ...)
    RISK = "risk"              # external risk score for an action (e.g. a classifier): -> {"risk": 0..1}
    TTS = "tts"
    CUSTOM = "custom"


ALL_KINDS = frozenset(k.value for k in Kind)


def kind_value(k: Any) -> str:
    return k.value if isinstance(k, enum.Enum) else str(k)


class Tier(enum.IntEnum):
    """Action-tier model the trace schema carries. The action layer owns semantics."""

    OBSERVE = 0        # read-only
    REVERSIBLE = 1     # state change with automatic undo
    CONSEQUENTIAL = 2  # hard to undo / externally visible -> confirm
    DESTRUCTIVE = 3    # irreversible / system-level -> confirm + snapshot


class Decision(str, enum.Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


STRICTNESS = {"allow": 0, "confirm": 1, "deny": 2}


def new_id(prefix: str = "") -> str:
    """Time-sortable id."""
    return f"{prefix}{time.time_ns():x}{os.urandom(3).hex()}"


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def blob_ref(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_normalized(norm: Any) -> str:
    return hashlib.sha256(canonical(norm).encode("utf-8")).hexdigest()[:24]


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    return ys[f] + (ys[c] - ys[f]) * (k - f)


# --------------------------------------------------------------------------- #
# normalization: arbitrary python -> canonical-JSON-safe, redacted, blob-ref'd
# --------------------------------------------------------------------------- #


@dataclass
class NormCtx:
    redactor: Any = None                                  # redact.Redactor | None
    put_blob: Optional[Callable[[bytes], str]] = None     # None => compute ref only (hashing path)
    max_inline: int = 64 * 1024
    max_depth: int = 40


def _blob(data: bytes, ctx: NormCtx, extra: Dict[str, Any]) -> Dict[str, Any]:
    ref = ctx.put_blob(data) if ctx.put_blob else blob_ref(data)
    out: Dict[str, Any] = {"__blob__": ref, "len": len(data)}
    out.update(extra)
    return out


def _norm_str(s: str, ctx: NormCtx) -> str | Dict[str, Any]:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = s.encode("utf-8", "replace").decode("utf-8")
    if ctx.redactor is not None:
        s = ctx.redactor.scrub_str(s)
    if len(s) > ctx.max_inline:
        return _blob(s.encode("utf-8"), ctx, {"as": "str"})
    return s


def _norm_dict(obj: dict, ctx: NormCtx, depth: int) -> Dict[str, Any]:
    if isinstance(obj.get("__blob__"), str):  # already-normalized blob ref: pass through untouched
        return {str(k): v for k, v in obj.items()}
    red = ctx.redactor
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        ks = k if isinstance(k, str) else str(k)
        if red is not None and not isinstance(v, bool):
            if isinstance(v, (str, bytes, bytearray)) and red.is_secret_key(ks):
                if not (isinstance(v, str) and red.is_placeholder(v)):
                    raw = bytes(v) if not isinstance(v, str) else v
                    out[ks] = red.placeholder("secret", raw)
                else:
                    out[ks] = v
                continue
            if isinstance(v, (int, float)) and red.is_numeric_secret_key(ks):
                out[ks] = red.placeholder("secret", str(v))
                continue
        if red is not None and isinstance(v, str) and red.is_exempt_key(ks):
            out[ks] = _norm_str(v, dataclasses.replace(ctx, redactor=None))
            continue
        out[ks] = normalize(v, ctx, depth)
    return out


def normalize(obj: Any, ctx: NormCtx, _depth: int = 0) -> Any:
    if _depth > ctx.max_depth:
        return {"__truncated__": "depth"}
    d = _depth + 1
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, enum.Enum):
        return normalize(obj.value, ctx, d)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else {"__float__": repr(obj)}
    if isinstance(obj, str):
        return _norm_str(obj, ctx)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return _blob(bytes(obj), ctx, {})
    if isinstance(obj, dict):
        return _norm_dict(obj, ctx, d)
    if isinstance(obj, (list, tuple)):
        return [normalize(x, ctx, d) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((normalize(x, ctx, d) for x in obj), key=canonical)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return normalize({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}, ctx, d)
    if isinstance(obj, pathlib.PurePath):
        return _norm_str(str(obj), ctx)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy-style
        nb = getattr(obj, "nbytes", None)
        if isinstance(nb, int) and nb > ctx.max_inline and hasattr(obj, "tobytes"):
            return _blob(obj.tobytes(), ctx, {"as": "ndarray", "dtype": str(getattr(obj, "dtype", "")),
                                               "shape": list(getattr(obj, "shape", ()))})
        return normalize(tolist(), ctx, d)
    r = repr(obj)[:200]
    return {"__repr__": ctx.redactor.scrub_str(r) if ctx.redactor is not None else r,
            "type": type(obj).__name__}


def safe_canonical(obj: Any) -> str:
    """canonical() for raw values (bytes, dataclasses, enums, ...): normalize first, never raises on odd types.
    For previews and dedupe keys, not for hashing recorded data."""
    try:
        return canonical(normalize(obj, NormCtx()))
    except Exception:  # noqa: BLE001
        return repr(obj)


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_id: Optional[str]
    kind: str
    name: str
    t_start_ns: int
    dur_ns: Optional[int] = None
    inputs: Any = None
    outputs: Any = None
    status: str = RUNNING
    error: Optional[dict] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""
    seq: int = 0

    @property
    def dur_ms(self) -> Optional[float]:
        return None if self.dur_ns is None else self.dur_ns / 1e6


@dataclass
class Trace:
    trace_id: str
    name: str
    t_start_ns: int
    dur_ns: Optional[int] = None
    inputs: Any = None
    outputs: Any = None
    status: str = RUNNING
    error: Optional[dict] = None
    tags: List[str] = field(default_factory=list)
    env: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    labels: List[dict] = field(default_factory=list)
    spans: List[Span] = field(default_factory=list)
    schema: int = SCHEMA_VERSION

    def spans_of(self, kind: Any = None, name: Optional[str] = None) -> List[Span]:
        k = None if kind is None else kind_value(kind)
        return [s for s in self.spans if (k is None or s.kind == k) and (name is None or s.name == name)]

    def actions(self) -> List[Span]:
        return self.spans_of("action")

    def label_set(self) -> set:
        out: set = set()
        for l in self.labels:
            if l.get("op", "add") == "remove":
                out.discard(l["label"])
            else:
                out.add(l["label"])
        return out

    def pre_tts_ms(self) -> float:
        """Estimated wake->first-audio latency: sum of root-level span durations before the first TTS span.

        Assumes sequential root-level stages. Frozen (replayed) spans carry their recorded duration,
        so the number is comparable between a baseline trace and its replay.
        """
        total = 0
        for s in sorted((s for s in self.spans if s.parent_id is None), key=lambda s: s.seq):
            if s.kind == "tts":
                break
            total += s.dur_ns or 0
        return total / 1e6

    def first_audio_ms(self) -> float:
        """pre_tts_ms plus the first TTS span's `first_audio_ms` meta when the TTS stage reports it (streaming TTS:
        `annotate(first_audio_ms=...)` when the first chunk plays). Falls back to pre_tts_ms."""
        extra = 0.0
        for s in sorted((s for s in self.spans if s.kind == "tts" and s.parent_id is None), key=lambda s: s.seq):
            v = s.meta.get("first_audio_ms")
            extra = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0
            break
        return self.pre_tts_ms() + extra

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Divergence:
    kind: str
    name: str
    reason: str  # input_mismatch | no_recording | unrequested
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
