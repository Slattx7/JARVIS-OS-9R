"""Runtime: one API for live recording and deterministic replay.

Agent code talks to a TraceRuntime:

    out = rt.call("recall", "graph.query", {"q": text, "k": 8}, recall.query)   # fn(**inputs)

Live  : fn runs, inputs/outputs/timing are written to the store (write-ahead for ACTION spans).
Replay: spans of *frozen* kinds are served from the baseline trace and fn is NOT called; spans of
        *live* kinds (default: llm, permission, custom) run the candidate code. ACTION spans are always
        frozen: replay can never touch the device. A new action the candidate proposes that has no
        recording is returned as a `blocked` stub and logged as a divergence.
"""
from __future__ import annotations

import contextvars
import copy
import functools
import inspect
import logging
import platform
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from .schema import (BLOCKED, ERROR, OK, SCHEMA_VERSION, Divergence, NormCtx, Span, Trace, canonical,
                     hash_normalized, kind_value, new_id, normalize)
from .store import Store

log = logging.getLogger("jarvis_trace")

_RT: "contextvars.ContextVar[Optional[TraceRuntime]]" = contextvars.ContextVar("jt_rt", default=None)
_SPAN: "contextvars.ContextVar[Optional[SpanHandle]]" = contextvars.ContextVar("jt_span", default=None)

DEFAULT_LIVE = frozenset({"llm", "permission", "custom"})
DEFAULT_STUBS: Dict[str, Any] = {
    "action": {"status": "not_executed", "reason": "replay: no recorded result for this action"},
    "confirm": {"approved": False, "reason": "replay: no recorded confirmation"},
    "risk": {"risk": 1.0, "reason": "replay: no recorded risk score (failing closed)"},
    "context": {},
}
# Kinds whose recordings are bound to the exact (kind, name, inputs): never served by position. Serving
# "the next recorded confirm" to a different action would transfer an approval to something nobody approved.
NO_POSITIONAL = frozenset({"action", "confirm", "risk", "context"})
_REPLAY_META_KEYS = frozenset({"frozen", "replay_of", "match", "stub", "executed"})
# Span meta that must stay byte-exact to be executable: an undo spec with a redacted argument cannot undo anything.
# Everything else (inputs, outputs, other meta, context) is redacted before it is written.
EXACT_META_KEYS = frozenset({"undo", "snapshot"})


class AuditUnavailable(RuntimeError):
    """The write-ahead audit record for an ACTION could not be persisted. The action did NOT run."""


class ReplayMiss(RuntimeError):
    """Candidate requested a frozen component call that has no recording."""


class ReplayedError(RuntimeError):
    """Re-raised in replay when the baseline span for this call had errored."""

    def __init__(self, etype: str, msg: str):
        super().__init__(f"{etype}: {msg}")
        self.original_type = etype


def default_env() -> Dict[str, Any]:
    return {"python": platform.python_version(), "os": sys.platform, "schema": SCHEMA_VERSION}


def _preview(obj: Any, n: int = 160) -> str:
    try:
        s = canonical(obj)
    except Exception:
        s = repr(obj)
    return s if len(s) <= n else s[: n - 1] + "…"


class SpanHandle:
    __slots__ = ("rt", "span_id", "kind", "name", "norm_inputs", "ihash", "perf0", "forced_dur_ns",
                 "forced_status", "meta")

    def __init__(self, rt: "TraceRuntime", span_id: str, kind: str, name: str, norm_inputs: Any, ihash: str):
        self.rt = rt
        self.span_id = span_id
        self.kind = kind
        self.name = name
        self.norm_inputs = norm_inputs
        self.ihash = ihash
        self.perf0 = 0
        self.forced_dur_ns: Optional[int] = None
        self.forced_status: Optional[str] = None
        self.meta: Dict[str, Any] = {}

    def annotate(self, **meta: Any) -> None:
        self.rt._annotate(self, meta)


class ReplayPlan:
    def __init__(self, baseline: Trace, base_store: Store, live: Iterable[str] = DEFAULT_LIVE,
                 on_miss: str = "positional", stubs: Optional[Dict[str, Any]] = None):
        if on_miss not in ("positional", "raise", "stub"):
            raise ValueError("on_miss must be positional|raise|stub")
        self.baseline = baseline
        self.base_store = base_store
        self.live = frozenset(kind_value(k) for k in live) - {"action"}
        self.on_miss = on_miss
        self.stubs = dict(DEFAULT_STUBS)
        self.stubs.update(stubs or {})
        self._pool: Dict[Tuple[str, str], List[Span]] = {}
        self._children: Dict[Optional[str], List[str]] = {}
        self._consumed: Set[str] = set()
        for s in baseline.spans:
            self._children.setdefault(s.parent_id, []).append(s.span_id)
            if self.is_frozen(s.kind) and s.status in (OK, ERROR, BLOCKED):
                self._pool.setdefault((s.kind, s.name), []).append(s)

    def is_frozen(self, kind: str) -> bool:
        return kind == "action" or kind not in self.live

    def take(self, kind: str, name: str, ihash: str, allow_position: bool) -> Tuple[Optional[Span], Optional[str]]:
        lst = self._pool.get((kind, name), [])
        for s in lst:
            if s.span_id not in self._consumed and s.input_hash == ihash:
                self._consumed.add(s.span_id)
                return s, "hash"
        if allow_position:
            for s in lst:
                if s.span_id not in self._consumed:
                    self._consumed.add(s.span_id)
                    return s, "position"
        return None, None

    def consume_subtree(self, span_id: str) -> None:
        stack = list(self._children.get(span_id, ()))
        while stack:
            sid = stack.pop()
            self._consumed.add(sid)
            stack.extend(self._children.get(sid, ()))

    def unconsumed(self) -> List[Span]:
        return [s for lst in self._pool.values() for s in lst if s.span_id not in self._consumed]


class TraceRuntime:
    def __init__(self, store: Store, name: str, inputs: Any = None, *, tags: Iterable[str] = (),
                 env: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None,
                 replay: Optional[ReplayPlan] = None):
        self.store = store
        self.name = name
        self.trace_id = new_id("t_")
        self.inputs = inputs
        self.tags = list(tags)
        self.env = dict(env if env is not None else default_env())
        self.meta = dict(meta or {})
        self.replay = replay
        self.divergences: List[Divergence] = []
        self.degraded = False  # True once any non-action event failed to persist
        self._ctx: NormCtx = store.norm_ctx()
        self._outputs: Any = None
        self._outputs_set = False
        self._perf0 = 0
        self._tok: Optional[contextvars.Token] = None
        self._started = False
        self._finished = False

    # -- construction for replay ------------------------------------------- #
    @classmethod
    def for_replay(cls, baseline: Trace, base_store: Store, cand_store: Store, *, live: Iterable[str] = DEFAULT_LIVE,
                   on_miss: str = "positional", label: str = "candidate", stubs: Optional[Dict[str, Any]] = None,
                   env: Optional[Dict[str, Any]] = None) -> "TraceRuntime":
        plan = ReplayPlan(baseline, base_store, live, on_miss, stubs)
        return cls(cand_store, baseline.name, base_store.rehydrate(baseline.inputs),
                   tags=list(baseline.tags) + ["replay"], env=env,
                   meta={"replay_of": baseline.trace_id, "label": label}, replay=plan)

    # -- persistence helper ------------------------------------------------- #
    def _append(self, typ: str, span_id: Optional[str], body: Dict[str, Any], *, durable: bool = False,
                strict: bool = False) -> Optional[int]:
        try:
            return self.store.append(typ, self.trace_id, span_id, body, durable=durable)
        except Exception as e:  # noqa: BLE001
            if strict:
                raise AuditUnavailable(f"audit write failed, refusing to run action: {e}") from e
            self.degraded = True
            log.warning("jarvis_trace: dropped %s event: %s", typ, e)
            return None

    # -- trace lifecycle ---------------------------------------------------- #
    def start(self) -> "TraceRuntime":
        if self._started:
            return self
        self._started = True
        self._perf0 = time.perf_counter_ns()
        try:
            body = {"schema": SCHEMA_VERSION, "name": self.name, "inputs": normalize(self.inputs, self._ctx),
                    "tags": self.tags, "env": normalize(self.env, NormCtx()),
                    "meta": normalize(self.meta, self._ctx)}
        except Exception as e:  # noqa: BLE001
            body = {"schema": SCHEMA_VERSION, "name": self.name, "inputs": {"__unserializable__": str(e)[:200]},
                    "tags": self.tags, "env": {}, "meta": {}}
        self._append("trace_start", None, body)
        return self

    def set_output(self, outputs: Any) -> None:
        self._outputs = outputs
        self._outputs_set = True

    def finish(self, error: Optional[BaseException] = None) -> None:
        if self._finished or not self._started:
            return
        self._finished = True
        body: Dict[str, Any] = {"dur_ns": time.perf_counter_ns() - self._perf0,
                                "status": ERROR if error is not None else OK}
        if error is not None:
            body["error"] = self._err(error)
        if self.replay is not None:
            for s in self.replay.unconsumed():
                self.divergences.append(Divergence(s.kind, s.name, "unrequested",
                                                   {"span_id": s.span_id, "input_hash": s.input_hash}))
            body["divergences"] = [d.to_dict() for d in self.divergences]
        if self._outputs_set:
            try:
                body["outputs"] = normalize(self._outputs, self._ctx)
            except Exception as e:  # noqa: BLE001
                body["outputs"] = {"__unserializable__": str(e)[:200]}
        self._append("trace_end", None, body)

    def __enter__(self) -> "TraceRuntime":
        self.start()
        self._tok = _RT.set(self)
        return self

    def __exit__(self, et: Any, ev: Optional[BaseException], tb: Any) -> bool:
        if self._tok is not None:
            try:
                _RT.reset(self._tok)
            except ValueError:
                pass
            self._tok = None
        self.finish(ev)
        return False

    async def __aenter__(self) -> "TraceRuntime":
        return self.__enter__()

    async def __aexit__(self, et: Any, ev: Optional[BaseException], tb: Any) -> bool:
        return self.__exit__(et, ev, tb)

    # -- spans -------------------------------------------------------------- #
    def put_blob(self, data: bytes) -> Dict[str, Any]:
        """Store bytes (e.g. a pre-destructive snapshot) and return a blob ref usable in annotate()/outputs."""
        return self.store.put_blob(data)

    def _norm_meta(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        meta = meta or {}
        exact = {k: v for k, v in meta.items() if k in EXACT_META_KEYS}
        out = normalize({k: v for k, v in meta.items() if k not in EXACT_META_KEYS}, self._ctx)
        if exact:
            raw_ctx = NormCtx(redactor=None, put_blob=self._ctx.put_blob, max_inline=self._ctx.max_inline)
            out.update(normalize(exact, raw_ctx))
        return out

    def _err(self, exc: BaseException) -> Dict[str, Any]:
        return {"type": type(exc).__name__, "msg": self.store.redactor.scrub_str(str(exc))[:500]}

    def _begin(self, kind: str, name: str, inputs: Any, meta: Optional[Dict[str, Any]]) -> SpanHandle:
        parent = _SPAN.get()
        parent_id = parent.span_id if (parent is not None and parent.rt is self) else None
        strict = kind == "action"
        span_id = new_id("s_")
        try:
            norm = normalize(inputs, self._ctx)
            nmeta = self._norm_meta(meta)
        except Exception as e:  # noqa: BLE001
            if strict:
                raise AuditUnavailable(f"cannot serialize action inputs: {e}") from e
            norm, nmeta = {"__unserializable__": str(e)[:200]}, {}
        ihash = hash_normalized(norm)
        h = SpanHandle(self, span_id, kind, name, norm, ihash)
        h.meta = nmeta if isinstance(nmeta, dict) else {}
        self._append("span_start", span_id,
                     {"parent_id": parent_id, "kind": kind, "name": name, "inputs": norm, "input_hash": ihash,
                      "meta": nmeta}, durable=strict, strict=strict)
        h.perf0 = time.perf_counter_ns()
        return h

    def _annotate(self, h: SpanHandle, meta: Dict[str, Any], durable: Optional[bool] = None) -> None:
        try:
            norm = self._norm_meta(meta)
        except Exception as e:  # noqa: BLE001
            norm = {"__unserializable__": str(e)[:200]}
        self._append("span_annot", h.span_id, {"meta": norm},
                     durable=(h.kind == "action") if durable is None else durable)

    def _end(self, h: SpanHandle, out: Any, exc: Optional[BaseException]) -> None:
        dur = h.forced_dur_ns if h.forced_dur_ns is not None else time.perf_counter_ns() - h.perf0
        if exc is not None:
            body: Dict[str, Any] = {"dur_ns": dur, "status": ERROR, "error": self._err(exc)}
        else:
            body = {"dur_ns": dur, "status": h.forced_status or OK}
            try:
                body["outputs"] = normalize(out, self._ctx)
            except Exception as e:  # noqa: BLE001
                body["outputs"] = {"__unserializable__": str(e)[:200]}
        self._append("span_end", h.span_id, body)

    def _diverge(self, h: SpanHandle, reason: str, detail: Dict[str, Any]) -> None:
        self.divergences.append(Divergence(h.kind, h.name, reason, detail))

    def _maybe_serve(self, h: SpanHandle) -> Tuple[bool, Any]:
        plan = self.replay
        if plan is None or not plan.is_frozen(h.kind):
            return False, None
        base, how = plan.take(h.kind, h.name, h.ihash,
                              plan.on_miss == "positional" and h.kind not in NO_POSITIONAL)
        if base is None:
            if h.kind in plan.stubs:
                out, has_stub = copy.deepcopy(plan.stubs[h.kind]), True
            elif plan.on_miss == "stub":
                out, has_stub = None, True
            else:
                out, has_stub = None, False
            self._diverge(h, "no_recording", {"live_hash": h.ihash, "live_inputs": _preview(h.norm_inputs)})
            if not has_stub:
                raise ReplayMiss(f"no recording for {h.kind}/{h.name} (input_hash={h.ihash}); "
                                 f"inputs={_preview(h.norm_inputs, 120)}")
            h.forced_dur_ns = 0
            h.forced_status = BLOCKED if h.kind == "action" else OK
            self._annotate(h, {"frozen": True, "stub": True, **({"executed": False} if h.kind == "action" else {})})
            return True, out
        if how == "position":
            self._diverge(h, "input_mismatch", {"recorded_hash": base.input_hash, "live_hash": h.ihash,
                                                "recorded_inputs": _preview(base.inputs),
                                                "live_inputs": _preview(h.norm_inputs)})
        plan.consume_subtree(base.span_id)
        h.forced_dur_ns = base.dur_ns or 0
        h.forced_status = base.status
        carry = {k: v for k, v in base.meta.items() if k not in h.meta and k not in _REPLAY_META_KEYS}
        carry = plan.base_store.rehydrate(carry)  # bytes again, so the candidate store re-stores them: no dangling refs
        self._annotate(h, {**carry, "frozen": True, "replay_of": base.span_id, "match": how})  # snapshot/undo/... ride along
        if base.status == ERROR:
            err = base.error or {}
            raise ReplayedError(err.get("type", "Error"), err.get("msg", ""))
        return True, plan.base_store.rehydrate(base.outputs)

    # -- public call API ---------------------------------------------------- #
    def call(self, kind: Any, name: str, inputs: Optional[Dict[str, Any]], fn: Callable[..., Any], *,
             meta: Optional[Dict[str, Any]] = None) -> Any:
        kind = kind_value(kind)
        inputs = {} if inputs is None else inputs
        h = self._begin(kind, name, inputs, meta)
        tok = _SPAN.set(h)
        out: Any = None
        exc: Optional[BaseException] = None
        try:
            served, out = self._maybe_serve(h)
            if not served:
                out = fn(**inputs)
        except BaseException as e:  # noqa: BLE001
            exc = e
            raise
        finally:
            _SPAN.reset(tok)
            self._end(h, out, exc)
        return out

    async def acall(self, kind: Any, name: str, inputs: Optional[Dict[str, Any]], fn: Callable[..., Any], *,
                    meta: Optional[Dict[str, Any]] = None) -> Any:
        """Like call(); fn may be sync or async."""
        kind = kind_value(kind)
        inputs = {} if inputs is None else inputs
        h = self._begin(kind, name, inputs, meta)
        tok = _SPAN.set(h)
        out: Any = None
        exc: Optional[BaseException] = None
        try:
            served, out = self._maybe_serve(h)
            if not served:
                out = fn(**inputs)
                if inspect.isawaitable(out):
                    out = await out
        except BaseException as e:  # noqa: BLE001
            exc = e
            raise
        finally:
            _SPAN.reset(tok)
            self._end(h, out, exc)
        return out


class Recorder:
    def __init__(self, store: Store, *, env: Optional[Dict[str, Any]] = None):
        self.store = store
        self.env = {**default_env(), **(env or {})}

    def trace(self, name: str, inputs: Any = None, *, tags: Iterable[str] = (),
              meta: Optional[Dict[str, Any]] = None) -> TraceRuntime:
        return TraceRuntime(self.store, name, inputs, tags=tags, env=self.env, meta=meta)

    def label(self, trace_id: str, label: str, note: str = "") -> None:
        self.store.label(trace_id, label, note)


# -- ambient helpers --------------------------------------------------------- #


def current() -> Optional[TraceRuntime]:
    return _RT.get()


def annotate(**meta: Any) -> None:
    """Attach metadata (undo spec, snapshot ref, ...) to the innermost open span. Durable for ACTION spans.
    No-op outside a trace."""
    h = _SPAN.get()
    if h is not None:
        h.annotate(**meta)


def traced(kind: Any, name: Optional[str] = None, *, meta: Optional[Dict[str, Any]] = None) -> Callable:
    """Decorator: route a function through the ambient TraceRuntime (live record or replay).
    Outside a trace it is a transparent passthrough. Supports sync + async; no *args signatures."""

    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
            raise TypeError("@traced does not support *args functions")
        first = params[0].name if params else None
        skip_self = first in ("self", "cls")
        nm = name or f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__qualname__}"

        def split(args: tuple, kwargs: dict) -> Tuple[tuple, Dict[str, Any]]:
            b = sig.bind(*args, **kwargs)
            b.apply_defaults()
            d = dict(b.arguments)
            bound: tuple = ()
            if skip_self and first in d:
                bound = (d.pop(first),)
            for p in params:
                if p.kind is inspect.Parameter.VAR_KEYWORD and p.name in d:
                    d.update(d.pop(p.name))
            return bound, d

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*args: Any, **kwargs: Any) -> Any:
                rt = _RT.get()
                if rt is None:
                    return await fn(*args, **kwargs)
                bound, ins = split(args, kwargs)
                return await rt.acall(kind, nm, ins, lambda **kw: fn(*bound, **kw), meta=meta)

            return aw

        @functools.wraps(fn)
        def w(*args: Any, **kwargs: Any) -> Any:
            rt = _RT.get()
            if rt is None:
                return fn(*args, **kwargs)
            bound, ins = split(args, kwargs)
            return rt.call(kind, nm, ins, lambda **kw: fn(*bound, **kw), meta=meta)

        return w

    return deco
