"""Action layer: everything the agent can *do* goes through Executor.run().

    registry = ActionRegistry(); registry.register(ActionSpec("kv.set", handler, tier=1, params={...}, undo=...))
    ex = Executor(registry, PolicyEngine(), confirmer=ScriptedConfirmer(...))
    res = ex.run(rt, "kv.set", {"key": "brightness", "value": 20}, intent="user asked to dim the screen")

Per call, inside the ambient trace (every step is a span, so the whole thing replays):
  validate args -> [risk] -> context -> policy.check -> [confirm] -> action (write-ahead record; snapshot; handler;
  undo spec) -> note_executed.
Fail closed: if the write-ahead audit record cannot be persisted, AuditUnavailable propagates and nothing runs.
In replay everything except `permission` (and `llm`) is served from the recording and the handler never runs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from . import audit
from .confirm import Confirmer, DenyAll
from .policy import PolicyEngine
from .runtime import AuditUnavailable, Recorder, TraceRuntime, annotate
from .schema import safe_canonical
from .store import Store


class ArgError(ValueError):
    pass


@dataclass
class Param:
    type: Any = object          # a type, a tuple of types, or object (anything)
    required: bool = True


@dataclass
class ActionSpec:
    id: str
    handler: Callable[..., Any]
    tier: int = 3
    params: Optional[Dict[str, Union[Param, type, tuple]]] = None   # None = unchecked (not recommended)
    describe: Optional[Callable[[Dict[str, Any]], str]] = None
    snapshot: Optional[Callable[..., Optional[bytes]]] = None       # fn(**args) -> pre-state bytes | None
    undo: Optional[Callable[[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]] = None
    #   undo(args, result, snapshot_ref) -> {"action": id, "args": {...}} | None


class ActionRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, ActionSpec] = {}

    def register(self, spec: ActionSpec) -> ActionSpec:
        if spec.id in self._specs:
            raise ValueError(f"action {spec.id!r} already registered")
        self._specs[spec.id] = spec
        return spec

    def action(self, id: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(ActionSpec(id, fn, **kw))
            return fn

        return deco

    def get(self, id: str) -> Optional[ActionSpec]:
        return self._specs.get(id)

    def ids(self) -> List[str]:
        return sorted(self._specs)


def validate_args(spec: ActionSpec, args: Dict[str, Any]) -> None:
    if spec.params is None:
        return
    for name in args:
        if name not in spec.params:
            raise ArgError(f"{spec.id}: unexpected argument {name!r}")
    for name, p in spec.params.items():
        p = p if isinstance(p, Param) else Param(p)
        if name not in args:
            if p.required:
                raise ArgError(f"{spec.id}: missing required argument {name!r}")
            continue
        v = args[name]
        if p.type is not object:
            types = p.type if isinstance(p.type, tuple) else (p.type,)
            # bool is an int subclass: only accept it where bool was asked for
            if isinstance(v, bool) and bool not in types:
                raise ArgError(f"{spec.id}: {name!r} must be {_tn(types)}, got bool")
            if not isinstance(v, types):
                raise ArgError(f"{spec.id}: {name!r} must be {_tn(types)}, got {type(v).__name__}")


def _tn(types: tuple) -> str:
    return "|".join(t.__name__ for t in types)


@dataclass
class ExecResult:
    status: str   # ok | dry_run | denied | declined | error | invalid_args | unknown_action | not_executed
    action: str
    tier: Optional[int] = None
    decision: Optional[str] = None
    rule: Optional[str] = None
    result: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "dry_run")


class Executor:
    def __init__(self, registry: ActionRegistry, engine: PolicyEngine, confirmer: Optional[Confirmer] = None, *,
                 risk_fn: Optional[Callable[..., Dict[str, Any]]] = None, risk_name: str = "risk.score",
                 dry_run: bool = False):
        self.registry = registry
        self.engine = engine
        self.confirmer = confirmer or DenyAll()
        self.risk_fn = risk_fn
        self.risk_name = risk_name
        self.dry_run = dry_run

    def run(self, rt: TraceRuntime, action_id: str, args: Optional[Dict[str, Any]] = None, *, intent: str = "",
            dry_run: Optional[bool] = None) -> ExecResult:
        dry = self.dry_run if dry_run is None else dry_run
        args = rt.store.rehydrate(dict(args or {}))   # blob refs (e.g. undo snapshots) become bytes for the handler
        spec = self.registry.get(action_id)
        if spec is None:
            rt.call("custom", "executor.reject", {"action": action_id, "args": args, "reason": "unknown_action"},
                    lambda **kw: kw)
            return ExecResult("unknown_action", action_id, error="no such action registered")
        try:
            validate_args(spec, args)
        except ArgError as e:
            rt.call("custom", "executor.reject", {"action": action_id, "args": args, "reason": f"invalid_args: {e}"},
                    lambda **kw: kw)
            return ExecResult("invalid_args", action_id, error=str(e))

        # External code (risk hook, policy, confirmer) gets deep copies: whatever it does to its arguments, the args
        # that were checked and confirmed are exactly the args the handler receives.
        def iso(fn: Callable[..., Any]) -> Callable[..., Any]:
            return lambda **kw: fn(**copy.deepcopy(kw))

        risk: Dict[str, Any] = {"risk": 0.0}
        if self.risk_fn is not None:
            risk = rt.call("risk", self.risk_name, {"action": action_id, "args": args}, iso(self.risk_fn))
        ctx0 = rt.call("context", "policy.ctx", {"action": action_id, "args": args}, iso(self.engine.context))
        ctx = {**ctx0, "risk": risk}
        v = rt.call("permission", "policy.check", {"action": action_id, "args": args, "ctx": ctx}, iso(self.engine.check))
        tier, decision, rule = v["tier"], v["decision"], v["rule"]
        if decision == "deny":
            return ExecResult("denied", action_id, tier, decision, rule, error=f"denied by rule {rule}")

        describe = spec.describe(args) if spec.describe else f"{action_id} {safe_canonical(args)[:120]}"
        if decision == "confirm":
            # An approval is bound to (action, args) only. tier / intent / describe ride in meta, so a candidate
            # that words its intent differently (or re-tiers the action) still matches the recorded approval.
            def _ask(action: str, args: Dict[str, Any]) -> Dict[str, Any]:
                return self.confirmer.ask(action=action, args=copy.deepcopy(args), tier=tier, intent=intent,
                                          describe=describe)

            ans = rt.call("confirm", "user.confirm", {"action": action_id, "args": args}, _ask,
                          meta={"tier": tier, "intent": intent, "describe": describe})
            if not (isinstance(ans, dict) and ans.get("approved") is True):
                return ExecResult("declined", action_id, tier, decision, rule)
        elif decision != "allow":
            return ExecResult("denied", action_id, tier, decision, rule, error=f"unrecognised decision {decision!r}")

        def _run(**kw: Any) -> Dict[str, Any]:
            if dry:
                return {"status": "dry_run", "would": describe}
            snap_ref = None
            snap = spec.snapshot(**kw) if spec.snapshot else None
            if snap is not None:
                snap_ref = rt.put_blob(snap)
                annotate(snapshot=snap_ref)          # durable: persisted before the handler touches anything
            out = spec.handler(**kw)
            if not isinstance(out, dict):
                out = {"value": out}
            out.setdefault("status", "ok")
            u = spec.undo(kw, out, snap_ref) if spec.undo else None
            if u:
                annotate(undo=u)
            elif spec.tier >= 1 or tier >= 1:
                annotate(irreversible=True)
            return out

        meta = {"tier": max(tier, spec.tier), "intent": intent, "policy_id": v.get("policy_id"), "dry_run": dry}
        try:
            res = rt.call("action", action_id, args, _run, meta=meta)
        except AuditUnavailable:
            raise
        except Exception as e:  # noqa: BLE001  handler failure is an outcome, not a crash of the agent loop
            return ExecResult("error", action_id, tier, decision, rule, error=f"{type(e).__name__}: {e}")
        if isinstance(res, dict) and res.get("status") == "not_executed":
            return ExecResult("not_executed", action_id, tier, decision, rule, result=res)
        if not dry:
            self.engine.note_executed(action_id)
        return ExecResult("dry_run" if dry else "ok", action_id, tier, decision, rule, result=res)


class UndoRunner:
    """Executes the undo specs the audit log recorded, *through the same Executor* (policy, confirm, audit): an undo
    is just another action, recorded in its own `undo` trace. Dry-run by default (returns the plan, runs nothing)."""

    def __init__(self, store: Store, executor: Executor):
        self.store, self.executor = store, executor

    def run(self, trace_id: str, *, apply: bool = False, force: bool = False) -> Dict[str, Any]:
        plan = audit.undo_plan(self.store, trace_id)
        out: Dict[str, Any] = {"trace_id": trace_id, "applied": False, "steps": [], "irreversible": plan.irreversible}
        for s in plan.steps:
            out["steps"].append({"action": s["undo"].get("action"), "args": s["undo"].get("args"),
                                 "undoes": s["action"], "maybe_applied": s["maybe_applied"], "status": "planned"})
        if not apply:
            return out
        if "undone" in self.store.load_trace(trace_id).label_set() and not force:
            raise RuntimeError(f"{trace_id} is already labeled 'undone' (use force to run again)")
        rec = Recorder(self.store)
        with rec.trace("undo", {"of": trace_id}, tags=["undo"], meta={"undoes": trace_id}) as rt:
            for st, s in zip(out["steps"], plan.steps):
                r = self.executor.run(rt, st["action"], st["args"] or {},
                                      intent=f"undo {s['action']} from trace {trace_id}")
                st["status"] = r.status
                st["error"] = r.error
            rt.set_output({"steps": [{k: st.get(k) for k in ("action", "status")} for st in out["steps"]]})
        out["applied"] = True
        out["undo_trace_id"] = rt.trace_id
        if out["steps"] and all(st["status"] == "ok" for st in out["steps"]):
            self.store.label(trace_id, "undone", note=rt.trace_id)
        return out
