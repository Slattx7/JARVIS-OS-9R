"""Declarative policy engine.

check(action, args, ctx) is a PURE function of its inputs (the executor records ctx in its own `context` span),
so a recorded permission check can be re-evaluated against a different policy offline (policy-diff) or in replay.

Monotone by construction: the base tier/decision come from `tiers` + `tier_decisions`; `rules` can raise the tier
and tighten the decision (allow < confirm < deny) but never loosen it. Loosening is an edit to the base maps,
which is exactly the diff `policy-diff` reviews. Unknown actions get the `default` tier (3 -> confirm): fail closed.

Rule `when` leaves read `{"arg": name}` or `{"ctx": "a/b/c"}` (ctx paths use "/" because action ids contain dots)
with exactly one op: eq ne gt gte lt lte in not_in matches not_matches under not_under contains exists.
Missing value: positive ops are False, negative ops (ne not_in not_matches not_under) are True, so a rule such as
"path not under the allowed roots" fires when the argument is absent or malformed. Combinators: all / any / not.
"""
from __future__ import annotations

import collections
import copy
import fnmatch
import functools
import hashlib
import json
import posixpath
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

from . import killswitch
from .schema import STRICTNESS, canonical, normalize, NormCtx, safe_canonical

DECISIONS = ("allow", "confirm", "deny")
_LEAF_OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "matches", "not_matches", "under", "not_under",
             "contains", "exists")
_NEG_OPS = frozenset({"ne", "not_in", "not_matches", "not_under"})


class PolicyError(ValueError):
    pass


DEFAULT_POLICY: Dict[str, Any] = {
    "version": 1,
    "default": {"tier": 3},
    "tiers": {
        "settings.get": 0, "screen.read": 0, "fs.read": 0, "kv.get": 0, "shell.read": 0,
        "timer.set": 1, "settings.set": 1, "kv.set": 1, "kv.unset": 1,
        "message.send": 2, "app.install": 2, "fs.write": 2, "fs.restore": 2, "fs.remove": 2, "fs.move": 2,
        "fs.delete": 3, "shell.exec": 3, "partition.write": 3,
    },
    "tier_decisions": {"0": "allow", "1": "allow", "2": "confirm", "3": "confirm"},
    "rules": [
        {"id": "paused", "min_tier": 1, "when": {"ctx": "paused", "eq": True}, "decision": "deny"},
        {"id": "risk-deny", "when": {"ctx": "risk/risk", "gte": 0.85}, "decision": "deny"},
        {"id": "risk-confirm", "min_tier": 1, "when": {"ctx": "risk/risk", "gte": 0.5}, "decision": "confirm"},
        {"id": "no-partition-writes", "action": "partition.*", "decision": "deny"},
        {"id": "shell-destructive", "action": "shell.*",
         "when": {"arg": "cmd", "matches": r"(^|[\s;&|'\"`(])(rm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]|dd\s|mkfs|fdisk|parted|wipefs|shred)"},
         "decision": "deny"},
        {"id": "message-rate", "action": "message.send", "when": {"ctx": "recent/message.send", "gte": 5},
         "decision": "deny"},
        {"id": "night-quiet", "min_tier": 2, "when": {"all": [{"ctx": "attended", "eq": False}, {"ctx": "hour", "lt": 6}]},
         "decision": "deny"},
    ],
}


@functools.lru_cache(maxsize=256)
def _rx(pattern: str, flags: str) -> "re.Pattern[str]":
    f = 0
    if "i" in flags:
        f |= re.IGNORECASE
    if "s" in flags:
        f |= re.DOTALL
    return re.compile(pattern, f)


def _ctx_get(ctx: Dict[str, Any], path: str) -> Tuple[bool, Any]:
    cur: Any = ctx
    for part in path.split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def _norm_path(p: Any) -> Optional[str]:
    if not isinstance(p, str) or not p.startswith("/"):
        return None
    return posixpath.normpath(p)


def _under(val: Any, roots: Iterable[str]) -> bool:
    v = _norm_path(val)
    if v is None:
        return False
    for r in roots:
        rn = _norm_path(r)
        if rn is not None and (v == rn or v.startswith(rn.rstrip("/") + "/")):
            return True
    return False


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _eval(cond: Dict[str, Any], args: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    if "all" in cond:
        return all(_eval(c, args, ctx) for c in cond["all"])
    if "any" in cond:
        return any(_eval(c, args, ctx) for c in cond["any"])
    if "not" in cond:
        return not _eval(cond["not"], args, ctx)
    if "arg" in cond:
        found, val = (cond["arg"] in args), args.get(cond["arg"])
    else:
        found, val = _ctx_get(ctx, cond["ctx"])
    op = next(k for k in _LEAF_OPS if k in cond)
    want = cond[op]
    if op == "exists":
        return found == bool(want)
    if not found:
        return op in _NEG_OPS
    if op == "eq":
        return val == want
    if op == "ne":
        return val != want
    if op in ("gt", "gte", "lt", "lte"):
        if not (_num(val) and _num(want)):
            return False
        return {"gt": val > want, "gte": val >= want, "lt": val < want, "lte": val <= want}[op]
    if op == "in":
        return val in want
    if op == "not_in":
        return val not in want
    if op in ("matches", "not_matches"):
        hit = isinstance(val, str) and _rx(want, cond.get("flags", "")).search(val) is not None
        return hit if op == "matches" else not hit
    if op in ("under", "not_under"):
        u = _under(val, want)
        return u if op == "under" else not u
    if op == "contains":
        return isinstance(val, (str, list, tuple)) and want in val
    raise PolicyError(f"unhandled op {op}")  # pragma: no cover


def _validate_cond(cond: Any, where: str) -> None:
    if not isinstance(cond, dict):
        raise PolicyError(f"{where}: condition must be an object")
    for comb in ("all", "any"):
        if comb in cond:
            if not isinstance(cond[comb], list) or not cond[comb]:
                raise PolicyError(f"{where}: '{comb}' needs a non-empty list")
            for i, c in enumerate(cond[comb]):
                _validate_cond(c, f"{where}.{comb}[{i}]")
            return
    if "not" in cond:
        _validate_cond(cond["not"], f"{where}.not")
        return
    if ("arg" in cond) == ("ctx" in cond):
        raise PolicyError(f"{where}: leaf needs exactly one of 'arg' / 'ctx'")
    ops = [k for k in _LEAF_OPS if k in cond]
    if len(ops) != 1:
        raise PolicyError(f"{where}: leaf needs exactly one op of {', '.join(_LEAF_OPS)} (got {ops or 'none'})")
    op = ops[0]
    if op in ("matches", "not_matches"):
        try:
            _rx(cond[op], cond.get("flags", ""))
        except (re.error, TypeError) as e:
            raise PolicyError(f"{where}: bad regex: {e}") from e
    if op in ("in", "not_in", "under", "not_under") and not isinstance(cond[op], list):
        raise PolicyError(f"{where}: '{op}' needs a list")


def validate_policy(p: Dict[str, Any]) -> None:
    if not isinstance(p, dict):
        raise PolicyError("policy must be an object")
    unknown = set(p) - {"version", "default", "tiers", "tier_decisions", "rules", "description"}
    if unknown:
        raise PolicyError(f"unknown top-level keys: {sorted(unknown)}")
    d = p.get("default", {"tier": 3})
    if not isinstance(d.get("tier"), int) or not 0 <= d["tier"] <= 3:
        raise PolicyError("default.tier must be an int 0..3")
    for k, v in p.get("tiers", {}).items():
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 3:
            raise PolicyError(f"tiers[{k!r}] must be an int 0..3")
    td = p.get("tier_decisions", {})
    for k, v in td.items():
        if k not in ("0", "1", "2", "3") or v not in DECISIONS:
            raise PolicyError(f"tier_decisions[{k!r}]={v!r}: keys '0'..'3', values {DECISIONS}")
    seen = set()
    for i, r in enumerate(p.get("rules", [])):
        rid = r.get("id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise PolicyError(f"rules[{i}]: needs a unique string id")
        seen.add(rid)
        unknown = set(r) - {"id", "action", "when", "min_tier", "tier", "decision", "description"}
        if unknown:
            raise PolicyError(f"rule {rid}: unknown keys {sorted(unknown)}")
        if "decision" not in r and "tier" not in r:
            raise PolicyError(f"rule {rid}: needs 'decision' and/or 'tier'")
        if "decision" in r and r["decision"] not in DECISIONS:
            raise PolicyError(f"rule {rid}: decision must be one of {DECISIONS}")
        if "tier" in r and (not isinstance(r["tier"], int) or not 0 <= r["tier"] <= 3):
            raise PolicyError(f"rule {rid}: tier must be an int 0..3")
        if "when" in r:
            _validate_cond(r["when"], f"rule {rid}.when")


class PolicyEngine:
    def __init__(self, policy: Optional[Dict[str, Any]] = None, *, window_s: float = 600.0,
                 pause_file: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.policy = copy.deepcopy(policy if policy is not None else DEFAULT_POLICY)
        validate_policy(self.policy)
        self.policy_id = hashlib.sha256(canonical(self.policy).encode()).hexdigest()[:12]
        self.window_s = window_s
        self.pause_file = pause_file
        self._clock = clock
        td = {"0": "allow", "1": "allow", "2": "confirm", "3": "confirm"}
        td.update(self.policy.get("tier_decisions", {}))
        self._td = {int(k): v for k, v in td.items()}
        self._tiers: Dict[str, int] = dict(self.policy.get("tiers", {}))
        self._default_tier: int = self.policy.get("default", {}).get("tier", 3)
        self._rules: List[Dict[str, Any]] = list(self.policy.get("rules", []))
        self._hist: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: str, **kw: Any) -> "PolicyEngine":
        with open(path) as f:
            return cls(json.load(f), **kw)

    # -- state -> ctx (recorded by the executor, so check() stays pure) ------------------------------------ #
    def note_executed(self, action: str) -> None:
        with self._lock:
            self._hist[action].append(self._clock())

    def context(self, action: Optional[str] = None, args: Optional[Dict[str, Any]] = None,
                extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = self._clock()
        recent: Dict[str, int] = {}
        with self._lock:
            for a, dq in self._hist.items():
                while dq and now - dq[0] > self.window_s:
                    dq.popleft()
                if dq:
                    recent[a] = len(dq)
        ps = killswitch.state(self.pause_file)
        lt = time.localtime(now)
        ctx: Dict[str, Any] = {"recent": recent, "paused": ps is not None, "hour": lt.tm_hour, "weekday": lt.tm_wday,
                               "attended": True}
        if ps is not None:
            ctx["pause_reason"] = str(ps.get("reason", ""))
        if extra:
            ctx.update(extra)
        return ctx

    # -- decision -------------------------------------------------------------------------------------------- #
    def tier_of(self, action: str) -> Tuple[int, bool]:
        if action in self._tiers:
            return self._tiers[action], True
        for pat, t in self._tiers.items():
            if any(c in pat for c in "*?[") and fnmatch.fnmatchcase(action, pat):
                return t, True
        return self._default_tier, False

    def check(self, action: str, args: Optional[Dict[str, Any]] = None, ctx: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
        args, ctx = args or {}, ctx or {}
        base_tier, known = self.tier_of(action)
        tier, decision = base_tier, self._td[base_tier]
        deciding = f"tier{base_tier}" if known else "default"
        matched: List[str] = []
        for r in self._rules:
            if "action" in r and not fnmatch.fnmatchcase(action, r["action"]):
                continue
            if base_tier < r.get("min_tier", 0):
                continue
            if "when" in r and not _eval(r["when"], args, ctx):
                continue
            matched.append(r["id"])
            new_tier = max(tier, r["tier"]) if "tier" in r else tier
            new_dec = decision
            if "decision" in r and STRICTNESS[r["decision"]] > STRICTNESS[new_dec]:
                new_dec = r["decision"]
            if STRICTNESS[self._td[new_tier]] > STRICTNESS[new_dec]:
                new_dec = self._td[new_tier]
            if new_tier != tier or new_dec != decision:
                deciding = r["id"]
            tier, decision = new_tier, new_dec
        return {"tier": tier, "decision": decision, "rule": deciding, "matched": matched,
                "policy_id": self.policy_id}

    def explain(self, action: str, args: Optional[Dict[str, Any]] = None, ctx: Optional[Dict[str, Any]] = None
                ) -> List[str]:
        args, ctx = args or {}, ctx or {}
        base_tier, known = self.tier_of(action)
        lines = [f"base: tier {base_tier} ({'mapped' if known else 'default: unknown action'}) -> "
                 f"{self._td[base_tier]}"]
        for r in self._rules:
            why = None
            if "action" in r and not fnmatch.fnmatchcase(action, r["action"]):
                why = f"action !~ {r['action']}"
            elif base_tier < r.get("min_tier", 0):
                why = f"base tier {base_tier} < min_tier {r['min_tier']}"
            elif "when" in r and not _eval(r["when"], args, ctx):
                why = "condition false"
            eff = ", ".join(f"{k}={r[k]}" for k in ("tier", "decision") if k in r)
            lines.append(f"rule {r['id']}: " + (f"skip ({why})" if why else f"MATCH -> {eff}"))
        res = self.check(action, args, ctx)
        lines.append(f"result: tier {res['tier']} {res['decision']} (deciding: {res['rule']})")
        return lines

    def lint(self) -> List[str]:
        w: List[str] = []
        for t in (2, 3):
            if self._td[t] == "allow":
                w.append(f"tier_decisions maps tier {t} to 'allow': consequential/destructive actions run unconfirmed")
        if self._default_tier < 3:
            w.append(f"default tier is {self._default_tier}: unknown actions are treated as lower-risk than tier 3")
        if not any(r["id"] == "paused" or (isinstance(r.get("when"), dict) and r["when"].get("ctx") == "paused")
                   for r in self._rules):
            w.append("no rule reacts to ctx.paused: the pause switch will not stop anything")
        names = {"shell.exec", "partition.write", "fs.delete"} & set(self._tiers)
        for n in sorted(names):
            if self._tiers[n] < 3:
                w.append(f"{n} is mapped to tier {self._tiers[n]} (expected 3)")
        ruled_actions = [r for r in self._rules if r.get("decision") == "allow"]
        for r in ruled_actions:
            w.append(f"rule {r['id']} has decision 'allow': rules only tighten, so it has no effect")
        return w


# ---------------------------------------------------------------------------------------------------------------- #
# offline tooling: test cases, diff between two policies over recorded checks
# ---------------------------------------------------------------------------------------------------------------- #


def run_cases(engine: PolicyEngine, cases: List[Dict[str, Any]]) -> List[str]:
    """Each case: {"action","args","ctx","expect":{"decision","tier","rule"}}. Returns failure messages."""
    fails: List[str] = []
    for i, c in enumerate(cases):
        got = engine.check(c["action"], c.get("args"), c.get("ctx"))
        for k, want in (c.get("expect") or {}).items():
            if got.get(k) != want:
                fails.append(f"case {i} ({c['action']} {safe_canonical(c.get('args', {}))[:60]}): {k} expected "
                             f"{want!r}, got {got.get(k)!r} [rule {got['rule']}]")
    return fails


@dataclass
class PolicyDiff:
    n: int = 0
    loosened: List[Dict[str, Any]] = field(default_factory=list)
    tightened: List[Dict[str, Any]] = field(default_factory=list)
    tier_changed: List[Dict[str, Any]] = field(default_factory=list)


def diff_policies(old: PolicyEngine, new: PolicyEngine, cases: Iterable[Dict[str, Any]], dedupe: bool = True
                  ) -> PolicyDiff:
    d = PolicyDiff()
    seen: Dict[str, Dict[str, Any]] = {}
    for c in cases:
        key = safe_canonical({"a": c["action"], "g": c.get("args", {}), "c": c.get("ctx", {})}) if dedupe else str(d.n)
        if key in seen:
            seen[key]["count"] += 1
            continue
        o, n = old.check(c["action"], c.get("args"), c.get("ctx")), new.check(c["action"], c.get("args"), c.get("ctx"))
        d.n += 1
        row = {"action": c["action"], "args": normalize(c.get("args", {}), NormCtx()), "old": f"T{o['tier']} {o['decision']} ({o['rule']})",
               "new": f"T{n['tier']} {n['decision']} ({n['rule']})", "count": 1}
        seen[key] = row
        so, sn = STRICTNESS[o["decision"]], STRICTNESS[n["decision"]]
        if sn < so:
            d.loosened.append(row)
        elif sn > so:
            d.tightened.append(row)
        elif o["tier"] != n["tier"]:
            d.tier_changed.append(row)
    return d


def cases_from_store(store: Any, since_ns: Optional[int] = None, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Every recorded policy check (action, args, ctx) as a case. Args are as recorded, i.e. already redacted."""
    for h in store.list_traces(since_ns=since_ns, limit=limit, newest_first=False):
        for s in store.load_trace(h.trace_id).spans_of("permission"):
            i = s.inputs if isinstance(s.inputs, dict) else {}
            if "action" in i:
                yield {"action": i["action"], "args": store.rehydrate(i.get("args") or {}), "ctx": i.get("ctx") or {}}
