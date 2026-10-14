"""Core: tiers, policy, grants, registry, hash-chained audit, dispatcher. stdlib only.

Flow:
  Action -> validate args -> classify tier -> taint/source/device/rate/lockdown -> Decision
         -> confirm (voice/CLI/callback) -> executor -> audit (hash chain) -> undo stack
"""
from __future__ import annotations
import enum, hashlib, json, re, threading, time, uuid
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from .schema import Arg, validate
from .taint import TaintLedger


class Tier(enum.IntEnum):
    T0_OBSERVE = 0      # read-only: screen read, list, recall query
    T1_REVERSIBLE = 1   # undoable, local: volume, open app, scratch writes
    T2_STATEFUL = 2     # recoverable side effects: UI taps, installs, edits, plain shell
    T3_DANGEROUS = 3    # irreversible / privileged: rm, root shell, frida, payments, wipe


class Verdict(enum.Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    CONFIRM_STRONG = "confirm_strong"
    DENY = "deny"

    @property
    def strictness(self) -> int:
        return {"allow": 0, "confirm": 1, "confirm_strong": 2, "deny": 3}[self.value]


class Source(enum.IntEnum):
    """Where the *intent* originated. Lower = less trusted. Set by the caller, never by the model."""
    UNTRUSTED = 0   # web page, notification, OCR'd screen, third-party message
    DERIVED = 1     # background / model-initiated plan with no live owner utterance
    OWNER = 2       # live, authenticated owner voice/UI command


@dataclass(frozen=True)
class Action:
    name: str
    args: dict = field(default_factory=dict)
    source: Source = Source.OWNER
    device: str = "nitro"
    tainted_args: tuple = ()
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Decision:
    verdict: Verdict
    tier: Tier
    base_tier: Tier
    reasons: list
    grant: object = None            # Grant that lowered the verdict (consumed at commit)

    def as_dict(self):
        return {"verdict": self.verdict.value, "tier": int(self.tier),
                "base_tier": int(self.base_tier), "reasons": self.reasons,
                "grant": self.grant is not None}


# ---- registry --------------------------------------------------------------
@dataclass
class Spec:
    tier: Tier
    executor: Callable[[Action], object]
    summary: str = ""
    devices: tuple = ("nitro", "9r", "14pro")
    requires_root: bool = False
    classify: Optional[Callable[[Action], Tier]] = None
    schema: dict = field(default_factory=dict)
    output_untrusted: bool = False        # results feed the taint ledger
    undo: Optional[Callable[[Action, object], Optional[Callable[[], object]]]] = None
    describe: Optional[Callable[[Action], str]] = None   # custom readback


class Registry:
    def __init__(self):
        self._s: dict = {}

    def register(self, name, tier, executor, **kw):
        self._s[name] = Spec(tier, executor, **kw)

    def get(self, name) -> Optional[Spec]:
        return self._s.get(name)

    def names(self):
        return sorted(self._s)

    def items(self):
        return sorted(self._s.items())


# ---- grants (time-boxed elevation) ----------------------------------------
MAX_GRANT_TTL_S = 3600
MAX_GRANT_TIER = Tier.T2_STATEFUL


@dataclass
class Grant:
    max_tier: Tier
    expires: float
    action_prefix: str = ""
    uses_left: int = 10**9


class Grants:
    def __init__(self, clock=time.time):
        self._g: list = []
        self.clock = clock

    def add(self, max_tier, ttl_s, action_prefix="", uses=10**9) -> Grant:
        g = Grant(Tier(max_tier), self.clock() + ttl_s, action_prefix, uses)
        self._g.append(g)
        return g

    def find(self, action: Action, tier: Tier) -> Optional[Grant]:
        now = self.clock()
        self._g = [g for g in self._g if g.expires > now and g.uses_left > 0]
        for g in self._g:
            if tier <= g.max_tier and action.name.startswith(g.action_prefix):
                return g
        return None

    def revoke_all(self):
        self._g.clear()


# ---- audit -----------------------------------------------------------------
_SECRET_KEY = re.compile(r"pass|token|secret|key|pin|otp|cookie|auth", re.I)
MAX_LOG_STR = 1024


def redact(args: dict) -> dict:
    out = {}
    for k, v in (args or {}).items():
        if _SECRET_KEY.search(str(k)):
            out[k] = "***"
        elif isinstance(v, str) and len(v) > MAX_LOG_STR:
            out[k] = v[:MAX_LOG_STR] + f"…(+{len(v) - MAX_LOG_STR})"
        else:
            out[k] = v
    return out


def _canon(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, default=str, separators=(",", ":"))


class Audit:
    """Append-only, hash-chained records. Chain resumes from the file across restarts."""
    GENESIS = "0" * 64

    def __init__(self, path=None, sinks=None, clock=time.time):
        self.path, self.sinks, self.clock = path, list(sinks or []), clock
        self.prev, self.seq, self.records = self.GENESIS, 0, []
        self._lock = threading.Lock()
        if path:
            try:
                last = None
                with open(path) as f:
                    for line in f:
                        if line.strip(): last = json.loads(line)
                if last:
                    self.prev, self.seq = last["hash"], last["seq"] + 1
            except FileNotFoundError:
                pass

    def write(self, event: str, **data):
        with self._lock:
            rec = {"seq": self.seq, "ts": self.clock(), "event": event, **data, "prev": self.prev}
            rec = json.loads(_canon(rec))
            rec["hash"] = hashlib.sha256(_canon(rec).encode()).hexdigest()
            self.prev, self.seq = rec["hash"], self.seq + 1
            self.records.append(rec)
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            for s in self.sinks:
                try: s.emit(rec)
                except Exception: pass     # sinks must never break enforcement
            return rec

    @staticmethod
    def verify(records) -> bool:
        prev = Audit.GENESIS
        for r in records:
            r = dict(r)
            h = r.pop("hash", None)
            if r.get("prev") != prev: return False
            if hashlib.sha256(_canon(r).encode()).hexdigest() != h: return False
            prev = h
        return True

    @staticmethod
    def load(path) -> list:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]


# ---- policy ----------------------------------------------------------------
BASE_VERDICT = {Tier.T0_OBSERVE: Verdict.ALLOW, Tier.T1_REVERSIBLE: Verdict.ALLOW,
                Tier.T2_STATEFUL: Verdict.CONFIRM, Tier.T3_DANGEROUS: Verdict.CONFIRM_STRONG}
ROOT_DEVICES = {"9r"}
DEFAULT_RATE = {Tier.T1_REVERSIBLE: (60, 60.0), Tier.T2_STATEFUL: (10, 60.0), Tier.T3_DANGEROUS: (3, 300.0)}


class Policy:
    def __init__(self, registry: Registry, grants: Grants, clock=time.time, rate=None,
                 root_devices=None):
        self.reg, self.grants, self.clock = registry, grants, clock
        self.rate = dict(rate or DEFAULT_RATE)
        self.root_devices = set(root_devices or ROOT_DEVICES)
        self._hist: dict = {t: [] for t in self.rate}
        self.lockdown_until = 0.0
        self.lockdown_reason = ""

    # lockdown
    def lockdown(self, seconds: float, reason: str):
        self.lockdown_until, self.lockdown_reason = self.clock() + seconds, reason

    def unlock(self):
        self.lockdown_until, self.lockdown_reason = 0.0, ""

    @property
    def locked(self) -> bool:
        return self.clock() < self.lockdown_until

    def _rate_ok(self, tier) -> bool:
        if tier not in self.rate: return True
        n, win = self.rate[tier]; now = self.clock()
        self._hist[tier] = [t for t in self._hist[tier] if now - t < win]
        return len(self._hist[tier]) < n

    def decide(self, a: Action) -> Decision:
        """Pure: no state is consumed. commit() consumes grants/rate budget."""
        spec = self.reg.get(a.name)
        if spec is None:
            return Decision(Verdict.DENY, Tier.T3_DANGEROUS, Tier.T3_DANGEROUS, ["unknown action"])
        try:
            base = spec.classify(a) if spec.classify else spec.tier
        except Exception as e:
            return Decision(Verdict.DENY, Tier.T3_DANGEROUS, Tier.T3_DANGEROUS, [f"classifier error: {e!r}"])
        tier, why = base, []

        errs = validate(spec.schema, a.args) if spec.schema else []
        if errs:
            return Decision(Verdict.DENY, tier, base, ["invalid args: " + "; ".join(errs)])
        if a.device not in spec.devices:
            return Decision(Verdict.DENY, tier, base, [f"{a.name} unsupported on {a.device}"])
        if spec.requires_root and a.device not in self.root_devices:
            return Decision(Verdict.DENY, tier, base, [f"root required; {a.device} not rooted"])

        if self.locked and tier > Tier.T0_OBSERVE:
            return Decision(Verdict.DENY, tier, base, [f"lockdown: {self.lockdown_reason}"])

        if a.tainted_args:
            tier = Tier(min(tier + 1, Tier.T3_DANGEROUS))
            why.append(f"tainted args {list(a.tainted_args)}: +1 tier")

        if a.source == Source.UNTRUSTED and tier > Tier.T0_OBSERVE:
            return Decision(Verdict.DENY, tier, base, why + ["untrusted source limited to T0"])
        if a.source == Source.DERIVED and tier >= Tier.T3_DANGEROUS:
            return Decision(Verdict.DENY, tier, base, why + ["T3 requires live owner intent"])

        if not self._rate_ok(tier):
            return Decision(Verdict.DENY, tier, base, why + ["rate limit"])

        verdict, grant = BASE_VERDICT[tier], None
        if verdict != Verdict.ALLOW and tier < Tier.T3_DANGEROUS and not a.tainted_args \
                and a.source == Source.OWNER:
            grant = self.grants.find(a, tier)
            if grant:
                verdict = Verdict.ALLOW; why.append("covered by grant")
        if not why:
            why.append("base policy")
        return Decision(verdict, tier, base, why, grant)

    def commit(self, d: Decision):
        if d.grant is not None:
            d.grant.uses_left -= 1
        if d.tier in self._hist:
            self._hist[d.tier].append(self.clock())


# ---- dispatcher ------------------------------------------------------------
Confirmer = Callable[[Action, Decision], bool]
Alert = Callable[[str], None]
STORM_N, STORM_WINDOW_S, STORM_LOCK_S = 5, 60.0, 300.0


def _summ(x, n=512):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"


class Dispatcher:
    def __init__(self, registry, grants=None, audit=None, confirm: Confirmer = lambda a, d: False,
                 dry_run=False, clock=time.time, ledger: Optional[TaintLedger] = None,
                 on_alert: Optional[Alert] = None, rate=None, root_devices=None):
        self.reg = registry
        self.grants = grants or Grants(clock)
        self.policy = Policy(registry, self.grants, clock, rate, root_devices)
        self.audit = audit or Audit(clock=clock)
        self.confirm, self.dry_run, self.clock = confirm, dry_run, clock
        self.ledger, self.on_alert = ledger, on_alert or (lambda m: None)
        self.undo_stack: deque = deque(maxlen=20)
        self._denials: deque = deque(maxlen=64)

    # -- helpers
    def explain(self, a: Action) -> Decision:
        return self.policy.decide(self._with_ledger(a))

    def _with_ledger(self, a: Action) -> Action:
        if self.ledger is None: return a
        extra = self.ledger.tainted_keys(a.args)
        merged = tuple(dict.fromkeys(tuple(a.tainted_args) + tuple(extra)))
        return replace(a, tainted_args=merged) if merged != tuple(a.tainted_args) else a

    def _ask(self, a, d) -> bool:
        try:
            return bool(self.confirm(a, d))
        except Exception as e:
            self.audit.write("confirmation_error", action_id=a.id, error=repr(e))
            return False

    def _note_denial(self, a: Action, d: Decision):
        suspicious = a.source != Source.OWNER or bool(a.tainted_args)
        if not suspicious: return
        now = self.clock()
        self._denials.append(now)
        recent = [t for t in self._denials if now - t <= STORM_WINDOW_S]
        if len(recent) >= STORM_N and not self.policy.locked:
            self.policy.lockdown(STORM_LOCK_S, "denial storm (possible injection)")
            self.audit.write("lockdown", reason=self.policy.lockdown_reason, seconds=STORM_LOCK_S)
            self.on_alert(f"Lockdown: {self.policy.lockdown_reason}. Actions above observe-only are blocked for "
                          f"{int(STORM_LOCK_S // 60)} minutes.")

    # -- main
    def run(self, a: Action):
        a = self._with_ledger(a)
        d = self.policy.decide(a)
        self.audit.write("decision", action_id=a.id, action=a.name, args=redact(a.args),
                         device=a.device, source=a.source.name, tainted_args=list(a.tainted_args),
                         **d.as_dict())
        if d.verdict == Verdict.DENY:
            self._note_denial(a, d)
            return {"ok": False, "denied": True, "reasons": d.reasons, "action_id": a.id}
        if d.verdict in (Verdict.CONFIRM, Verdict.CONFIRM_STRONG):
            ok = self._ask(a, d)
            self.audit.write("confirmation", action_id=a.id, action=a.name, granted=ok,
                             strong=d.verdict == Verdict.CONFIRM_STRONG)
            if not ok:
                return {"ok": False, "denied": True, "reasons": ["not confirmed"], "action_id": a.id}
        spec = self.reg.get(a.name)
        self.policy.commit(d)
        if self.dry_run:
            self.audit.write("dry_run", action_id=a.id, action=a.name)
            return {"ok": True, "dry_run": True, "decision": d.as_dict(), "action_id": a.id}
        try:
            out = spec.executor(a)
        except Exception as e:  # noqa
            self.audit.write("executed", action_id=a.id, action=a.name, ok=False, error=repr(e))
            return {"ok": False, "error": repr(e), "decision": d.as_dict(), "action_id": a.id}
        self.audit.write("executed", action_id=a.id, action=a.name, ok=True, result=_summ(out, 256))
        if spec.output_untrusted and self.ledger is not None and out is not None:
            self.ledger.add(out if isinstance(out, str) else json.dumps(out, default=str), a.name)
        if spec.undo:
            try:
                fn = spec.undo(a, out)
                if fn:
                    self.undo_stack.append((a.id, a.name, fn))
            except Exception as e:
                self.audit.write("undo_prepare_failed", action_id=a.id, error=repr(e))
        return {"ok": True, "result": out, "decision": d.as_dict(), "action_id": a.id}

    def undo_last(self):
        if not self.undo_stack:
            return {"ok": False, "error": "nothing to undo"}
        aid, name, fn = self.undo_stack.pop()
        try:
            r = fn()
            self.audit.write("undo", action_id=aid, action=name, ok=True)
            return {"ok": True, "undid": name, "result": r}
        except Exception as e:
            self.audit.write("undo", action_id=aid, action=name, ok=False, error=repr(e))
            return {"ok": False, "error": repr(e)}

    # -- privileged meta-operations (always owner + strong confirm)
    def _meta(self, name, args, source=Source.OWNER):
        a = Action(name, args, source)
        d = Decision(Verdict.CONFIRM_STRONG, Tier.T3_DANGEROUS, Tier.T3_DANGEROUS, ["meta operation"])
        if source != Source.OWNER:
            self.audit.write("meta_denied", action=name, source=source.name)
            return a, d, False
        ok = self._ask(a, d)
        self.audit.write("meta", action=name, args=args, granted=ok)
        return a, d, ok

    def request_grant(self, max_tier: Tier, ttl_s: float, prefix: str = "", uses: Optional[int] = None,
                      source: Source = Source.OWNER):
        if Tier(max_tier) > MAX_GRANT_TIER or ttl_s > MAX_GRANT_TTL_S or ttl_s <= 0:
            return {"ok": False, "reasons": ["grant exceeds limits (max T2, <=1h)"]}
        _, _, ok = self._meta("grant.request", {"max_tier": int(max_tier), "ttl_s": ttl_s,
                                                 "prefix": prefix, "uses": uses}, source)
        if not ok:
            return {"ok": False, "reasons": ["not confirmed"]}
        self.grants.add(max_tier, ttl_s, prefix, uses if uses is not None else 10**9)
        return {"ok": True}

    def unlock(self, source: Source = Source.OWNER):
        _, _, ok = self._meta("lockdown.unlock", {"reason": self.policy.lockdown_reason}, source)
        if ok: self.policy.unlock()
        return {"ok": ok}

    def lockdown(self, seconds=3600.0, reason="manual kill switch"):
        """Kill switch: no confirmation needed to *tighten*."""
        self.policy.lockdown(seconds, reason)
        self.grants.revoke_all()
        self.audit.write("lockdown", reason=reason, seconds=seconds)
