"""Permission broker. The LLM only *proposes*; this process holds capabilities and enforces tiers.

  T0 read-only            -> auto
  T1 reversible           -> auto (logged)
  T2 state-changing       -> spoken yes/no, adjudicated here, never by the LLM
  T3 root / arbitrary     -> out-of-band approval: Ed25519 signature over (id, nonce, action digest) made on a
                             second device. The host holds only the public key, so compromising the host (or the LLM,
                             or spoofing voice) cannot mint an approval. (`hmac` scheme: shared secret, weaker.)

Taint: a proposal whose provenance includes untrusted content (tool output, screen, notifications, web...)
is escalated one tier (T1->T2, T2->T3). Untrusted text can inform a proposal, never lower its bar.
Every transition is written to a hash-chained audit log.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .actions import ActionSpec, ArgError, ExecContext, ExecResult, Registry, Tier
from .approval import HmacVerifier, Verifier
from .audit import AuditLog
from .intents import Intent


class Verdict(str, Enum):
    ALLOW = "allow"
    NEED_CONFIRM = "need_confirm"
    NEED_OOB = "need_oob"
    DENY = "deny"


@dataclass
class Decision:
    verdict: Verdict
    base_tier: Tier | None
    tier: Tier | None
    reason: str = ""
    args: dict = field(default_factory=dict)
    spec: ActionSpec | None = None


@dataclass
class Outcome:
    status: str  # executed | dry_run | pending_confirm | pending_oob | denied | failed | cancelled | expired
    say: str = ""
    result: ExecResult | None = None
    pending_id: str | None = None
    decision: Decision | None = None


@dataclass
class Pending:
    id: str
    intent: Intent
    spec: ActionSpec
    args: dict
    tier: Tier
    kind: str  # confirm | oob
    nonce: str
    created: float
    expires: float
    readback: str
    bad_tokens: int = 0


RATE = {Tier.T0: (60, 60.0), Tier.T1: (10, 60.0), Tier.T2: (4, 60.0), Tier.T3: (2, 300.0)}  # (count, window_s)
YES = {"yes", "yeah", "yep", "yup", "confirm", "confirmed", "affirmative", "sure", "proceed", "ok", "okay"}
YES_PHRASES = ("do it", "go ahead", "go for it", "that's right", "thats right")
NO = {"no", "nope", "cancel", "stop", "abort", "negative", "don't", "dont", "never", "nah"}


def parse_yes_no(text: str) -> bool | None:
    words = set(re.findall(r"[a-z']+", text.lower()))
    low = text.lower()
    yes = bool(words & YES) or any(p in low for p in YES_PHRASES)
    no = bool(words & NO)
    if yes == no:
        return None  # neither, or contradictory: treat as not an answer
    return yes


def approval_token(secret: bytes, digest: str, nonce: str, pending_id: str) -> str:
    """Legacy HMAC token (kept for the `hmac` scheme and existing callers)."""
    return hmac.new(secret, f"{digest}:{nonce}:{pending_id}".encode(), hashlib.sha256).hexdigest()


def load_or_create_secret(path: str | os.PathLike) -> bytes:
    p = Path(os.path.expanduser(str(path)))
    if p.exists():
        return bytes.fromhex(p.read_text().strip())
    p.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key.hex() + "\n")
    return key


class Broker:
    def __init__(self, cfg, registry: Registry, audit: AuditLog, verifier: Verifier | bytes, ctx: ExecContext,
                 clock=time.time):
        if isinstance(verifier, (bytes, bytearray)):  # legacy: raw HMAC secret
            verifier = HmacVerifier(bytes(verifier))
        self.cfg, self.registry, self.audit, self.verifier, self.ctx, self.clock = cfg, registry, audit, verifier, ctx, clock
        self._pending: dict[str, Pending] = {}
        self._lock = threading.RLock()
        self._rate: dict[Tier, deque] = {t: deque() for t in Tier}

    # ---- policy ----
    def evaluate(self, intent: Intent) -> Decision:
        try:
            spec, args = self.registry.validate(intent.action, intent.args)
        except ArgError as e:
            return Decision(Verdict.DENY, None, None, str(e))
        base = spec.tier
        tier = Tier(min(Tier.T3, base + 1)) if (intent.tainted and base >= Tier.T1) else base
        verdict = {Tier.T0: Verdict.ALLOW, Tier.T1: Verdict.ALLOW,
                   Tier.T2: Verdict.NEED_CONFIRM, Tier.T3: Verdict.NEED_OOB}[tier]
        reason = "escalated: untrusted provenance" if tier != base else ""
        return Decision(verdict, base, tier, reason, args, spec)

    def _rate_ok(self, tier: Tier) -> bool:
        n, window = RATE[tier]
        now = self.clock()
        q = self._rate[tier]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= n:
            return False
        q.append(now)
        return True

    # ---- propose ----
    def propose(self, intent: Intent) -> Outcome:
        with self._lock:
            self._expire()
            d = self.evaluate(intent)
            base_log = dict(action=intent.action, args=intent.args, digest=intent.digest(),
                            provenance=list(intent.provenance))
            if d.verdict is Verdict.DENY:
                self.audit.append("denied", reason=d.reason, **base_log)
                return Outcome("denied", f"I can't do that: {d.reason}.", decision=d)
            if not self._rate_ok(d.tier):
                self.audit.append("rate_limited", tier=int(d.tier), **base_log)
                return Outcome("denied", "That's happening too often; I've rate limited it.", decision=d)
            self.audit.append("proposed", tier=int(d.tier), base_tier=int(d.base_tier), verdict=d.verdict.value,
                              note=d.reason, **base_log)
            if d.verdict is Verdict.ALLOW:
                return self._execute(intent, d)
            return self._make_pending(intent, d)

    def _make_pending(self, intent: Intent, d: Decision) -> Outcome:
        kind = "confirm" if d.verdict is Verdict.NEED_CONFIRM else "oob"
        if kind == "oob" and not self.verifier.configured:
            self.audit.append("denied", reason="no approver key configured", action=intent.action, digest=intent.digest())
            return Outcome("denied", "Out of band approval isn't set up, so I can't do that.", decision=d)
        if kind == "confirm":  # one live confirmation at a time
            for p in [p for p in self._pending.values() if p.kind == "confirm"]:
                self._drop(p, "superseded")
        pid = secrets.token_hex(3)
        ttl = self.cfg.broker.confirm_ttl_s if kind == "confirm" else self.cfg.broker.oob_ttl_s
        now = self.clock()
        what = d.spec.describe(d.args) or d.spec.name
        p = Pending(pid, intent, d.spec, d.args, d.tier, kind, secrets.token_hex(16), now, now + ttl, what)
        self._pending[pid] = p
        self.audit.append("pending", id=pid, kind=kind, tier=int(d.tier), readback=what, digest=intent.digest())
        if kind == "confirm":
            return Outcome("pending_confirm", f"I'm about to {what}. Say yes to confirm, or no to cancel.", pending_id=pid, decision=d)
        return Outcome("pending_oob", f"That needs out of band approval. Approve request {pid} from your phone.", pending_id=pid, decision=d)

    def _drop(self, p: Pending, why: str) -> None:
        self._pending.pop(p.id, None)
        self.audit.append("cancelled" if why != "expired" else "expired", id=p.id, why=why)

    def _expire(self) -> None:
        now = self.clock()
        for p in [p for p in self._pending.values() if p.expires < now]:
            self._drop(p, "expired")

    # ---- resolve ----
    def has_pending(self, kind: str | None = None) -> bool:
        with self._lock:
            self._expire()
            return any(kind is None or p.kind == kind for p in self._pending.values())

    def list_pending(self) -> list[dict]:
        with self._lock:
            self._expire()
            return [{"id": p.id, "kind": p.kind, "tier": int(p.tier), "readback": p.readback,
                     "digest": p.intent.digest(), "nonce": p.nonce, "expires_in": round(p.expires - self.clock(), 1),
                     "action": p.intent.action, "args": p.args} for p in self._pending.values()]

    def resolve_confirm(self, yes: bool) -> Outcome:
        """Voice yes/no for the (single) live T2 confirmation. Never resolves OOB requests."""
        with self._lock:
            self._expire()
            ps = [p for p in self._pending.values() if p.kind == "confirm"]
            if not ps:
                return Outcome("expired", "There's nothing waiting for confirmation.")
            p = ps[0]
            if not yes:
                self._pending.pop(p.id)
                self.audit.append("declined", id=p.id, via="voice")
                return Outcome("cancelled", "Okay, cancelled.", pending_id=p.id)
            self._pending.pop(p.id)
            self.audit.append("confirmed", id=p.id, via="voice", digest=p.intent.digest())
            return self._run(p.intent, p.spec, p.args, p.tier, p.id)

    def approve_oob(self, pending_id: str, token: str) -> Outcome:
        with self._lock:
            self._expire()
            p = self._pending.get(pending_id)
            if p is None or p.kind != "oob":
                return Outcome("expired", "No such pending approval.")
            if not self.verifier.verify_approval(p.id, p.nonce, p.intent.digest(), str(token)):
                p.bad_tokens += 1
                self.audit.append("oob_bad_token", id=p.id, n=p.bad_tokens)
                if p.bad_tokens >= 3:
                    self._drop(p, "too many bad tokens")
                return Outcome("denied", "Approval rejected.", pending_id=p.id)
            self._pending.pop(p.id)
            self.audit.append("approved", id=p.id, via="oob", digest=p.intent.digest())
            return self._run(p.intent, p.spec, p.args, p.tier, p.id)

    def cancel_all(self, why: str = "cancelled") -> int:
        with self._lock:
            ps = list(self._pending.values())
            for p in ps:
                self._drop(p, why)
            return len(ps)

    # ---- execute ----
    def _execute(self, intent: Intent, d: Decision) -> Outcome:
        return self._run(intent, d.spec, d.args, d.tier, None, d)

    def _run(self, intent, spec: ActionSpec, args: dict, tier: Tier, pid, d: Decision | None = None) -> Outcome:
        if self.cfg.broker.dry_run and spec.mutating:
            self.audit.append("dry_run", action=spec.name, args=args, tier=int(tier), id=pid)
            what = spec.describe(args) or spec.name
            return Outcome("dry_run", f"Dry run: I would {what}, but actions are disabled.", ExecResult(True, "DRY RUN", dry_run=True), pid, d)
        try:
            res = spec.run(args, self.ctx)
        except Exception as e:  # noqa: BLE001
            self.audit.append("failed", action=spec.name, id=pid, error=f"{type(e).__name__}: {e}")
            return Outcome("failed", f"That failed: {e}.", ExecResult(False, str(e)), pid, d)
        self.audit.append("executed" if res.ok else "failed", action=spec.name, id=pid, tier=int(tier),
                          output=res.output[:300])
        if res.ok:
            return Outcome("executed", "", res, pid, d)
        return Outcome("failed", f"That failed: {res.output}.", res, pid, d)
