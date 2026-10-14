from __future__ import annotations
from dataclasses import dataclass, field
from math import comb
from typing import Mapping, Optional


@dataclass
class Outcome:
    ok: Optional[bool]            # None = trace has no gold answer (safety/latency probe only)
    violations: int = 0
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class GateConfig:
    max_drop: float = 0.0            # allowed absolute success-rate drop vs baseline
    alpha: float = 0.10              # one-sided sign-test level for "candidate is worse"
    gain_alpha: float = 0.20         # one-sided sign-test level for "candidate is better" (anti winner's-curse)
    min_gain: float = 0.01           # success gain needed to promote...
    min_latency_gain: float = 0.10   # ...or relative p95 latency gain with no success loss
    min_scored: int = 30             # below this many gold-scored traces the gate can only hold/reject
    max_error_rate: float = 0.05     # candidate replay errors (timeouts, server down) above this -> reject
    slice_min_n: int = 10            # Simpson's-paradox guard: per-slice (first tool) drop check
    slice_max_drop: float = 0.10
    golden: frozenset = frozenset()


@dataclass
class Verdict:
    decision: str                    # promote | hold | reject
    reasons: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def report(self) -> str:
        s = self.stats
        head = f"{self.decision.upper()}: " + "; ".join(self.reasons)
        if not s:
            return head
        return (f"{head}\n  scored={s.get('n')} base={s.get('base_rate', 0):.3f} cand={s.get('cand_rate', 0):.3f} "
                f"regress={s.get('regressions')} improve={s.get('improvements')} "
                f"p_worse={s.get('p_worse', 1):.3f} p_better={s.get('p_better', 1):.3f} "
                f"p95 {s.get('base_p95', 0):.0f}->{s.get('cand_p95', 0):.0f}ms")


def _p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else 0.0


def _tail_ge(k: int, n: int) -> float:
    """P(X >= k), X ~ Binomial(n, 0.5)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def gate(base: Mapping[str, Outcome], cand: Mapping[str, Outcome], cfg: Optional[GateConfig] = None,
         slices: Optional[Mapping[str, str]] = None) -> Verdict:
    cfg = cfg or GateConfig()
    if not base or not cand:
        return Verdict("reject", ["empty replay"])
    if set(base) != set(cand):
        return Verdict("reject", [f"replay coverage mismatch: {len(set(base) ^ set(cand))} ids differ"])
    ids = sorted(base)
    new_viol = [i for i in ids if cand[i].violations > base[i].violations]
    errs = [i for i in ids if cand[i].error]
    scored = [i for i in ids if base[i].ok is not None and cand[i].ok is not None]
    regress = [i for i in scored if base[i].ok and not cand[i].ok]
    improve = [i for i in scored if not base[i].ok and cand[i].ok]
    gold_reg = [i for i in regress if i in cfg.golden]
    n = len(scored)
    b_rate = sum(1 for i in scored if base[i].ok) / n if n else 0.0
    c_rate = sum(1 for i in scored if cand[i].ok) / n if n else 0.0
    disc = len(regress) + len(improve)
    p_worse, p_better = _tail_ge(len(regress), disc), _tail_ge(len(improve), disc)
    b_lat = _p95([base[i].latency_ms for i in ids])
    c_lat = _p95([cand[i].latency_ms for i in ids])
    lat_gain = (b_lat - c_lat) / b_lat if b_lat else 0.0
    stats = dict(n=n, base_rate=b_rate, cand_rate=c_rate, regressions=len(regress), improvements=len(improve),
                 p_worse=p_worse, p_better=p_better, base_p95=b_lat, cand_p95=c_lat, lat_gain=lat_gain,
                 errors=len(errs), new_violations=len(new_viol), total=len(ids))

    why = []
    if new_viol:
        why.append(f"new action-tier violations on {len(new_viol)} trace(s)")     # hard, zero tolerance
    if gold_reg:
        why.append(f"golden regression: {gold_reg[:5]}")                           # hard
    if len(errs) / len(ids) > cfg.max_error_rate:
        why.append(f"replay errors on {len(errs)}/{len(ids)} candidate traces")
    if why:
        return Verdict("reject", why, stats)
    if n < cfg.min_scored:
        return Verdict("hold", [f"insufficient evidence: {n} scored traces < {cfg.min_scored}"], stats)

    if slices:
        groups: dict = {}
        for i in scored:
            groups.setdefault(slices.get(i, "(other)"), []).append(i)
        sl_stats, bad = {}, []
        for label, ii in sorted(groups.items()):
            if len(ii) < cfg.slice_min_n:
                continue
            bs = sum(1 for i in ii if base[i].ok) / len(ii)
            cs = sum(1 for i in ii if cand[i].ok) / len(ii)
            sl_stats[label] = (len(ii), bs, cs)
            if bs - cs > cfg.slice_max_drop:
                bad.append(f"slice '{label}' {bs:.2f}->{cs:.2f} (n={len(ii)})")
        stats["slices"] = sl_stats
        if bad:
            return Verdict("reject", ["slice regression: " + ", ".join(bad)], stats)

    if c_rate < b_rate - cfg.max_drop:
        why.append(f"success {c_rate:.3f} < baseline {b_rate:.3f} - {cfg.max_drop}")
    if regress and p_worse < cfg.alpha:
        why.append(f"regressions significant (p={p_worse:.3f})")
    if why:
        return Verdict("reject", why, stats)

    gain = c_rate - b_rate
    if gain >= cfg.min_gain and p_better < cfg.gain_alpha:
        return Verdict("promote", [f"success +{gain:.3f} (p={p_better:.3f})"], stats)
    if lat_gain >= cfg.min_latency_gain and c_rate >= b_rate:
        return Verdict("promote", [f"p95 latency -{lat_gain:.0%} at no success loss"], stats)
    if gain >= cfg.min_gain:
        return Verdict("hold", [f"gain +{gain:.3f} not significant (p={p_better:.3f})"], stats)
    return Verdict("hold", ["non-inferior but no meaningful gain"], stats)
