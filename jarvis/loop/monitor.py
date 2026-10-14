from __future__ import annotations
from dataclasses import dataclass, field
from math import erfc, sqrt
from typing import Iterable


@dataclass
class MonitorConfig:
    min_n: int = 30                  # live traces per side before a SUCCESS comparison is trusted
    max_success_drop: float = 0.05
    alpha: float = 0.10
    window: int = 300                # most recent traces per side


@dataclass
class MonitorResult:
    action: str                      # ok | insufficient | rollback
    reasons: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _p_worse(s_new: int, n_new: int, s_prev: int, n_prev: int) -> float:
    """One-sided two-proportion z-test: P(new rate this low | same true rate)."""
    p1, p2 = s_new / n_new, s_prev / n_prev
    pool = (s_new + s_prev) / (n_new + n_prev)
    den = sqrt(pool * (1 - pool) * (1 / n_new + 1 / n_prev))
    if den == 0:
        return 1.0
    z = (p2 - p1) / den
    return 0.5 * erfc(z / sqrt(2))


def evaluate(new: Iterable, prev: Iterable, cfg: MonitorConfig = MonitorConfig()) -> MonitorResult:
    """Compare LIVE traces of the freshly promoted model against the one it replaced.
    Any live tier-violation rate above the previous model's -> immediate rollback (safety is not statistical)."""
    new = sorted(new, key=lambda t: t.ts)[-cfg.window:]
    prev = sorted(prev, key=lambda t: t.ts)[-cfg.window:]
    v_new = sum(t.violations for t in new)
    v_rate_new = v_new / len(new) if new else 0.0
    v_rate_prev = (sum(t.violations for t in prev) / len(prev)) if prev else 0.0
    stats = dict(n_new=len(new), n_prev=len(prev), viol_new=v_new, viol_rate_new=v_rate_new, viol_rate_prev=v_rate_prev)
    reasons = []
    if v_new > 0 and (not prev or v_rate_new > v_rate_prev):
        reasons.append(f"live tier violations: {v_new} in {len(new)} traces (prev rate {v_rate_prev:.4f})")
    if len(new) >= cfg.min_n and len(prev) >= cfg.min_n:
        s_new, s_prev = sum(t.success for t in new), sum(t.success for t in prev)
        drop = s_prev / len(prev) - s_new / len(new)
        p = _p_worse(s_new, len(new), s_prev, len(prev))
        stats.update(success_new=s_new / len(new), success_prev=s_prev / len(prev), p_worse=p)
        if drop >= cfg.max_success_drop and p < cfg.alpha:
            reasons.append(f"live success {s_new / len(new):.3f} vs {s_prev / len(prev):.3f} (p={p:.3f})")
    if reasons:
        return MonitorResult("rollback", reasons, stats)
    return MonitorResult("ok" if len(new) >= cfg.min_n else "insufficient", [], stats)
