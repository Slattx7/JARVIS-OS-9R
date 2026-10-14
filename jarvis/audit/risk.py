"""Risk hook: plug an external scorer (e.g. the warden app-behaviour classifier) into the policy.

The Executor records the score as its own `risk` span (frozen in replay, stubbed to risk=1.0 when there is no
recording: fail closed) and passes it to the policy as ctx.risk, where the default rules do:
risk >= 0.5 -> at least confirm, risk >= 0.85 -> deny.

    risk = RiskAdapter(classify=warden.classify,
                       extractors={"app.install": lambda a: {"package": a["pkg"]},
                                   "net.*":       lambda a: {"host": a["host"]}})
    Executor(registry, engine, confirmer, risk_fn=risk)

`classify(features) -> float | {"risk": float, "label": str, "reasons": [...]}`. Actions with no extractor get
`default_risk`. Any exception from the classifier becomes risk=1.0 (fail closed), never a silent 0.
"""
from __future__ import annotations

import fnmatch
from typing import Any, Callable, Dict, Optional


def _clip(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 1.0
    if v != v:  # NaN
        return 1.0
    return max(0.0, min(1.0, v))


def fail_closed(fn: Callable[..., Any]) -> Callable[..., Dict[str, Any]]:
    def wrapper(**kw: Any) -> Dict[str, Any]:
        try:
            out = fn(**kw)
        except Exception as e:  # noqa: BLE001
            return {"risk": 1.0, "label": "scorer_error", "error": f"{type(e).__name__}: {e}"}
        if isinstance(out, dict):
            return {**out, "risk": _clip(out.get("risk"))}
        return {"risk": _clip(out)}

    wrapper.__name__ = getattr(fn, "__name__", "risk")
    return wrapper


class RiskAdapter:
    def __init__(self, classify: Callable[[Dict[str, Any]], Any],
                 extractors: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
                 default_risk: float = 0.0):
        self.classify = classify
        self.extractors = dict(extractors or {})
        self.default_risk = default_risk

    def _extractor(self, action: str) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
        if action in self.extractors:
            return self.extractors[action]
        for pat, ex in self.extractors.items():
            if fnmatch.fnmatchcase(action, pat):
                return ex
        return None

    def __call__(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        ex = self._extractor(action)
        if ex is None:
            return {"risk": _clip(self.default_risk), "label": "unscored"}
        return fail_closed(lambda: self.classify(ex(args)))()
