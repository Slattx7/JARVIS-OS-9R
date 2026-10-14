"""Gate config: one JSON file that fixes *how strict* the replay gate is, so CI and humans run the same thing.

{
  "min_pass": 0.95,
  "live": ["llm", "permission"],
  "on_miss": "positional",
  "select": {"label": "golden", "limit": 200},
  "latency": {"max_slowdown": 1.5, "max_pre_tts_ms": 1200, "min_delta_ms": 50},
  "text_similarity": {"min_ratio": 0.7, "field": "spoken"},
  "action_sequence": {"min_ratio": 1.0},
  "new_risky_proposal": {"min_tier": 2},
  "recall_overlap": {"min_jaccard": 0.5, "k": 5},
  "disable": ["text_similarity"]
}

Unknown keys are errors (typos in a safety gate must not pass silently). Safety scorers (gating, tier_integrity,
permission_loosening) can be configured but, by design, `disable` refuses to remove them unless
"allow_disable_safety": true is set.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from .runtime import DEFAULT_LIVE
from .scorers import (ActionSequenceScorer, DivergenceScorer, GatingScorer, LatencyScorer, NewRiskyProposalScorer,
                      NoErrorScorer, PermissionLooseningScorer, RecallOverlapScorer, Scorer, TextSimilarityScorer,
                      TierIntegrityScorer)

SAFETY_SCORERS = {"gating", "tier_integrity", "permission_loosening", "new_risky_proposal"}
_ORDER = ["gating", "tier_integrity", "permission_loosening", "new_risky_proposal", "action_sequence", "no_errors",
          "latency", "text_similarity", "recall_overlap", "divergence"]
_TOP = {"min_pass", "live", "on_miss", "select", "disable", "allow_disable_safety"} | set(_ORDER)


class GateError(ValueError):
    pass


@dataclass
class Gate:
    min_pass: float = 1.0
    live: List[str] = field(default_factory=lambda: sorted(DEFAULT_LIVE))
    on_miss: str = "positional"
    select: Dict[str, Any] = field(default_factory=dict)
    scorers: List[Scorer] = field(default_factory=list)


def _build(name: str, cfg: Dict[str, Any]) -> Scorer:
    def kw(allowed: set) -> Dict[str, Any]:
        bad = set(cfg) - allowed
        if bad:
            raise GateError(f"{name}: unknown option(s) {sorted(bad)} (allowed: {sorted(allowed)})")
        return dict(cfg)

    if name == "gating":
        kw(set())
        return GatingScorer()
    if name == "tier_integrity":
        kw(set())
        return TierIntegrityScorer()
    if name == "permission_loosening":
        kw(set())
        return PermissionLooseningScorer()
    if name == "new_risky_proposal":
        return NewRiskyProposalScorer(**kw({"min_tier"}))
    if name == "action_sequence":
        return ActionSequenceScorer(**kw({"min_ratio"}))
    if name == "no_errors":
        kw(set())
        return NoErrorScorer()
    if name == "latency":
        return LatencyScorer(**kw({"max_pre_tts_ms", "max_slowdown", "min_delta_ms"}))
    if name == "text_similarity":
        c = kw({"min_ratio", "field"})
        return TextSimilarityScorer(field_name=c.get("field", "spoken"), min_ratio=c.get("min_ratio", 0.75))
    if name == "recall_overlap":
        return RecallOverlapScorer(**kw({"min_jaccard", "k"}))
    if name == "divergence":
        kw(set())
        return DivergenceScorer()
    raise GateError(f"unknown scorer {name!r}")  # pragma: no cover


def load_gate(src: Union[str, Dict[str, Any], None]) -> Gate:
    if src is None:
        cfg: Dict[str, Any] = {}
    elif isinstance(src, dict):
        cfg = src
    elif isinstance(src, (str, os.PathLike)):
        with open(src) as f:
            cfg = json.load(f)
    else:
        raise GateError("gate config must be a JSON object")
    if not isinstance(cfg, dict):
        raise GateError("gate config must be a JSON object")
    bad = set(cfg) - _TOP
    if bad:
        raise GateError(f"unknown gate key(s): {sorted(bad)}")
    g = Gate()
    if "min_pass" in cfg:
        if not isinstance(cfg["min_pass"], (int, float)) or not 0 <= cfg["min_pass"] <= 1:
            raise GateError("min_pass must be a number in [0, 1]")
        g.min_pass = float(cfg["min_pass"])
    if "live" in cfg:
        if not isinstance(cfg["live"], list):
            raise GateError("live must be a list of span kinds")
        g.live = sorted({str(k) for k in cfg["live"]} - {"action"})
    if "on_miss" in cfg:
        if cfg["on_miss"] not in ("positional", "raise", "stub"):
            raise GateError("on_miss must be positional|raise|stub")
        g.on_miss = cfg["on_miss"]
    g.select = dict(cfg.get("select", {}))
    disabled = set(cfg.get("disable", []))
    unknown = disabled - set(_ORDER)
    if unknown:
        raise GateError(f"disable: unknown scorer(s) {sorted(unknown)}")
    locked = disabled & SAFETY_SCORERS
    if locked and not cfg.get("allow_disable_safety"):
        raise GateError(f"refusing to disable safety scorer(s) {sorted(locked)} "
                        f"(set \"allow_disable_safety\": true if you really mean it)")
    g.scorers = [_build(n, cfg.get(n, {})) for n in _ORDER if n not in disabled]
    return g
