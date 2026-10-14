"""SHAP-based explanation of a single session's risk score, optionally
rendered into plain English via a local Ollama model.

The SHAP computation is the real, tested signal. The Ollama narration is
a nice-to-have layered on top and must degrade gracefully - scoring and
explaining never depend on an LLM being reachable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
try:
    import shap
except ImportError:
    shap = None
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_COLUMNS

logger = logging.getLogger("warden.explain")


@dataclass
class FeatureContribution:
    feature: str
    value: float
    shap_value: float  # positive = pushes the score toward "malicious"


def explain_session(
    model: HistGradientBoostingClassifier,
    row: dict,
    top_k: int = 5,
) -> list[FeatureContribution]:
    """Top-k features driving this session's score, ranked by |SHAP value|."""
    if shap is None:
        return [FeatureContribution(feature=col, value=float(row.get(col, 0.0)), shap_value=0.0) for col in FEATURE_COLUMNS[:top_k]]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Different SHAP/model version combos return either a single
    # (n_samples, n_features) array for the positive class, or a list of
    # per-class arrays - handle both rather than assuming one shape.
    if isinstance(shap_values, list):
        values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        values = shap_values[0]

    contributions = [
        FeatureContribution(feature=col, value=float(X.iloc[0][col]), shap_value=float(val))
        for col, val in zip(FEATURE_COLUMNS, values)
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
    return contributions[:top_k]


def _fallback_summary(risk_score: float, contributions: list[FeatureContribution]) -> str:
    factors = "; ".join(
        f"{c.feature}={c.value:g} ({'+' if c.shap_value >= 0 else ''}{c.shap_value:.3f})"
        for c in contributions
    )
    return f"risk={risk_score:.2f} — top factors: {factors}"


def explain_in_plain_english(
    package_name: str,
    risk_score: float,
    contributions: list[FeatureContribution],
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5",
    timeout_s: float = 15.0,
) -> str:
    """Ask a local Ollama model to narrate why this session got flagged.

    Falls back to a plain structured summary of the SHAP contributions if
    Ollama isn't reachable, returns an empty/odd response, or times out.
    """
    factors = "; ".join(
        f"{c.feature}={c.value:g} ({'+' if c.shap_value >= 0 else ''}{c.shap_value:.3f})"
        for c in contributions
    )
    prompt = (
        f"An Android app '{package_name}' was flagged by a behavioral risk "
        f"classifier with a risk score of {risk_score:.2f} (0=benign, 1=malicious). "
        f"The top contributing features (feature=value, SHAP contribution) were: "
        f"{factors}. In 2-3 plain-English sentences, explain what this app's "
        f"behavior pattern suggests and why it looks risky. Be concrete about "
        f"which behaviors mattered most."
    )

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=timeout_s,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        if text:
            return text
        logger.warning("Ollama returned an empty response, falling back to raw SHAP summary")
    except requests.RequestException as exc:
        logger.warning("Ollama unreachable (%s), falling back to raw SHAP summary", exc)

    return _fallback_summary(risk_score, contributions)
