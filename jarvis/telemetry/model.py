"""GBT baseline: train, evaluate, persist, and score.

HistGradientBoostingClassifier rather than XGBoost/LightGBM - it's part of
scikit-learn core (no extra native build dependency), well-supported by
SHAP's TreeExplainer, and plenty for a tabular feature set this size.
Swapping in XGBoost later is a small, contained change if you outgrow it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS


# Attribute stamped onto a trained model recording which feature columns it
# was fit on, so a model from an older feature set fails loudly and clearly
# at load time instead of with an opaque sklearn shape error at score time.
FEATURE_COLUMNS_ATTR = "warden_feature_columns_"


class TrainingError(RuntimeError):
    """Raised when there isn't enough/varied enough data to train or evaluate."""


def _auto_min_samples_leaf(n_train: int) -> int:
    """HistGradientBoostingClassifier defaults to min_samples_leaf=20, which
    silently produces near-unsplittable (near-random) trees on the small
    labeled sets this project will realistically start with - a training
    run on ~37 rows scored 46% accuracy on a trivially separable synthetic
    problem until this was capped down. Scales down for small N, caps at
    sklearn's own default once there's enough data for it to make sense.
    """
    return max(1, min(20, n_train // 10))


def train_classifier(
    rows: list[dict],
    y: list[int],
    test_size: float = 0.25,
    random_state: int = 0,
    min_samples_leaf: int | None = None,
) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    """Train the GBT baseline and evaluate on a held-out split.

    Returns (fitted_model, metrics). Raises TrainingError if there isn't
    enough data, or only one class is present, to do a meaningful split.

    `min_samples_leaf` is auto-scaled to the training set size unless you
    override it - see `_auto_min_samples_leaf`.
    """
    if len(set(y)) < 2:
        raise TrainingError(
            "training data has only one class present - need both benign and malicious examples"
        )
    if len(rows) < 4:
        raise TrainingError(f"only {len(rows)} labeled sessions - need at least 4 to hold out a test split")

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    if min_samples_leaf is None:
        min_samples_leaf = _auto_min_samples_leaf(len(X_train))

    model = HistGradientBoostingClassifier(random_state=random_state, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    setattr(model, FEATURE_COLUMNS_ATTR, list(FEATURE_COLUMNS))

    y_pred = model.predict(X_test)
    metrics: dict[str, Any] = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "min_samples_leaf": min_samples_leaf,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if len(set(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return model, metrics


def save_model(model: HistGradientBoostingClassifier, path: str | Path) -> None:
    joblib.dump(model, path)


def load_model(path: str | Path) -> HistGradientBoostingClassifier:
    path = Path(path)
    if not path.exists():
        raise TrainingError(f"model file not found: {path}")
    model = joblib.load(path)
    trained_on = getattr(model, FEATURE_COLUMNS_ATTR, None)
    if trained_on != list(FEATURE_COLUMNS):
        was = "an unknown feature set" if trained_on is None else f"{len(trained_on)} feature columns"
        raise TrainingError(
            f"{path} was trained on {was}, but this version of warden uses "
            f"{len(FEATURE_COLUMNS)} - retrain with `warden train`"
        )
    return model


def score_session(model: HistGradientBoostingClassifier, row: dict) -> float:
    """Risk score in [0, 1] for one session's feature row."""
    X = pd.DataFrame([{col: row[col] for col in FEATURE_COLUMNS}])
    return float(model.predict_proba(X)[0, 1])
