"""Loading and joining labeled data for training.

Labels are keyed by package_name, not session_id - you label "this app is
malicious/benign" once, then may collect multiple sessions for it over
time and all of them inherit that label.
"""
from __future__ import annotations

import json
from pathlib import Path

from .features import FEATURE_COLUMNS, extract_all_sessions


class LabelError(RuntimeError):
    """Raised when labels can't be loaded or joined with features."""


def load_labels(path: str | Path) -> dict[str, int]:
    """Load a {package_name: 0|1} JSON file. 0 = benign, 1 = malicious."""
    path = Path(path)
    if not path.exists():
        raise LabelError(f"labels file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, int] = {}
    for package_name, label in raw.items():
        if label not in (0, 1):
            raise LabelError(f"label for {package_name!r} must be 0 or 1, got {label!r}")
        labels[package_name] = int(label)
    return labels


def build_training_set(
    log_paths: list[str | Path],
    labels_path: str | Path,
) -> tuple[list[dict], list[int], list[str]]:
    """Join session features against labels by package_name.

    Returns (feature_rows, y, session_ids) for every collected session
    whose package_name has a label. Raises LabelError if nothing matches.
    """
    labels = load_labels(labels_path)
    features = extract_all_sessions(log_paths)

    rows: list[dict] = []
    y: list[int] = []
    session_ids: list[str] = []
    for f in features:
        if f.package_name not in labels:
            continue
        row = f.to_row()
        rows.append({col: row[col] for col in FEATURE_COLUMNS})
        y.append(labels[f.package_name])
        session_ids.append(f.session_id)

    if not rows:
        raise LabelError(
            "no collected sessions matched any package_name in the labels file - "
            "check that labels.json keys match the package names you collected"
        )

    return rows, y, session_ids
