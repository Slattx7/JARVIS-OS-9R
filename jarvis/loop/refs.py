from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRef:
    """A registry version + where its artifact lives (adapter dir, gguf path, or base-model id)."""
    version: str
    artifact: str = ""
