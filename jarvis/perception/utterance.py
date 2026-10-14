from __future__ import annotations

import re

_DANGLING = {"and", "or", "but", "so", "then", "to", "the", "a", "an", "for", "with", "of", "in", "on", "at", "my",
             "your", "is", "are", "if", "because", "that", "which"}


def looks_complete(text: str) -> bool:
    """Heuristic used by fast-commit: Whisper-style terminal punctuation, >= 2 words, no trailing ellipsis or
    dangling function word. Conservative on purpose: a false 'incomplete' only costs the normal endpoint wait."""
    t = text.strip()
    if not t or t.endswith(("...", "…")) or t[-1] not in ".?!":
        return False
    words = re.findall(r"[a-z']+", t.lower())
    return len(words) >= 2 and words[-1] not in _DANGLING
