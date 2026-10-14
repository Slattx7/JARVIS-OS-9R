"""File-based pause switch. While the file exists the policy engine (rule `paused`) denies every action above
tier 0, and `jarvis_trace pause` / `resume` toggle it from any shell, SSH session or cron job.

Fail closed: an unreadable/corrupt pause file still counts as paused.
Path: argument > $JARVIS_PAUSE_FILE > ~/.jarvis/PAUSE
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def default_path() -> str:
    return os.environ.get("JARVIS_PAUSE_FILE") or os.path.expanduser("~/.jarvis/PAUSE")


def state(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    p = os.path.expanduser(path) if path else default_path()
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {"reason": "pause file has unexpected content", "corrupt": True}
    except (OSError, ValueError):
        return {"reason": "pause file unreadable", "corrupt": True}


def engage(path: Optional[str] = None, reason: str = "", who: str = "cli") -> Dict[str, Any]:
    p = os.path.expanduser(path) if path else default_path()
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    d = {"ts": time.time(), "reason": reason, "who": who}
    tmp = p + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, p)
    return d


def release(path: Optional[str] = None) -> bool:
    p = os.path.expanduser(path) if path else default_path()
    try:
        os.remove(p)
        return True
    except FileNotFoundError:
        return False
