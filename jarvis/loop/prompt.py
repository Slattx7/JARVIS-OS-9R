from __future__ import annotations
import json
from typing import Optional

from .traces import canon

SYSTEM_PROMPT = (
    "You are Jarvis's action planner. Given a request (plus optional recalled context and the granted "
    "permission tier), reply with ONLY a JSON array of tool calls in execution order: "
    '[{"tool": "<name>", "args": {...}}]. Reply [] if no action is needed. '
    "Never invent tools. Never call a tool above the granted tier."
)


def render_user(task: str, context: str = "", granted_tier: Optional[int] = None) -> str:
    parts = []
    if granted_tier is not None:
        parts.append(f"granted_tier: {granted_tier}")
    if context and context.strip():
        parts.append("context:\n" + context.strip())
    parts.append("request: " + task.strip())
    return "\n".join(parts)


def build_messages(task: str, context: str = "", granted_tier: Optional[int] = None) -> list:
    """The ONE prompt used for both SFT export and replay, so train/eval can't drift apart."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user(task, context, granted_tier)},
    ]


def assistant_text(calls: list) -> str:
    return canon([{"tool": c["tool"], "args": c.get("args", {})} for c in calls])


def _coerce(obj) -> Optional[list]:
    if not isinstance(obj, list):
        return None
    out = []
    for c in obj:
        if not isinstance(c, dict) or not isinstance(c.get("tool"), str):
            return None
        args = c.get("args", {})
        out.append({"tool": c["tool"], "args": args if isinstance(args, dict) else {"value": args}})
    return out


def extract_calls(text) -> Optional[list]:
    """Parse the first JSON array of tool calls out of model output (fences/prose tolerated). None = unparseable."""
    if not isinstance(text, str):
        return None
    dec = json.JSONDecoder()
    i = 0
    while True:
        j = text.find("[", i)
        if j < 0:
            return None
        try:
            obj, _ = dec.raw_decode(text[j:])
        except ValueError:
            i = j + 1
            continue
        calls = _coerce(obj)
        if calls is not None:
            return calls
        i = j + 1
