"""Intent lines: the LLM proposes actions as `@@ {json}` lines; they are never spoken."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

TAINTED_SOURCES = frozenset({"tool_result", "screen", "notification", "web", "clipboard", "file"})


class IntentError(ValueError):
    pass


@dataclass(frozen=True)
class Intent:
    action: str
    args: dict = field(default_factory=dict)
    reason: str = ""
    provenance: tuple[str, ...] = ("user_voice",)

    @property
    def tainted(self) -> bool:
        return any(p in TAINTED_SOURCES for p in self.provenance)

    def digest(self) -> str:
        blob = json.dumps({"action": self.action, "args": self.args}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def parse_intent(raw: str, provenance: tuple[str, ...] = ("user_voice",)) -> Intent:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise IntentError(f"invalid JSON: {e.msg}") from None
    if not isinstance(obj, dict):
        raise IntentError("intent must be a JSON object")
    action, args = obj.get("action"), obj.get("args", {})
    if not isinstance(action, str) or not action:
        raise IntentError("missing action")
    if not isinstance(args, dict):
        raise IntentError("args must be an object")
    reason = obj.get("reason", "")
    return Intent(action, args, reason if isinstance(reason, str) else "", provenance)


class StreamSplitter:
    """Splits a token stream into (speech, intent_lines). A line is an intent iff it starts with '@@'.
    Holds back at most two characters at a line start, so speech latency is unaffected."""

    def __init__(self) -> None:
        self._line_start = True
        self._hold = ""
        self._in_intent = False
        self._buf = ""

    def feed(self, text: str) -> tuple[str, list[str]]:
        speech: list[str] = []
        intents: list[str] = []
        for ch in text:
            if self._in_intent:
                if ch == "\n":
                    intents.append(self._buf)
                    self._buf, self._in_intent, self._line_start = "", False, True
                else:
                    self._buf += ch
                continue
            if self._line_start:
                self._hold += ch
                if self._hold == "@@":
                    self._in_intent, self._hold, self._line_start = True, "", False
                    continue
                if "@@".startswith(self._hold):
                    continue
                speech.append(self._hold)
                self._line_start = self._hold.endswith("\n")
                self._hold = ""
                continue
            speech.append(ch)
            if ch == "\n":
                self._line_start = True
        return "".join(speech), intents

    def flush(self) -> tuple[str, list[str]]:
        speech, intents = "", []
        if self._in_intent and self._buf.strip():
            intents.append(self._buf)
        elif self._hold:
            speech = self._hold
        self.__init__()
        return speech, intents
