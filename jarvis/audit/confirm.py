"""Confirmers: how a `confirm` decision gets answered. Contract: ask(action, args, tier, intent, describe) ->
{"approved": bool, ...}. Only a literal `True` counts as approval. Implement a voice confirmer by subclassing
Confirmer: speak `describe`, listen, return {"approved": heard_yes, "via": "voice"} (False on timeout)."""
from __future__ import annotations

import fnmatch
from typing import Any, Callable, Dict, List, Optional, Tuple


class Confirmer:
    def ask(self, action: str, args: Dict[str, Any], tier: int, intent: str, describe: str) -> Dict[str, Any]:
        raise NotImplementedError


class DenyAll(Confirmer):
    def ask(self, action, args, tier, intent, describe):
        return {"approved": False, "via": "deny-all"}


class ApproveAll(Confirmer):
    """Unsafe: tests and throwaway sandboxes only."""

    def ask(self, action, args, tier, intent, describe):
        return {"approved": True, "via": "approve-all"}


class ScriptedConfirmer(Confirmer):
    """Deterministic answers by action glob, first match wins; unmatched -> default. Records every question."""

    def __init__(self, rules: Optional[Dict[str, bool]] = None, default: bool = False):
        self.rules: List[Tuple[str, bool]] = list((rules or {}).items())
        self.default = default
        self.asked: List[Dict[str, Any]] = []

    def ask(self, action, args, tier, intent, describe):
        self.asked.append({"action": action, "args": args, "tier": tier, "intent": intent})
        for pat, ans in self.rules:
            if fnmatch.fnmatchcase(action, pat):
                return {"approved": bool(ans), "via": "scripted"}
        return {"approved": self.default, "via": "scripted"}


class CallbackConfirmer(Confirmer):
    def __init__(self, fn: Callable[[str, Dict[str, Any], int, str, str], Any], via: str = "callback"):
        self.fn, self.via = fn, via

    def ask(self, action, args, tier, intent, describe):
        try:
            return {"approved": self.fn(action, args, tier, intent, describe) is True, "via": self.via}
        except Exception as e:  # noqa: BLE001  a crashing confirmer must never approve
            return {"approved": False, "via": self.via, "error": f"{type(e).__name__}: {e}"}


class CLIConfirmer(Confirmer):
    def __init__(self, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print):
        self.input_fn, self.output_fn = input_fn, output_fn

    def ask(self, action, args, tier, intent, describe):
        self.output_fn(f"[tier {tier}] {describe}" + (f"   (why: {intent})" if intent else ""))
        try:
            ans = self.input_fn("approve? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return {"approved": False, "via": "cli", "error": "no input"}
        return {"approved": ans in ("y", "yes"), "via": "cli"}
