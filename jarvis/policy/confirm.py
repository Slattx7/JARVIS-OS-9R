"""Confirmation channels. The readback is built deterministically from the *exact* action args --
never from model-written text -- so the owner confirms what will actually execute.

CONFIRM          : readback + "say yes".
CONFIRM_STRONG   : readback + random challenge ("confirm falcon 38") + min delay + echo guard.
                   Args too long to verify by voice are refused (use a screen confirmer).
"""
from __future__ import annotations
import random, re, time
from typing import Callable, Optional, Protocol

from .core import Action, Decision, Verdict

MAX_READBACK_CHARS = 160

WORDS = ["falcon", "copper", "river", "maple", "cobalt", "harbor", "lantern", "meadow", "orchid", "pebble",
         "quartz", "saddle", "thunder", "velvet", "walnut", "yonder", "amber", "bishop", "cinder", "dagger",
         "ember", "fjord", "glacier", "hazel", "indigo", "juniper", "kettle", "lotus", "marble", "nickel"]
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
YES = {"yes", "yeah", "yep", "yup", "confirm", "confirmed", "proceed", "affirmative", "sure", "ok", "okay"}
YES_PHRASES = ("go ahead", "do it")
NO = {"no", "nope", "cancel", "stop", "abort", "negative", "nah", "don't", "dont", "never", "wait"}
NO_PHRASES = ("do not", "hold on")


class Speaker(Protocol):
    def say(self, text: str) -> None: ...


class Listener(Protocol):
    def listen(self, timeout_s: float) -> Optional[str]: ...


def normalize_speech(text: str) -> str:
    """lowercase, strip punctuation, map number words -> digits ('thirty eight' -> '38')."""
    toks = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower().replace("-", " ")).split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in _TENS:
            val = _TENS[t]
            if i + 1 < len(toks) and toks[i + 1] in _ONES[1:10]:
                val += _ONES.index(toks[i + 1]); i += 1
            out.append(str(val))
        elif t in _ONES:
            out.append(str(_ONES.index(t)))
        else:
            out.append(t)
        i += 1
    return " ".join(out)


def parse_yes_no(text: Optional[str]) -> Optional[bool]:
    """True/False/None(ambiguous). Any NO signal wins; ambiguity is never a yes."""
    if not text: return None
    n = normalize_speech(text); toks = set(n.split())
    if toks & NO or any(p in n for p in NO_PHRASES): return False
    if toks & YES or any(p in n for p in YES_PHRASES): return True
    return None


def _fmt(v) -> str:
    return str(v) if not isinstance(v, str) else f"'{v}'"


def describe(a: Action, spec_describe: Optional[Callable[[Action], str]] = None) -> str:
    if spec_describe:
        return spec_describe(a)
    if a.name == "grant.request":
        x = a.args
        return (f"Grant: allow tier {x['max_tier']} actions"
                f"{' matching ' + x['prefix'] if x.get('prefix') else ''} for {int(x['ttl_s'] // 60)} minutes")
    if a.name == "lockdown.unlock":
        return f"Lift lockdown ({a.args.get('reason', '')})"
    parts = ", ".join(f"{k} {_fmt(v)}" for k, v in a.args.items())
    return f"{a.name} on {a.device}" + (f" with {parts}" if parts else "")


def max_arg_len(a: Action) -> int:
    return max([len(str(v)) for v in a.args.values()] or [0])


def _echo(reply_n: str, desc_n: str) -> bool:
    """Mic heard our own prompt? Reject replies containing a chunk of the readback."""
    chunk = desc_n[:24]
    return len(chunk) >= 12 and chunk in reply_n


class VoiceConfirmer:
    def __init__(self, speaker: Speaker, listener: Listener, *, clock=time.monotonic, sleep=time.sleep,
                 rng: Optional[random.Random] = None, timeout_s=8.0, strong_min_delay_s=2.0,
                 max_attempts=2, spec_lookup: Optional[Callable[[str], object]] = None):
        self.spk, self.lis, self.clock, self.sleep = speaker, listener, clock, sleep
        self.rng = rng or random.SystemRandom()
        self.timeout, self.delay, self.attempts = timeout_s, strong_min_delay_s, max_attempts
        self.spec_lookup = spec_lookup

    def _desc(self, a):
        spec = self.spec_lookup(a.name) if self.spec_lookup else None
        return describe(a, getattr(spec, "describe", None))

    def __call__(self, a: Action, d: Decision) -> bool:
        desc = self._desc(a)
        if d.verdict == Verdict.CONFIRM_STRONG:
            return self._strong(a, desc)
        return self._simple(a, desc)

    def _simple(self, a, desc) -> bool:
        note = " Note: arguments came from untrusted content." if a.tainted_args else ""
        self.spk.say(f"{desc}.{note} Say yes to confirm.")
        for i in range(self.attempts):
            r = parse_yes_no(self.lis.listen(self.timeout))
            if r is not None: return r
            if i + 1 < self.attempts: self.spk.say("Please say yes or cancel.")
        return False

    def _strong(self, a, desc) -> bool:
        if max_arg_len(a) > MAX_READBACK_CHARS:
            self.spk.say("That action has details too long to verify by voice. Use screen confirmation.")
            return False
        code = f"{self.rng.choice(WORDS)} {self.rng.randrange(10, 100)}"
        warn = " Warning: arguments came from untrusted content." if a.tainted_args else ""
        self.spk.say(f"High risk. {desc}.{warn} To confirm, say: confirm {code}. Or say cancel.")
        t0 = self.clock()
        desc_n = normalize_speech(desc)
        for _ in range(self.attempts):
            reply = self.lis.listen(self.timeout)
            if reply is None: continue
            rn = normalize_speech(reply)
            if self.clock() - t0 < self.delay:
                self.spk.say("Too fast. Try again.")
                self.sleep(max(0.0, self.delay - (self.clock() - t0)))
                t0 = self.clock(); continue
            if parse_yes_no(reply) is False: return False
            if _echo(rn, desc_n): continue
            if normalize_speech(f"confirm {code}") in rn: return True
            self.spk.say("Code did not match.")
        return False


class CLIConfirmer:
    def __init__(self, input_fn=input, print_fn=print, spec_lookup=None):
        self.inp, self.out, self.spec_lookup = input_fn, print_fn, spec_lookup

    def __call__(self, a, d) -> bool:
        spec = self.spec_lookup(a.name) if self.spec_lookup else None
        desc = describe(a, getattr(spec, "describe", None))
        flag = " [TAINTED: " + ",".join(a.tainted_args) + "]" if a.tainted_args else ""
        try:
            if d.verdict == Verdict.CONFIRM_STRONG:
                self.out(f"!! {desc}{flag}  (tier {int(d.tier)})")
                return self.inp("type the action name to confirm: ").strip() == a.name
            self.out(f"?? {desc}{flag}")
            return self.inp("proceed? [y/N] ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False                      # no tty / interrupted = deny


class ScriptedConfirmer:
    """Tests / automation: answers from a list, records what it was asked."""
    def __init__(self, answers=None, default=False):
        self.answers, self.default, self.asked = list(answers or []), default, []

    def __call__(self, a, d) -> bool:
        self.asked.append((a.name, d.verdict))
        return self.answers.pop(0) if self.answers else self.default


class CallbackConfirmer:
    """Adapter for a phone push / GUI prompt: fn(description, verdict, tainted)->bool."""
    def __init__(self, fn, spec_lookup=None):
        self.fn, self.spec_lookup = fn, spec_lookup

    def __call__(self, a, d) -> bool:
        spec = self.spec_lookup(a.name) if self.spec_lookup else None
        return bool(self.fn(describe(a, getattr(spec, "describe", None)), d.verdict, bool(a.tainted_args)))
