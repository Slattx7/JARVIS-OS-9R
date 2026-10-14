"""Deterministic local command shortcuts. Common requests skip the LLM entirely: STT -> broker -> TTS.

Why: ~0.5-1 s faster than a model round-trip, works when llama-server is down, and can't hallucinate.
Safety: a fastpath match is just an Intent with provenance=user_voice that goes through the same broker
(validation, tiers, rate limits, audit) as anything the LLM proposes. Rules are whole-utterance anchored, so
"play some jazz" or "what's the battery saver setting" fall through to the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .actions import DEVNAME, ExecResult, fmt_duration
from .intents import Intent

MAX_WORDS = 12


@dataclass(frozen=True)
class FastMatch:
    intent: Intent | None = None
    say: str = ""                                   # canned reply when there is no intent ("" = stay silent)
    render: Callable[[ExecResult], str] | None = None  # spoken text on success


# ---------------- text normalisation ----------------
_FILLER = re.compile(r"^(?:(?:hey|hi|ok|okay|please|jarvis|can you|could you|would you|will you|i want you to|i need you to|"
                     r"i'd like you to|go ahead and|just)\b[ ,]*)+", re.I)
_TAIL = re.compile(r"(?:[ ,]*\b(?:please|thanks|thank you|now|right now|for me|jarvis)\b)+$")
_UNITS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])}
_TENS = {w: 10 * (i + 2) for i, w in enumerate(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])}


def normalize(text: str, strip_tail: bool = True) -> str:
    t = text.lower().strip().replace("%", " percent")
    t = re.sub(r"[^\w\s'.-]", " ", t)
    t = t.replace("-", " ")
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _FILLER.sub("", t).strip()
    if strip_tail:
        t = _TAIL.sub("", t).strip()
    return t


def numberize(t: str) -> str:
    """'ninety minutes' -> '90 minutes'; 'forty five percent' -> '45 percent'; digits pass through."""
    toks, out, i = t.split(), [], 0
    while i < len(toks):
        total, cur, j, seen = 0, 0, i, False
        while j < len(toks):
            w = toks[j]
            if w in ("a", "an") and not seen and j + 1 < len(toks) and toks[j + 1] in ("hundred", "thousand"):
                cur += 1   # "a hundred" == one hundred
            elif w in _UNITS:
                cur += _UNITS[w]
            elif w in _TENS:
                cur += _TENS[w]
            elif w == "hundred" and seen:
                cur = (cur or 1) * 100
            elif w == "thousand" and seen:
                total, cur = total + (cur or 1) * 1000, 0
            elif w == "and" and seen and j + 1 < len(toks) and (toks[j + 1] in _UNITS or toks[j + 1] in _TENS):
                j += 1
                continue
            else:
                break
            seen = True
            j += 1
        if seen:
            out.append(str(total + cur))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def parse_duration(t: str) -> int | None:
    t = numberize(t)
    t = re.sub(r"\bhalf an? hour\b|\bhalf hour\b", "30 minutes", t)
    t = re.sub(r"\b(?:an?|1) hour and a half\b", "1.5 hours", t)
    t = re.sub(r"\b(?:an?|1) minute and a half\b", "1.5 minutes", t)
    total, found = 0.0, False
    for m in re.finditer(r"(\d+(?:\.\d+)?|an?)( and a half)? (hours?|hrs?|minutes?|mins?|seconds?|secs?)\b", t):
        n = 1.0 if m.group(1) in ("a", "an") else float(m.group(1))
        n += 0.5 if m.group(2) else 0.0
        total += n * {"h": 3600, "m": 60, "s": 1}[m.group(3)[0]]
        found = True
    return round(total) if found else None


# ---------------- device detection ----------------
_R9 = re.compile(r"\b(?:(?:nine|9) ?(?:r|are)|one ?plus (?:nine|9)(?: ?(?:r|are))?|"
                 r"(?:my |the )?(?:other|second|rooted) (?:phone|one|device))\b")
_PRO = re.compile(r"\b(?:one ?plus (?:fourteen|14)(?: ?pro)?|(?:fourteen|14) pro|"
                  r"(?:my |the )?(?:main|daily|primary) (?:phone|one|device))\b")
_DANGLING = re.compile(r"\b(?:on|of|for|from|in|at|with)\s*(?:my|the)?\s*$")
_ON_PHONE = re.compile(r"\b(?:on|of|for|from|in)\s+(?:my|the)\s+(?:phone|device)\s*$")


def detect_device(t: str, default: str) -> tuple[str, str]:
    dev = default
    if _R9.search(t):
        dev = "r9"
    elif _PRO.search(t):
        dev = "pro14"
    t = _R9.sub(" ", t)
    t = _PRO.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _ON_PHONE.sub("", t).strip()
    t = _DANGLING.sub("", t).strip()
    return dev, t


# ---------------- rules ----------------
_STOP = re.compile(r"^(?:stop|cancel|never ?mind|forget it|be quiet|quiet|silence|shut up|enough|that's enough|"
                   r"that will be all|that'll be all|nothing|no thanks|no thank you|all done|i'm done|nope)$")
_THANKS = re.compile(r"^(?:thanks|thank you|thanks a lot|thank you very much|cheers|much appreciated|thanks jarvis|thank you jarvis)$")
_PLAY = re.compile(r"^(?:pause|play|resume|unpause)(?: the)?(?: music| song| track| playback| media| video| podcast| audio)?$")
_MSTOP = re.compile(r"^stop (?:the )?(?:music|song|track|playback|media|video|podcast|audio)$")
_NEXT = re.compile(r"^(?:next|skip)(?: this| the)?(?: song| track| one| video)?$")
_PREV = re.compile(r"^(?:previous|prev)(?: song| track| one)?$|^(?:last|go back to the (?:last|previous)|play the (?:last|previous)) (?:song|track|one)$")
_VOL_SET = re.compile(r"^(?:(?:set|turn|put|make|change) )?(?:the )?(?:media )?(?:volume|sound)(?: level)? (?:to |at |on )?(\d{1,3})(?: percent)?$")
_VOL_SET2 = re.compile(r"^(?:set|turn|put|make) (?:it|the volume|the sound) (?:to|at) (\d{1,3})(?: percent)?$")
_MUTE = re.compile(r"^(?:mute|silence)(?: the)?(?: phone| sound| volume| music| media)?$")
_VOL_STEP = re.compile(r"^(?:(?:turn|crank|bring|make)(?: it| the volume| the music| the sound)? (up|down)|(?:volume|sound) (up|down)|"
                       r"(?:a little |a bit )?(louder|quieter|softer))$")
_TIME = re.compile(r"^(?:what(?:'s| is)|tell me|give me|do you (?:have|know)|got)? ?(?:the )?(?:current )?time(?: is it)?$|^what time is it$")
_DATE = re.compile(r"^(?:what(?:'s| is)|tell me)(?: the)? (?:date|day)(?: is it| it is)?(?: today)?$|^what day is (?:it|today)$|^what(?:'s| is) today(?:'s date)?$")
_TIMER_CANCEL = re.compile(r"^(?:cancel|stop|clear|delete|remove|turn off|kill)(?: all)?(?: the| my)?(?: running| active)? timers?(?: off)?$")
_TIMER_LIST = re.compile(r"\btimers?\b.*\b(?:left|remaining|running|active)\b|\b(?:how much|how long|what|any|list|check|show)\b.*\btimers?\b")
_NOTE = re.compile(r"^(?:take a note|make a note|add a note|note to self|note that|note|remember that|remember to)[ ,:]*(.+)$", re.I)
_BATT_BAD = ("saver", "optimi", "setting", "replace", "charger", "drain", "health", "cycle", "last", "long", "app", "percentage of")
_BATT_LEAD = re.compile(r"^(?:what(?:'s| is| about)?|how much|how(?:'s| is)|check|tell me|show me|give me|read|and|my|the|phone|battery)\b")


def _upfirst(s: str) -> str:
    return s[:1].upper() + s[1:]


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _say_time(r: ExecResult) -> str:
    h, m = r.data["hour"], r.data["minute"]
    h12 = h % 12 or 12
    ap = "AM" if h < 12 else "PM"
    return f"It's {h12} {ap}." if m == 0 else f"It's {h12}:{m:02d} {ap}."


def _say_date(r: ExecResult) -> str:
    d = r.data
    return f"Today is {d['weekday']}, {d['month']} {_ordinal(d['day'])}."


def _say_battery(r: ExecResult) -> str:
    d = r.data
    return f"{_upfirst(DEVNAME.get(d['device'], d['device']))} is at {d['level']} percent and {d['status']}."


class Fastpath:
    def __init__(self, default_device: str = "pro14"):
        self.default_device = default_device

    def _intent(self, action: str, **args) -> Intent:
        return Intent(action, args, "fastpath", ("user_voice",))

    def match(self, text: str) -> FastMatch | None:
        t0 = normalize(text, strip_tail=False)
        if not t0 or len(t0.split()) > MAX_WORDS:
            return None
        if _THANKS.match(t0):
            return FastMatch(say="You're welcome.")
        t = normalize(text)
        if not t:
            return None
        if _STOP.match(t):
            return FastMatch(say="")
        dev, t = detect_device(t, self.default_device)
        nt = numberize(t)

        if _PLAY.match(t):
            return FastMatch(self._intent("device.media.control", device=dev, command="play_pause"), render=lambda r: "Okay.")
        if _MSTOP.match(t):
            return FastMatch(self._intent("device.media.control", device=dev, command="stop"), render=lambda r: "Okay.")
        if _NEXT.match(t):
            return FastMatch(self._intent("device.media.control", device=dev, command="next"), render=lambda r: "Okay.")
        if _PREV.match(t):
            return FastMatch(self._intent("device.media.control", device=dev, command="previous"), render=lambda r: "Okay.")

        m = _VOL_SET.match(nt) or _VOL_SET2.match(nt)
        if m:
            n = int(m.group(1))
            return FastMatch(self._intent("device.volume.set", device=dev, percent=n), render=lambda r, n=n: f"Volume {n} percent.")
        if _MUTE.match(t):
            return FastMatch(self._intent("device.volume.set", device=dev, percent=0), render=lambda r: "Muted.")
        m = _VOL_STEP.match(t)
        if m:
            word = next(g for g in m.groups() if g)
            return FastMatch(self._intent("device.volume.step", device=dev, direction="up" if word in ("up", "louder") else "down"),
                             render=lambda r: "Okay.")

        if _TIME.match(t):
            return FastMatch(self._intent("host.time"), render=_say_time)
        if _DATE.match(t):
            return FastMatch(self._intent("host.time"), render=_say_date)

        if "battery" in t and len(t.split()) <= 10 and not any(b in t for b in _BATT_BAD) and _BATT_LEAD.match(t):
            return FastMatch(self._intent("device.battery", device=dev), render=_say_battery)

        if re.search(r"\btimers?\b", t):
            if _TIMER_CANCEL.match(t):
                return FastMatch(self._intent("host.timer.cancel"),
                                 render=lambda r: "No timers were running." if not r.data.get("count") else
                                 f"Cancelled {r.data['count']} timer{'s' if r.data['count'] != 1 else ''}.")
            secs = parse_duration(t)
            if secs is not None and not _TIMER_LIST.search(t):
                return FastMatch(self._intent("host.timer.set", seconds=secs),
                                 render=lambda r, s=secs: f"Timer set for {fmt_duration(s)}.")
            if _TIMER_LIST.search(t):
                return FastMatch(self._intent("host.timer.list"), render=lambda r: _upfirst(r.output) + ".")

        orig = _FILLER.sub("", re.sub(r"\s+", " ", text.strip())).strip()
        orig = re.sub(r"[.!?\s]+$", "", orig)
        m = _NOTE.match(orig)
        if m and 2 <= len(m.group(1).strip()) <= 500:
            return FastMatch(self._intent("host.note.append", text=m.group(1).strip()), render=lambda r: "Noted.")
        return None
