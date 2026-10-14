"""Streaming sentence chunker for TTS: emit early, never split mid-abbreviation/decimal."""
from __future__ import annotations

import re

ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "a.m", "p.m",
          "no", "inc", "ltd", "fig", "approx", "mt", "u.s", "u.k"}
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_`#>~|]")
_BULLET = re.compile(r"(?m)^\s*(?:[-•]|\d+[.)])\s+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
_WS = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    t = _URL.sub("link", text)
    t = _BULLET.sub("", t)
    t = _EMOJI.sub("", t)
    t = _MD.sub("", t)
    t = t.replace("&", " and ")
    t = _WS.sub(" ", t).strip()
    return t if any(c.isalnum() for c in t) else ""


class SentenceChunker:
    def __init__(self, first_min: int = 12, min_chars: int = 40, first_clause_min: int = 28,
                 clause_min: int = 70, max_chars: int = 240):
        self.first_min, self.min_chars = first_min, min_chars
        self.first_clause_min, self.clause_min, self.max_chars = first_clause_min, clause_min, max_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        self._buf += text
        return self._extract(final=False)

    def flush(self) -> list[str]:
        return self._extract(final=True)

    def _extract(self, final: bool) -> list[str]:
        out: list[str] = []
        while self._buf:
            cut = self._find_cut(final)
            if cut is None:
                break
            seg, self._buf = self._buf[:cut], self._buf[cut:].lstrip(" \t")
            if self._buf.startswith("\n"):
                self._buf = self._buf.lstrip()
            cleaned = clean_for_speech(seg)
            if cleaned:
                out.append(cleaned)
                self._emitted += 1
        if final:
            self._buf = ""
        return out

    def _is_abbrev(self, buf: str, i: int) -> bool:
        k = i
        while k > 0 and not buf[k - 1].isspace():
            k -= 1
        word = buf[k:i].lower().strip("\"'([")
        return word in ABBREV or (len(word) == 1 and word.isalpha())

    def _find_cut(self, final: bool) -> int | None:
        buf, n = self._buf, len(self._buf)
        first = self._emitted == 0
        minc = self.first_min if first else self.min_chars
        clause_min = self.first_clause_min if first else self.clause_min
        i = 0
        while i < n:
            ch = buf[i]
            if ch in ".!?…\n":
                j = i + 1
                while j < n and buf[j] in ".!?…\"')”’]":
                    j += 1
                if ch == "\n":
                    ok = True
                elif j >= n:
                    ok = final
                else:
                    ok = buf[j].isspace()
                if ok and ch == "." and self._is_abbrev(buf, i):
                    ok = False
                if ok and (ch == "\n" or len(buf[:j].strip()) >= minc):
                    return j
                i = j
                continue
            if ch in ",;:—" and i + 1 < n and buf[i + 1].isspace() and len(buf[: i + 1].strip()) >= clause_min:
                return i + 1
            i += 1
        if n > self.max_chars:
            sp = buf.rfind(" ", 0, self.max_chars)
            return sp + 1 if sp > 0 else self.max_chars
        if final and buf.strip():
            return n
        return None
