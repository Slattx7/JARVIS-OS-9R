"""Record-time redaction.

Traces of a root agent with a personal context graph attached are a credential/PII honeypot, so
scrubbing happens *before* anything hits disk. Placeholders are deterministic (HMAC with a local key):
the same secret always maps to the same placeholder, so input hashes computed over redacted forms
stay stable between a live run and its replay. Redaction is idempotent: scrub(scrub(x)) == scrub(x).

Best-effort regex + key-name based. It is a seatbelt, not a vault: keep the store on encrypted storage.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from typing import Iterable, List, Pattern, Tuple, Union

PLACEHOLDER_RE = re.compile(r"⟦[a-z0-9_]+:[0-9a-f]{6}⟧")

_SECRET_KEY_RE = re.compile(
    r"(?i)(?:^|[_\-.])(?:pass(?:word|wd|code)?|pwd|secret|token|api[_\-]?key|apikey|authorization|auth|"
    r"cookie|otp|pin|private[_\-]?key|credentials?|session[_\-]?id)(?:$|[_\-.])"
)
_NUMERIC_SECRET_KEY_RE = re.compile(r"(?i)(?:^|[_\-.])(?:pass(?:word|wd|code)?|pwd|otp|pin|secret)(?:$|[_\-.])")

# values under these keys are identifiers/digests, not secrets: file hashes, commit ids, etags
_EXEMPT_KEY_RE = re.compile(r"(?i)(?:^|[_\-.])(?:sha\d*|md5|hash|digest|checksum|etag|commit|rev)(?:$|[_\-.])")

# (kind, regex, group-to-replace)  group 0 == whole match
_PATTERNS: List[Tuple[str, Pattern[str], int]] = [
    ("secret", re.compile(
        r"(?i)\b(?:password|passwd|pwd|passcode|secret|token|api[_-]?key|apikey|authorization)\b"
        r"\s*[:=]\s*[\"']?([^\s\"',;]+)"), 1),
    ("bearer", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{12,})"), 1),
    ("apikey", re.compile(r"\b(?:sk|pk|rk|ghp|gho|ghs|ghu|glpat|xox[abprs])[-_][A-Za-z0-9_-]{16,}"), 0),
    ("awskey", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), 0),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), 0),
    ("phone", re.compile(r"(?<![\w+])\+\d[\d\s().-]{8,}\d"), 0),
    ("phone", re.compile(r"(?<![\w(])\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)"), 0),
    ("phone", re.compile(r"(?<![\w.])[6-9]\d{9}(?![\w.])"), 0),
    ("hex", re.compile(r"(?<!sha256:)(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])"), 0),
    ("token", re.compile(
        r"(?<![A-Za-z0-9+_-])(?=[A-Za-z0-9+_-]*[A-Z])(?=[A-Za-z0-9+_-]*[a-z])(?=[A-Za-z0-9+_-]*\d)"
        r"[A-Za-z0-9+_-]{40,}(?![A-Za-z0-9+_-])"), 0),
]


class Redactor:
    def __init__(self, key: bytes, *, extra_patterns: Iterable[Tuple[str, Union[str, Pattern[str]], int]] = (),
                 enabled: bool = True):
        self.key = key
        self.enabled = enabled
        self._patterns: List[Tuple[str, Pattern[str], int]] = list(_PATTERNS)
        for kind, pat, grp in extra_patterns:
            self._patterns.append((kind, re.compile(pat) if isinstance(pat, str) else pat, grp))

    def placeholder(self, kind: str, value: Union[str, bytes]) -> str:
        data = bytes(value) if isinstance(value, (bytes, bytearray)) else str(value).encode("utf-8", "replace")
        d = hmac.new(self.key, kind.encode() + b"\0" + data, hashlib.sha256).hexdigest()[:6]
        return f"⟦{kind}:{d}⟧"

    @staticmethod
    def is_placeholder(s: object) -> bool:
        return isinstance(s, str) and PLACEHOLDER_RE.fullmatch(s) is not None

    def is_secret_key(self, k: str) -> bool:
        return self.enabled and bool(_SECRET_KEY_RE.search(k))

    def is_exempt_key(self, k: str) -> bool:
        return self.enabled and bool(_EXEMPT_KEY_RE.search(k)) and not _SECRET_KEY_RE.search(k)

    def is_numeric_secret_key(self, k: str) -> bool:
        return self.enabled and bool(_NUMERIC_SECRET_KEY_RE.search(k))

    def _sub(self, kind: str, rx: Pattern[str], grp: int, s: str) -> str:
        def rep(m: "re.Match[str]") -> str:
            val = m.group(grp)
            if val is None or PLACEHOLDER_RE.fullmatch(val):
                return m.group(0)
            ph = self.placeholder(kind, val)
            if grp == 0:
                return ph
            whole = m.group(0)
            a, b = m.span(grp)
            s0 = m.start(0)
            return whole[: a - s0] + ph + whole[b - s0:]

        return rx.sub(rep, s)

    def scrub_str(self, s: str) -> str:
        if not self.enabled or len(s) < 6:
            return s
        for kind, rx, grp in self._patterns:
            s = self._sub(kind, rx, grp, s)
        return s
