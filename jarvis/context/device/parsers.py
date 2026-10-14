"""Parsers for `dumpsys` output. Stdlib-only.

Formats are modeled on AOSP output (verified here only against fixtures). On the phone, run
`python -m recall.device.capture doctor` to confirm they parse your ROM's live output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..schema import content_hash

FLAG_ONGOING = 0x2
FLAG_FOREGROUND_SERVICE = 0x40
FLAG_GROUP_SUMMARY = 0x200

_HDR = re.compile(r"^\s*NotificationRecord\(0x[0-9a-fA-F]+:?\s*(?P<rest>.*)$")
_EXTRAS_OPEN = re.compile(r"^\s*extras=\{\s*$")
_EXTRA_START = re.compile(r"^\s*([A-Za-z_][\w.]*)=([\w.$\[\]]+)(?: \((.*))?$")
_FLAGS_LINE = re.compile(r"^\s*flags=0x([0-9a-fA-F]+)\s*$")


def _g(rx: str, s: str) -> str | None:
    m = re.search(rx, s)
    return m.group(1) if m else None


@dataclass
class NotifRecord:
    key: str
    pkg: str
    user: int
    nid: int
    tag: str | None
    importance: int | None
    channel: str | None
    category: str | None
    flags: int
    extras: dict[str, str] = field(default_factory=dict)

    def _x(self, name: str) -> str:
        return self.extras.get(name, "")

    @property
    def title(self) -> str:
        return self._x("android.title")

    @property
    def text(self) -> str:
        return self._x("android.text")

    @property
    def big_text(self) -> str:
        return self._x("android.bigText")

    @property
    def sub_text(self) -> str:
        return self._x("android.subText")

    @property
    def conversation(self) -> str:
        return self._x("android.conversationTitle")

    @property
    def ongoing(self) -> bool:
        return bool(self.flags & (FLAG_ONGOING | FLAG_FOREGROUND_SERVICE))

    @property
    def group_summary(self) -> bool:
        return bool(self.flags & FLAG_GROUP_SUMMARY)

    @property
    def chash(self) -> str:
        return content_hash(self.title, self.text, self.big_text, self.sub_text, self.conversation)

    def to_data(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "key": self.key, "id": self.nid, "tag": self.tag, "channel": self.channel,
            "category": self.category, "importance": self.importance, "title": self.title,
            "text": self.text, "big_text": self.big_text, "sub_text": self.sub_text,
            "conversation": self.conversation, "template": self._x("android.template").rsplit(".", 1)[-1],
            "is_group": self._x("android.isGroupConversation") == "true",
        }
        return {k: v for k, v in d.items() if v not in ("", None, False)}


def is_valid_notification_dump(dump: str) -> bool:
    """False for error output ("Can't find service", empty) so a failed poll never looks like 'all cleared'."""
    return "Notification List" in dump or "NotificationRecord(" in dump


def _extras_lines(block: list[str]) -> list[str]:
    for i, ln in enumerate(block):
        if _EXTRAS_OPEN.match(ln):
            indent = len(ln) - len(ln.lstrip())
            out = []
            for ln2 in block[i + 1:]:
                if ln2.strip() == "}" and (len(ln2) - len(ln2.lstrip())) <= indent:
                    break
                out.append(ln2)
            return out
    return []


def _parse_extras(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    key: str | None = None
    buf: str | None = None

    def flush() -> None:
        nonlocal key, buf
        if key is not None and buf is not None:
            out[key] = buf[:-1] if buf.endswith(")") else buf
        key, buf = None, None

    for ln in lines:
        m = _EXTRA_START.match(ln)
        if m and (buf is None or buf.endswith(")")):
            flush()
            k, _typ, rest = m.groups()
            if rest is None:  # "key=null"
                continue
            key, buf = k, rest
        elif buf is not None and not buf.endswith(")"):
            buf += "\n" + ln.rstrip("\r")
    flush()
    return out


def _parse_record(block: list[str]) -> NotifRecord | None:
    m = _HDR.match(block[0])
    if not m:
        return None
    rest = m.group("rest")
    pkg = _g(r"\bpkg=(\S+)", rest)
    if not pkg:
        return None
    pre_key = rest.split(" key=", 1)[0]
    user = int(_g(r"user=UserHandle\{(-?\d+)\}", rest) or 0)
    nid = int(_g(r"\bid=(-?\d+)", pre_key) or 0)
    tag = _g(r"\btag=(\S+)", pre_key)
    tag = None if tag in (None, "null") else tag
    imp = _g(r"importance=(-?\d+)", rest)
    key = _g(r"\bkey=(\S+?):\s+Notification\(", rest) or f"{user}|{pkg}|{nid}|{tag}"
    flags_s = None
    for ln in block[1:40]:
        fm = _FLAGS_LINE.match(ln)
        if fm:
            flags_s = fm.group(1)
            break
    flags_s = flags_s or _g(r"\bflags=0x([0-9a-fA-F]+)", rest) or "0"
    return NotifRecord(
        key=key, pkg=pkg, user=user, nid=nid, tag=tag, importance=int(imp) if imp is not None else None,
        channel=_g(r"Notification\(channel=([^\s)]+)", rest), category=_g(r"\bcategory=([\w.-]+)", rest),
        flags=int(flags_s, 16), extras=_parse_extras(_extras_lines(block)))


def parse_notifications(dump: str) -> list[NotifRecord]:
    lines = dump.splitlines()
    idx = [i for i, ln in enumerate(lines) if _HDR.match(ln)]
    recs = []
    for n, i in enumerate(idx):
        j = idx[n + 1] if n + 1 < len(idx) else len(lines)
        rec = _parse_record(lines[i:j])
        if rec:
            recs.append(rec)
    return recs


@dataclass(frozen=True)
class Focus:
    pkg: str
    activity: str
    user: int


_WIN = re.compile(r"mCurrentFocus=Window\{[0-9a-fA-F]+ u(?P<u>\d+) (?P<pkg>[\w.]+)/(?P<act>[\w.$]+)\}")
_FAPP = re.compile(r"mFocusedApp=\S*\{[0-9a-fA-F]+ u(?P<u>\d+) (?P<pkg>[\w.]+)/(?P<act>[\w.$]+)")
_RES = re.compile(
    r"(?:topResumedActivity|mResumedActivity|ResumedActivity:?)\W+ActivityRecord\{[0-9a-fA-F]+ u(?P<u>\d+) "
    r"(?P<pkg>[\w.]+)/(?P<act>[\w.$]+)")
_WAKE = re.compile(r"mWakefulness=(\w+)")


def parse_focus(text: str) -> Focus | None:
    """Foreground app from window/activity dumps. Windows without a pkg/activity (shade, IME) fall through
    to the focused/resumed activity, i.e. the app underneath."""
    for rx in (_WIN, _FAPP, _RES):
        m = rx.search(text)
        if m:
            pkg, act = m.group("pkg"), m.group("act")
            if act.startswith("."):
                act = pkg + act
            return Focus(pkg, act, int(m.group("u")))
    return None


def parse_screen(text: str) -> str | None:
    """'on' | 'off' | None. Dozing counts as off."""
    m = _WAKE.search(text)
    if not m:
        return None
    return "on" if m.group(1) == "Awake" else "off"
