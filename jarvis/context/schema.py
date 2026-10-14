"""Event schema shared by device capture and server ingest. Stdlib-only (runs under Termux)."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1
EVENT_TYPES = ("notification", "app_switch", "screen", "heartbeat")
BLOCKED_APP = "redacted.blocked"
TEXT_KEYS = ("title", "text", "big_text", "sub_text", "conversation")
MAX_DATA_BYTES = 16 * 1024
MAX_TEXT_CHARS = 4096
TS_MIN = 1_420_070_400_000  # 2015-01-01
TS_FUTURE_SLACK = 24 * 3600 * 1000

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_DEVICE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_APP_RE = re.compile(r"^[A-Za-z0-9_.$-]{1,200}$")


def now_ms() -> int:
    return int(time.time() * 1000)


def content_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(("" if p is None else str(p)).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def make_event_id(device: str, etype: str, key: str, ts: int, chash: str = "") -> str:
    raw = f"{device}|{etype}|{key}|{ts}|{chash}".encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:32]


@dataclass
class Event:
    id: str
    device: str
    type: str
    ts: int
    app: str
    data: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_event(d: Any, now: int | None = None) -> dict[str, Any]:
    """Return a normalized event dict or raise ValueError(reason)."""
    if not isinstance(d, dict):
        raise ValueError("event must be an object")
    now = now_ms() if now is None else now
    eid, dev, etype, app, ts = d.get("id"), d.get("device"), d.get("type"), d.get("app"), d.get("ts")
    if not isinstance(eid, str) or not _ID_RE.match(eid):
        raise ValueError("bad id")
    if not isinstance(dev, str) or not _DEVICE_RE.match(dev):
        raise ValueError("bad device")
    if etype not in EVENT_TYPES:
        raise ValueError(f"bad type: {etype!r}")
    if isinstance(ts, bool) or not isinstance(ts, int):
        raise ValueError("ts must be int epoch ms")
    if not (TS_MIN <= ts <= now + TS_FUTURE_SLACK):
        raise ValueError("ts out of range")
    if not isinstance(app, str) or not _APP_RE.match(app):
        raise ValueError("bad app")
    data = d.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    flags = d.get("flags", [])
    if not isinstance(flags, list) or len(flags) > 16 or not all(isinstance(f, str) and len(f) <= 32 for f in flags):
        raise ValueError("bad flags")
    data = dict(data)
    for k in TEXT_KEYS:
        if isinstance(data.get(k), str) and len(data[k]) > MAX_TEXT_CHARS:
            data[k] = data[k][:MAX_TEXT_CHARS]
    try:
        size = len(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        raise ValueError("data not JSON-serializable") from None
    if size > MAX_DATA_BYTES:
        raise ValueError("data too large")
    return {"id": eid, "device": dev, "type": etype, "ts": ts, "app": app, "data": data,
            "flags": list(flags), "v": SCHEMA_VERSION}


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def event_text(etype: str, data: dict[str, Any]) -> str | None:
    """Indexable text for an event (notifications only)."""
    if etype != "notification":
        return None
    title, body, conv = _s(data.get("title")), _s(data.get("big_text") or data.get("text")), _s(data.get("conversation"))
    parts = []
    if conv and conv != title:
        parts.append(f"[{conv}]")
    if title and body:
        parts.append(f"{title}: {body}")
    elif title or body:
        parts.append(title or body)
    text = " ".join(parts).strip()
    return text[:MAX_TEXT_CHARS] or None
