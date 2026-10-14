"""Entity/relation extraction.

Deterministic rules always run. An optional local LLM adds person/org/place/task mentions. LLM output is never
trusted: every entity and relation must cite an `evidence` snippet that is a literal substring of the event text,
and names must appear inside their evidence, or the item is dropped. That is the guard against hallucinated edges.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import BLOCKED_APP, event_text

APP_LABELS = {
    "com.whatsapp": "WhatsApp", "com.whatsapp.w4b": "WhatsApp Business", "org.telegram.messenger": "Telegram",
    "org.thoughtcrime.securesms": "Signal", "com.google.android.gm": "Gmail", "com.google.android.apps.messaging": "Messages",
    "com.google.android.dialer": "Phone", "com.slack": "Slack", "com.discord": "Discord", "com.instagram.android": "Instagram",
    "com.linkedin.android": "LinkedIn", "com.microsoft.teams": "Teams", "com.android.chrome": "Chrome",
    "com.google.android.youtube": "YouTube", "com.spotify.music": "Spotify", "com.twitter.android": "X",
    "com.reddit.frontpage": "Reddit", "com.google.android.calendar": "Calendar", "com.google.android.apps.maps": "Maps",
    "com.android.launcher3": "Launcher", "org.lineageos.trebuchet": "Launcher", "android": "System",
}
ENTITY_TYPES = ("app", "person", "thread", "org", "place", "task", "phone", "email", "domain")
REL_TYPES = ("VIA", "IN", "HAS_HANDLE", "MENTIONS", "ASKS_TO", "WORKS_AT", "AT", "INVITES_TO")
LLM_ENTITY_TYPES = ("person", "org", "place", "task")
LLM_REL_TYPES = ("ASKS_TO", "WORKS_AT", "AT", "INVITES_TO", "MENTIONS")
PERSON_CATS = ("msg", "email", "call", "missed_call")


def make_labels(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {k.lower(): v for k, v in {**APP_LABELS, **(extra or {})}.items()}


def app_label(pkg: str, labels: dict[str, str]) -> str:
    if pkg.lower() in labels:
        return labels[pkg.lower()]
    parts = [p for p in pkg.split(".") if p]
    generic = {"android", "app", "mobile", "main", "client", "free", "lite", "prod"}
    for p in reversed(parts):
        if p.lower() not in generic:
            return p[:1].upper() + p[1:]
    return pkg


@dataclass(frozen=True)
class Ent:
    type: str
    name: str
    role: str
    evidence: str = ""
    source: str = "det"


@dataclass(frozen=True)
class Rel:
    src: tuple[str, str]
    rel: str
    dst: tuple[str, str]
    evidence: str = ""
    source: str = "det"


@dataclass
class Extraction:
    entities: list[Ent] = field(default_factory=list)
    relations: list[Rel] = field(default_factory=list)


_COUNT_SUFFIX = re.compile(r"\s*\(\d+\s+(?:new\s+)?messages?\)\s*$", re.I)
_JUNK_TITLE = re.compile(r"^(?:\d+\s+(?:new\s+)?(?:messages?|chats?)|new messages?|you|me|messages?|whatsapp|signal|telegram|sms|unknown)$", re.I)
_SENDER_PREFIX = re.compile(r"^(?P<s>[^:\n]{1,40}):\s+\S")
_SENDER_ID = re.compile(r"^[A-Z]{2}-[A-Z0-9]{3,9}$")  # DLT-style SMS sender ids, e.g. AX-HDFCBK
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_HOST = re.compile(r"https?://([^/\s:?#]+)")
_PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}[\s-]?\d{2,5}[\s-]?\d{3,5}(?:[\s-]?\d{1,5})?(?!\d)")
_PHONE_10 = re.compile(r"(?<![\d+])\d{5}[\s-]?\d{5}(?!\d)")


def _classify_sender(name: str, app: str, labels: dict[str, str]) -> str | None:
    n = name.strip()
    if not (1 <= len(n) <= 60) or _JUNK_TITLE.match(n) or n.startswith("#"):
        return None  # junk titles and channel names (#deploys) are not people
    if n.casefold() in (app.casefold(), app_label(app, labels).casefold()):
        return None
    compact = re.sub(r"[\s\-+().]", "", n)
    if compact.isdigit() and len(compact) < 7:
        return None  # short code
    return "org" if _SENDER_ID.match(n) else "person"


def deterministic(ev: dict[str, Any], labels: dict[str, str]) -> Extraction:
    if ev["type"] != "notification" or ev["app"] == BLOCKED_APP:
        return Extraction()
    app, d = ev["app"], ev["data"]
    ents: list[Ent] = [Ent("app", app, "app")]
    if "meta-only" in (ev.get("flags") or []):
        return Extraction(ents, [])
    rels: list[Rel] = []
    title = _COUNT_SUFFIX.sub("", str(d.get("title") or "")).strip()
    text = str(d.get("big_text") or d.get("text") or "")
    conv = str(d.get("conversation") or "").strip()
    is_group = bool(d.get("is_group")) or bool(conv)
    sender: str | None = None
    thread: str | None = None
    if d.get("category") in PERSON_CATS or "MessagingStyle" in str(d.get("template", "")):
        if is_group:
            thread = conv or title
            m = _SENDER_PREFIX.match(text)
            sender = m.group("s").strip() if m else None
        else:
            sender = title
    stype = _classify_sender(sender, app, labels) if sender else None
    ev_snip = (title or text)[:120]
    if stype:
        ents.append(Ent(stype, sender, "sender", ev_snip))  # type: ignore[arg-type]
        rels.append(Rel((stype, sender), "VIA", ("app", app), ev_snip))  # type: ignore[arg-type]
    if thread and 1 <= len(thread) <= 80 and not _JUNK_TITLE.match(thread):
        ents.append(Ent("thread", thread, "thread", thread[:120]))
        rels.append(Rel(("thread", thread), "VIA", ("app", app), thread[:120]))
        if stype:
            rels.append(Rel((stype, sender), "IN", ("thread", thread), ev_snip))  # type: ignore[arg-type]
    seen: set[tuple[str, str]] = set()

    def mention(etype: str, name: str, snippet: str) -> None:
        if (etype, name) in seen:
            return
        seen.add((etype, name))
        ents.append(Ent(etype, name, "mention", snippet[:120]))
        if stype:
            rels.append(Rel((stype, sender), "MENTIONS", (etype, name), snippet[:120]))  # type: ignore[arg-type]

    for m in _EMAIL.finditer(text):
        mention("email", m.group(0).lower(), m.group(0))
    for m in _URL_HOST.finditer(text):
        mention("domain", m.group(1).lower(), m.group(0))
    for rx in (_PHONE_INTL, _PHONE_10):
        for m in rx.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if 10 <= len(digits) <= 14:
                mention("phone", digits[-10:], m.group(0).strip())
    return Extraction(ents, rels)


# ---- optional LLM extraction --------------------------------------------------------------------------

LLM_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["entities", "relations"],
    "properties": {
        "entities": {"type": "array", "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False, "required": ["type", "name", "evidence"],
            "properties": {"type": {"enum": list(LLM_ENTITY_TYPES)}, "name": {"type": "string", "maxLength": 80},
                           "evidence": {"type": "string", "maxLength": 200}}}},
        "relations": {"type": "array", "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False, "required": ["src", "rel", "dst", "evidence"],
            "properties": {"src": {"type": "string", "maxLength": 80}, "rel": {"enum": list(LLM_REL_TYPES)},
                           "dst": {"type": "string", "maxLength": 80}, "evidence": {"type": "string", "maxLength": 200}}}},
    },
}
SYSTEM_PROMPT = (
    "You extract structured facts from ONE phone notification or message. Reply with JSON only.\n"
    "Rules: (1) Include only things literally present in the text. (2) Every entity and relation needs 'evidence': "
    "an EXACT substring copied from the text. (3) Entity types: person, org, place, task. A task is something someone "
    "asks or plans to do. (4) Relation types: ASKS_TO (person -> task), WORKS_AT (person -> org), AT (person|task -> place), "
    "INVITES_TO (person -> place|org|task), MENTIONS (person -> any). (5) src and dst must be entity names you listed. "
    "(6) If unsure or nothing qualifies, return empty lists. Do not guess."
)


def _norm_ws(s: str) -> str:
    return " ".join(s.casefold().split())


def _loads_lenient(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        pass
    try:
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (TypeError, ValueError):
        return None


def validate_llm_output(raw: Any, text: str) -> Extraction:
    obj = raw if isinstance(raw, dict) else _loads_lenient(raw)
    if not isinstance(obj, dict):
        return Extraction()
    tn = _norm_ws(text)
    ents: list[Ent] = []
    by_name: dict[str, tuple[str, str]] = {}
    raw_ents, raw_rels = obj.get("entities"), obj.get("relations")
    raw_ents = raw_ents if isinstance(raw_ents, list) else []
    raw_rels = raw_rels if isinstance(raw_rels, list) else []
    for e in raw_ents[:8]:
        if not isinstance(e, dict):
            continue
        t, name, evid = e.get("type"), str(e.get("name") or "").strip(), str(e.get("evidence") or "").strip()
        if t not in LLM_ENTITY_TYPES or not (0 < len(name) <= 80):
            continue
        en = _norm_ws(evid)
        if not en or en not in tn:
            continue  # evidence must be literally in the source text
        if t != "task" and _norm_ws(name) not in en:
            continue  # names must be grounded in their evidence (tasks are paraphrases)
        ents.append(Ent(t, name, "mention", evid[:200], "llm"))
        by_name[_norm_ws(name)] = (t, name)
    rels: list[Rel] = []
    for r in raw_rels[:8]:
        if not isinstance(r, dict) or r.get("rel") not in LLM_REL_TYPES:
            continue
        evid = str(r.get("evidence") or "").strip()
        s, d = by_name.get(_norm_ws(str(r.get("src") or ""))), by_name.get(_norm_ws(str(r.get("dst") or "")))
        if s and d and s != d and _norm_ws(evid) in tn and evid:
            rels.append(Rel(s, r["rel"], d, evid[:200], "llm"))
    return Extraction(ents, rels)


class LLMExtractor:
    def __init__(self, base_url: str, model: str, *, mode: str = "json_schema", timeout: float = 60.0,
                 min_chars: int = 24, categories: tuple[str, ...] = ("msg", "email"), min_interval_s: float = 2.0,
                 labels: dict[str, str] | None = None, call: Callable[[list[dict[str, str]]], str] | None = None):
        self.base_url, self.model, self.mode, self.timeout = base_url.rstrip("/"), model, mode, timeout
        self.min_chars, self.categories, self.min_interval_s = min_chars, tuple(categories), min_interval_s
        self.labels = labels or make_labels()
        self._call = call or self._http_call
        self._last = 0.0

    def eligible(self, ev: dict[str, Any]) -> bool:
        return (len(ev.get("text") or "") >= self.min_chars and "meta-only" not in (ev.get("flags") or [])
                and ev["data"].get("category") in self.categories)

    def _http_call(self, messages: list[dict[str, str]]) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 400}
        if self.mode == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "extraction", "strict": True, "schema": LLM_SCHEMA}}
        elif self.mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def extract(self, ev: dict[str, Any]) -> Extraction:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)  # low priority: the same GPU serves the voice loop
        self._last = time.monotonic()
        text = ev.get("text") or event_text(ev["type"], ev["data"]) or ""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"App: {app_label(ev['app'], self.labels)}\nText: {text}"}]
        return validate_llm_output(self._call(msgs), text)
