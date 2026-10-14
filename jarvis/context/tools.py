"""Agent-facing tool layer: one definition feeds the HTTP API, in-process use by Jarvis, and OpenAI-style
function specs. Read-only by design (plus a harmless enrichment pause); destructive operations live behind the
admin token, never in the agent's tool list."""
from __future__ import annotations

import logging
from typing import Any

from . import query as Q
from .pipeline import EnrichWorker, Pipeline
from .query import Clock

log = logging.getLogger("recall.tools")

TIME = ("Time bound: relative ('90m','24h','7d','2w'), preset ('today','yesterday','this_week','last_week',"
        "'this_month'), ISO-8601 or epoch ms. A preset given as `since` also sets the end of its range.")
S = {"type": "string"}


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


TOOLS: list[dict[str, Any]] = [
    {"name": "recall",
     "description": ("Search the user's phone history (notifications, messages, email alerts) by keywords and meaning. "
                     "Returns cited events newest-relevant first; cite `event_id`s when answering. "
                     "Leave `query` empty to list recent events under the filters."),
     "parameters": _obj({"query": {**S, "description": "What to look for."},
                         "since": {**S, "description": TIME}, "until": {**S, "description": TIME},
                         "apps": {"type": "array", "items": S, "description": "App labels or package names."},
                         "entity": {**S, "description": "Restrict to events involving this person/org/place."},
                         "limit": {"type": "integer", "description": "1-50, default 10."}}, ["query"])},
    {"name": "who_is",
     "description": "Entity card for a person/org/place seen in the user's phone: aliases, apps, last contact, related entities.",
     "parameters": _obj({"name": S}, ["name"])},
    {"name": "what_changed",
     "description": "Compare a time window with the equally long window before it: notifications and screen time by app, top senders, new entities.",
     "parameters": _obj({"since": {**S, "description": TIME + " Default 24h."}, "until": {**S, "description": TIME},
                         "app": {**S, "description": "Limit to one app."}})},
    {"name": "timeline",
     "description": "App usage sessions (with dwell time) plus notification counts for a window. Default: today.",
     "parameters": _obj({"since": {**S, "description": TIME}, "until": {**S, "description": TIME}})},
    {"name": "explain",
     "description": "Provenance of one event: raw data, privacy flags, extracted entities and edges, enrichment status.",
     "parameters": _obj({"event_id": S}, ["event_id"])},
    {"name": "context_now",
     "description": ("Live device state (foreground app, screen, online) plus the last notifications and sessions. "
                     "Use as input for permission decisions about device actions."),
     "parameters": _obj({"minutes": {"type": "integer", "description": "Lookback, 1-1440, default 30."}})},
    {"name": "pause_enrichment",
     "description": "Pause background embedding/LLM extraction so the GPU stays free (e.g. during a voice turn).",
     "parameters": _obj({"seconds": {"type": "number", "description": "0-3600, default 60. 0 resumes."}})},
]
SPECS = {t["name"]: t for t in TOOLS}


def openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": t} for t in TOOLS]


class RecallTools:
    def __init__(self, pipeline: Pipeline, clock: Clock, labels: dict[str, str], worker: EnrichWorker | None = None):
        self.p, self.clock, self.labels, self.worker = pipeline, clock, labels, worker

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = SPECS.get(name)
        if spec is None:
            return {"ok": False, "error": f"unknown tool {name!r}", "tools": sorted(SPECS)}
        args = dict(args or {})
        props = spec["parameters"]["properties"]
        if set(args) - set(props):
            return {"ok": False, "error": f"unknown argument(s) {sorted(set(args) - set(props))}; allowed: {sorted(props)}"}
        missing = [r for r in spec["parameters"]["required"] if r not in args]
        if missing:
            return {"ok": False, "error": f"missing required argument(s) {missing}"}
        try:
            return {"ok": True, "result": getattr(self, f"_t_{name}")(**args)}
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)}
        except Exception:
            log.exception("tool %s crashed", name)
            return {"ok": False, "error": "internal error"}

    def _t_recall(self, query: str, since: Any = None, until: Any = None, apps: Any = None, entity: Any = None,
                  limit: Any = 10) -> dict[str, Any]:
        start, end = Q.resolve_range(since, until, None, self.clock)
        if apps is not None and not isinstance(apps, list):
            apps = [str(apps)]
        return Q.search(self.p.store, self.p.embedder, self.p.vindex, str(query), start=start, end=end, apps=apps,
                        entity=entity, limit=int(limit), labels=self.labels, clock=self.clock)

    def _t_who_is(self, name: str) -> dict[str, Any]:
        return Q.who_is(self.p.store, str(name), labels=self.labels, clock=self.clock)

    def _t_what_changed(self, since: Any = "24h", until: Any = None, app: str | None = None) -> dict[str, Any]:
        return Q.what_changed(self.p.store, since=since, until=until, app=app, labels=self.labels, clock=self.clock)

    def _t_timeline(self, since: Any = "today", until: Any = None) -> dict[str, Any]:
        return Q.timeline(self.p.store, since=since, until=until, labels=self.labels, clock=self.clock)

    def _t_explain(self, event_id: str) -> dict[str, Any]:
        return Q.explain(self.p.store, str(event_id), labels=self.labels, clock=self.clock)

    def _t_context_now(self, minutes: Any = 30) -> dict[str, Any]:
        return Q.context_now(self.p.store, minutes=max(1, min(int(minutes), 1440)), labels=self.labels, clock=self.clock)

    def _t_pause_enrichment(self, seconds: Any = 60) -> dict[str, Any]:
        if self.worker is None:
            return {"paused": False, "reason": "no background worker in this process"}
        self.worker.pause(float(seconds))
        return {"paused": float(seconds) > 0, "seconds": max(0.0, min(float(seconds), 3600.0))}
