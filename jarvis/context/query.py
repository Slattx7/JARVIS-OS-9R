"""Read side: hybrid retrieval plus the agent-facing views (recall, who_is, timeline, what_changed, context_now,
explain). Every returned fact carries event ids so an agent can cite and verify it."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from . import extract, resolve
from .schema import BLOCKED_APP, now_ms
from .store import Store, VectorIndex

DAY = 86_400_000
MAX_SESSION_MS = 4 * 3600 * 1000
ONLINE_WINDOW_MS = 15 * 60 * 1000
STOP = frozenset(
    "a an the and or of to in on at for is are was were be been i you he she it we they my me your our this that "
    "with from by as do did does who what when where which how about say said tell".split())
PRESETS = ("today", "yesterday", "this_week", "last_week", "this_month")
_REL = re.compile(r"^(\d+)\s*([smhdw])$")
_UNIT = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": DAY, "w": 7 * DAY}


class Clock:
    def __init__(self, tz: str | None = "local"):
        self.tz = None if tz in (None, "", "local") else ZoneInfo(tz)

    def dt(self, ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, self.tz)

    def midnight(self, d: date) -> int:
        return int(datetime(d.year, d.month, d.day, tzinfo=self.tz).timestamp() * 1000)

    def day_start(self, ms: int) -> int:
        return self.midnight(self.dt(ms).date())

    def next_day(self, day_start_ms: int) -> int:
        return self.midnight(self.dt(day_start_ms).date() + timedelta(days=1))

    def week_start(self, ms: int) -> int:
        d = self.dt(ms).date()
        return self.midnight(d - timedelta(days=d.weekday()))

    def month_start(self, ms: int) -> int:
        return self.midnight(self.dt(ms).date().replace(day=1))

    def day_str(self, ms: int) -> str:
        return self.dt(ms).strftime("%Y-%m-%d")

    def iso(self, ms: int) -> str:
        d = self.dt(ms)
        return (d.astimezone() if self.tz is None else d).isoformat(timespec="seconds")


def preset_range(name: str, now: int, clock: Clock) -> tuple[int, int]:
    today, d = clock.day_start(now), clock.dt(now).date()
    if name == "today":
        return today, now
    if name == "yesterday":
        return clock.midnight(d - timedelta(days=1)), today
    if name == "this_week":
        return clock.week_start(now), now
    if name == "last_week":
        return clock.midnight(d - timedelta(days=d.weekday() + 7)), clock.week_start(now)
    if name == "this_month":
        return clock.month_start(now), now
    raise ValueError(f"unknown preset {name!r}")


def _point(v: Any, now: int, clock: Clock) -> int | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        raise ValueError("bad time value")
    if isinstance(v, (int, float)):
        return int(v if v > 1e11 else v * 1000)
    s = str(v).strip()
    low = s.lower()
    if low == "now":
        return now
    if low in PRESETS:
        return preset_range(low, now, clock)[0]  # as a bound: where the preset starts
    m = _REL.match(low)
    if m:
        return now - int(m.group(1)) * _UNIT[m.group(2)]
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"unrecognized time {v!r}; use e.g. '7d', 'yesterday', 'this_week', ISO-8601 or epoch ms") from None
    if dt.tzinfo is None and clock.tz is not None:
        dt = dt.replace(tzinfo=clock.tz)
    return int(dt.timestamp() * 1000)


def resolve_range(since: Any, until: Any, now: int | None, clock: Clock) -> tuple[int | None, int | None]:
    now = now_ms() if now is None else now
    if isinstance(since, str) and since.strip().lower() in PRESETS:
        start, end = preset_range(since.strip().lower(), now, clock)
        if until not in (None, ""):
            end = _point(until, now, clock) or end
        return start, end
    return _point(since, now, clock), _point(until, now, clock)


def resolve_apps(apps: Iterable[str], labels: dict[str, str]) -> list[str]:
    """Accept display labels ("WhatsApp") or package names; return package names."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for pkg, lab in labels.items():
        by_label[lab.casefold()].append(pkg)
    out: list[str] = []
    for a in apps:
        a = a.strip()
        out.extend(by_label.get(a.casefold(), []))
        if "." in a or a.casefold() not in by_label:
            out.append(a)
    return list(dict.fromkeys(out))


def fts_query(q: str) -> str | None:
    seen: set[str] = set()
    parts: list[str] = []
    for t in re.findall(r"\w+", q.casefold()):
        if t in STOP or t in seen:
            continue
        seen.add(t)
        parts.append(f'"{t}"*' if len(t) >= 3 else f'"{t}"')
    return " OR ".join(parts) or None


def _label(app: str, labels: dict[str, str]) -> str:
    return "Private app" if app == BLOCKED_APP else extract.app_label(app, labels)


def _entity_names(store: Store, seqs: list[int]) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = defaultdict(list)
    for i in range(0, len(seqs), 500):
        ch = seqs[i:i + 500]
        for r in store.q(f"SELECT ee.event_seq s, en.type t, en.name n, ee.role r FROM event_entities ee "
                         f"JOIN entities en ON en.id=ee.entity_id WHERE ee.event_seq IN ({','.join('?' * len(ch))}) "
                         f"AND en.type!='app'", ch):
            out[r["s"]].append({"type": r["t"], "name": r["n"], "role": r["r"]})
    return out


def _hits(store: Store, seqs: list[int], scores: dict[int, float], via: dict[int, dict[str, Any]],
          labels: dict[str, str], clock: Clock) -> list[dict[str, Any]]:
    if not seqs:
        return []
    rows = {r["seq"]: r for r in store.q(
        f"SELECT seq,id,ts,app,type,text FROM events WHERE seq IN ({','.join('?' * len(seqs))})", seqs)}
    ents = _entity_names(store, seqs)
    return [{"event_id": rows[s]["id"], "ts": rows[s]["ts"], "time": clock.iso(rows[s]["ts"]), "app": rows[s]["app"],
             "app_label": _label(rows[s]["app"], labels), "type": rows[s]["type"], "text": rows[s]["text"],
             "score": round(scores.get(s, 0.0), 5), "via": via.get(s, {}), "entities": ents.get(s, [])}
            for s in seqs if s in rows]


def _pick_entities(store: Store, entity: Any) -> list[int]:
    if isinstance(entity, int) or (isinstance(entity, str) and entity.strip().isdigit()):
        return [int(entity)]
    cands = resolve.lookup(store, str(entity), limit=5)
    exact = [c["id"] for c in cands if c["score"] >= 0.999]
    if exact:
        return exact
    near = [c for c in cands if c["score"] >= resolve.CANDIDATE]
    return [near[0]["id"]] if len(near) == 1 else ([c["id"] for c in near] if near else [])


def search(store: Store, embedder: Any, vindex: VectorIndex, q: str, *, start: int | None = None, end: int | None = None,
           apps: list[str] | None = None, entity: Any = None, limit: int = 10, labels: dict[str, str], clock: Clock,
           now: int | None = None, pool: int = 60, half_life_days: float = 30.0) -> dict[str, Any]:
    now = now_ms() if now is None else now
    conds, args = ["e.purged=0", "e.text IS NOT NULL"], []
    filtered = False
    if start is not None:
        conds.append("e.ts>=?")
        args.append(start)
        filtered = True
    if end is not None:
        conds.append("e.ts<?")
        args.append(end)
        filtered = True
    if apps:
        pk = resolve_apps(apps, labels)
        conds.append(f"e.app IN ({','.join('?' * len(pk))})")
        args += pk
        filtered = True
    note = None
    if entity not in (None, ""):
        ids = _pick_entities(store, entity)
        if not ids:
            return {"query": q, "hits": [], "citations": [], "note": f"no entity matches {entity!r}"}
        conds.append(f"e.seq IN (SELECT event_seq FROM event_entities WHERE entity_id IN ({','.join('?' * len(ids))}))")
        args += ids
        filtered = True
    where = " AND ".join(conds)
    limit = max(1, min(int(limit), 50))

    if not q.strip():  # browse mode: newest first under the filters
        seqs = [r["seq"] for r in store.q(f"SELECT e.seq FROM events e WHERE {where} ORDER BY e.ts DESC LIMIT ?", (*args, limit))]
        hits = _hits(store, seqs, {}, {}, labels, clock)
        return {"query": q, "hits": hits, "citations": [h["event_id"] for h in hits], "note": note}

    scores: dict[int, float] = defaultdict(float)
    via: dict[int, dict[str, Any]] = defaultdict(dict)
    match = fts_query(q)
    if match:
        rows = store.q(f"SELECT e.seq FROM events_fts JOIN events e ON e.seq=events_fts.rowid "
                       f"WHERE events_fts MATCH ? AND {where} ORDER BY bm25(events_fts) LIMIT ?", (match, *args, pool))
        for rank, r in enumerate(rows):
            scores[r["seq"]] += 1 / (60 + rank + 1)
            via[r["seq"]]["fts_rank"] = rank + 1
    if embedder is not None:
        allowed = {r["seq"] for r in store.q(f"SELECT e.seq FROM events e WHERE {where}", args)} if filtered else None
        qv = embedder.embed([q])[0]
        for rank, (seq, sc) in enumerate(vindex.search(qv, pool, allowed, embedder.name, embedder.min_score)):
            scores[seq] += 1 / (60 + rank + 1)
            via[seq]["vec_cos"] = round(sc, 3)
    if scores:  # mild recency prior, small enough that relevance dominates
        ts = {r["seq"]: r["ts"] for r in store.q(f"SELECT seq,ts FROM events WHERE seq IN ({','.join('?' * len(scores))})", list(scores))}
        for s in scores:
            scores[s] += 0.004 * 0.5 ** (max(0, now - ts.get(s, now)) / DAY / half_life_days)
    seqs = sorted(scores, key=lambda s: -scores[s])[:limit]
    hits = _hits(store, seqs, scores, via, labels, clock)
    return {"query": q, "hits": hits, "citations": [h["event_id"] for h in hits], "note": note}


# ---- sessions / timeline ------------------------------------------------------------------------------

def derive_sessions(rows: list[Any], last_seen: int | None, window_end: int, now: int,
                    cap: int = MAX_SESSION_MS) -> list[tuple[str, int, int, bool]]:
    out: list[tuple[str, int, int, bool]] = []
    cur: tuple[str, int] | None = None

    def close(end: int) -> None:
        nonlocal cur
        if cur and end > cur[1]:
            out.append((cur[0], cur[1], min(end, cur[1] + cap), False))
        cur = None

    for r in rows:
        if r["type"] == "screen":
            if json.loads(r["data"]).get("state") == "off":
                close(r["ts"])
        elif r["type"] == "app_switch":
            if cur and cur[0] == r["app"]:
                continue
            close(r["ts"])
            cur = (r["app"], r["ts"])
    if cur:
        known_until = max(cur[1], min(window_end, last_seen if last_seen else cur[1]))
        live = window_end >= now - 60_000 and now - known_until < 360_000  # only a live window can be "ongoing"
        out.append((cur[0], cur[1], min(known_until, cur[1] + cap), live))
    return out


def sessions_in(store: Store, start: int, end: int, labels: dict[str, str], now: int | None = None) -> list[dict[str, Any]]:
    now = now_ms() if now is None else now
    rows = store.q("SELECT device,type,ts,app,data FROM events WHERE type IN ('app_switch','screen') "
                   "AND ts>=? AND ts<? ORDER BY device, ts", (start - MAX_SESSION_MS, end))
    seen = {r["device"]: r["last_seen"] for r in store.q("SELECT device,last_seen FROM devices")}
    by_dev: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        by_dev[r["device"]].append(r)
    out = []
    for dev, rs in by_dev.items():
        for app, s, e, ongoing in derive_sessions(rs, seen.get(dev), end, now):
            s2, e2 = max(s, start), min(e, end)
            if e2 > s2:
                out.append({"device": dev, "app": app, "app_label": _label(app, labels), "start": s2, "end": e2,
                            "dur_s": round((e2 - s2) / 1000), "ongoing": ongoing})
    out.sort(key=lambda x: x["start"])
    return out


def _dwell(sessions: list[dict[str, Any]]) -> dict[str, int]:
    d: dict[str, int] = defaultdict(int)
    for s in sessions:
        d[s["app_label"]] += s["dur_s"]
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


def _window_stats(store: Store, start: int, end: int, labels: dict[str, str], clock: Clock,
                  now: int | None = None) -> dict[str, Any]:
    notif = {_label(r["app"], labels): r["n"] for r in store.q(
        "SELECT app, COUNT(*) n FROM events WHERE type='notification' AND ts>=? AND ts<? GROUP BY app ORDER BY n DESC", (start, end))}
    senders = [[r["name"], r["n"]] for r in store.q(
        "SELECT en.name, COUNT(DISTINCT e.seq) n FROM event_entities ee JOIN entities en ON en.id=ee.entity_id "
        "JOIN events e ON e.seq=ee.event_seq WHERE ee.role='sender' AND e.ts>=? AND e.ts<? "
        "GROUP BY en.id ORDER BY n DESC, en.name LIMIT 8", (start, end))]
    sessions = sessions_in(store, start, end, labels, now)
    return {"notifications_by_app": notif, "top_senders": senders, "dwell_s_by_app": _dwell(sessions),
            "sessions": sessions}


def _range_info(start: int, end: int, clock: Clock) -> dict[str, Any]:
    return {"start": start, "end": end, "start_time": clock.iso(start), "end_time": clock.iso(end)}


def timeline(store: Store, *, since: Any = "today", until: Any = None, labels: dict[str, str], clock: Clock,
             now: int | None = None, limit: int = 200) -> dict[str, Any]:
    now = now_ms() if now is None else now
    start, end = resolve_range(since, until, now, clock)
    end = end or now
    start = start if start is not None else end - DAY
    st = _window_stats(store, start, end, labels, clock, now)
    sessions = st.pop("sessions")
    return {"range": _range_info(start, end, clock), "sessions": sessions[:limit], "session_count": len(sessions), **st}


def what_changed(store: Store, *, since: Any = "24h", until: Any = None, app: str | None = None,
                 labels: dict[str, str], clock: Clock, now: int | None = None) -> dict[str, Any]:
    now = now_ms() if now is None else now
    start, end = resolve_range(since, until, now, clock)
    end = end or now
    start = start if start is not None else end - DAY
    span = max(end - start, 1)
    cur = _window_stats(store, start, end, labels, clock, now)
    prev = _window_stats(store, start - span, start, labels, clock, now)
    if app:
        keep = {_label(p, labels) for p in resolve_apps([app], labels)} | {app}
        for st in (cur, prev):
            st["notifications_by_app"] = {k: v for k, v in st["notifications_by_app"].items() if k in keep}
            st["dwell_s_by_app"] = {k: v for k, v in st["dwell_s_by_app"].items() if k in keep}
    apps = set(cur["notifications_by_app"]) | set(prev["notifications_by_app"]) | set(cur["dwell_s_by_app"]) | set(prev["dwell_s_by_app"])
    deltas = {}
    for a in apps:
        n0, n1 = cur["notifications_by_app"].get(a, 0), prev["notifications_by_app"].get(a, 0)
        d0, d1 = cur["dwell_s_by_app"].get(a, 0), prev["dwell_s_by_app"].get(a, 0)
        deltas[a] = {"notifications": n0, "notifications_delta": n0 - n1, "dwell_s": d0, "dwell_s_delta": d0 - d1}
    new_entities = [{"type": r["type"], "name": r["name"], "mentions": r["mentions"]} for r in store.q(
        "SELECT type,name,mentions FROM entities WHERE created_ts>=? AND created_ts<? AND type IN ('person','org','place','domain') "
        "ORDER BY mentions DESC, name LIMIT 10", (start, end))]
    return {"range": _range_info(start, end, clock), "compared_to": _range_info(start - span, start, clock),
            "by_app": dict(sorted(deltas.items(), key=lambda kv: -(abs(kv[1]["notifications_delta"]) + abs(kv[1]["dwell_s_delta"]) / 60))),
            "top_senders": cur["top_senders"], "new_entities": new_entities}


def context_now(store: Store, *, minutes: int = 30, labels: dict[str, str], clock: Clock, now: int | None = None) -> dict[str, Any]:
    now = now_ms() if now is None else now
    since = now - minutes * 60_000
    devices = [{"device": d["device"], "screen": d["screen"], "foreground": _label(d["fg"], labels) if d["fg"] else None,
                "foreground_package": d["fg"], "last_seen_age_s": max(0, round((now - d["last_seen"]) / 1000)),
                "online": now - d["last_seen"] < ONLINE_WINDOW_MS}
               for d in store.q("SELECT * FROM devices ORDER BY last_seen DESC")]
    rows = store.q("SELECT seq FROM events WHERE type='notification' AND purged=0 AND text IS NOT NULL AND ts>=? "
                   "ORDER BY ts DESC LIMIT 10", (since,))
    return {"now": clock.iso(now), "devices": devices,
            "recent_notifications": _hits(store, [r["seq"] for r in rows], {}, {}, labels, clock),
            "recent_sessions": sessions_in(store, since, now, labels, now)}


def who_is(store: Store, name: str, *, labels: dict[str, str], clock: Clock) -> dict[str, Any]:
    cands = resolve.lookup(store, name, limit=5)
    if not cands:
        return {"query": name, "found": False}
    best = cands[0]
    same = [c for c in cands if c["type"] == best["type"] and c["score"] >= resolve.CANDIDATE]
    eid = best["id"]
    e = store.one("SELECT * FROM entities WHERE id=?", (eid,))
    apps = store.q("SELECT e.app, COUNT(DISTINCT e.seq) n FROM event_entities ee JOIN events e ON e.seq=ee.event_seq "
                   "WHERE ee.entity_id=? GROUP BY e.app ORDER BY n DESC LIMIT 5", (eid,))
    related = store.q(
        "SELECT x.rel rel, x.dir dir, en.type t, en.name n, COUNT(*) c FROM ("
        " SELECT rel, dst AS o, 'out' AS dir FROM edges WHERE src=? AND rel!='VIA'"
        " UNION ALL SELECT rel, src, 'in' FROM edges WHERE dst=? AND rel!='VIA') x JOIN entities en ON en.id=x.o "
        "GROUP BY x.rel, x.dir, x.o ORDER BY c DESC LIMIT 10", (eid, eid))
    recent = store.q("SELECT e.seq FROM event_entities ee JOIN events e ON e.seq=ee.event_seq WHERE ee.entity_id=? "
                     "AND e.purged=0 AND e.text IS NOT NULL GROUP BY e.seq ORDER BY e.ts DESC LIMIT 5", (eid,))
    card = {"id": eid, "type": e["type"], "name": e["name"], "mentions": e["mentions"],
            "first_seen": clock.iso(e["created_ts"]), "last_seen": clock.iso(e["last_seen"]),
            "aliases": [r["alias"] for r in store.q("SELECT alias FROM aliases WHERE entity_id=? LIMIT 10", (eid,))],
            "apps": [{"app": _label(r["app"], labels), "events": r["n"]} for r in apps],
            "related": [{"rel": r["rel"], "direction": r["dir"], "entity": {"type": r["t"], "name": r["n"]}, "count": r["c"]} for r in related],
            "recent_events": _hits(store, [r["seq"] for r in recent], {}, {}, labels, clock)}
    return {"query": name, "found": True, "entity": card, "ambiguous": len(same) > 1 and best["score"] < 1.0,
            "other_matches": cands[1:]}


def explain(store: Store, event_id: str, *, labels: dict[str, str], clock: Clock) -> dict[str, Any]:
    r = store.one("SELECT * FROM events WHERE id=?", (event_id,))
    if not r:
        return {"found": False, "event_id": event_id}
    links = store.q("SELECT en.id,en.type,en.name,ee.role,ee.source FROM event_entities ee JOIN entities en "
                    "ON en.id=ee.entity_id WHERE ee.event_seq=?", (r["seq"],))
    edges = store.q("SELECT sa.name src, e.rel, sb.name dst, e.source, e.evidence FROM edges e JOIN entities sa ON sa.id=e.src "
                    "JOIN entities sb ON sb.id=e.dst WHERE e.event_seq=?", (r["seq"],))
    return {"found": True, "event_id": event_id, "time": clock.iso(r["ts"]), "app": r["app"], "app_label": _label(r["app"], labels),
            "type": r["type"], "device": r["device"], "text": r["text"], "data": json.loads(r["data"]),
            "privacy_flags": json.loads(r["flags"]), "purged": bool(r["purged"]),
            "enrichment": {0: "pending", 1: "done", 2: "gave_up"}[r["enriched"]],
            "vector_model": store.scalar("SELECT model FROM vectors WHERE event_seq=?", (r["seq"],)),
            "entities": [dict(x) for x in links], "edges": [dict(x) for x in edges]}
