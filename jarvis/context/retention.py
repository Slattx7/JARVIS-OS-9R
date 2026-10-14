"""Retention and forgetting.

Retention keeps long-term memory without keeping raw text: whole days past the raw TTL are summarized into
digests, then notification text/payloads, FTS rows, vectors and edge evidence are purged; the graph persists.
`forget` removes data on demand and cascades to every derived structure, then scrubs digests, optimizes FTS and
(by default) VACUUMs so the bytes are physically gone, not just unlinked.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

from . import query, resolve
from .query import DAY, Clock
from .schema import now_ms
from .store import Store

CHUNK = 500


def _chunks(xs: list[Any]) -> Iterable[list[Any]]:
    for i in range(0, len(xs), CHUNK):
        yield xs[i:i + CHUNK]


def build_digest(store: Store, start: int, end: int, labels: dict[str, str], clock: Clock) -> dict[str, Any]:
    st = query._window_stats(store, start, end, labels, clock)
    return {"day": clock.day_str(start), "notifications": {"total": sum(st["notifications_by_app"].values()),
                                                           "by_app": st["notifications_by_app"]},
            "top_senders": st["top_senders"], "dwell_s_by_app": st["dwell_s_by_app"], "sessions": len(st["sessions"])}


def _delete_events(c: Any, seqs: list[int]) -> set[int]:
    affected: set[int] = set()
    for ch in _chunks(seqs):
        q = ",".join("?" * len(ch))
        affected |= {r[0] for r in c.execute(f"SELECT DISTINCT entity_id FROM event_entities WHERE event_seq IN ({q})", ch)}
        c.execute(f"INSERT INTO tombstones(kind,ref) SELECT 'edge', id FROM edges WHERE event_seq IN ({q})", ch)
        c.execute(f"DELETE FROM edges WHERE event_seq IN ({q})", ch)
        c.execute(f"DELETE FROM event_entities WHERE event_seq IN ({q})", ch)
        c.execute(f"DELETE FROM events_fts WHERE rowid IN ({q})", ch)
        c.execute(f"DELETE FROM vectors WHERE event_seq IN ({q})", ch)
        c.execute(f"DELETE FROM events WHERE seq IN ({q})", ch)
    return affected


def _drop_entity(c: Any, eid: int) -> None:
    c.execute("INSERT INTO tombstones(kind,ref) SELECT 'edge', id FROM edges WHERE src=? OR dst=?", (eid, eid))
    c.execute("DELETE FROM edges WHERE src=? OR dst=?", (eid, eid))
    c.execute("DELETE FROM event_entities WHERE entity_id=?", (eid,))
    c.execute("DELETE FROM aliases WHERE entity_id=?", (eid,))
    c.execute("DELETE FROM merge_candidates WHERE a=? OR b=?", (eid, eid))
    c.execute("DELETE FROM entities WHERE id=?", (eid,))
    c.execute("INSERT INTO tombstones(kind,ref) VALUES('entity',?)", (eid,))


def _physical_cleanup(store: Store, vacuum: bool) -> None:
    with store.lock:
        store.conn.execute("INSERT INTO events_fts(events_fts) VALUES('optimize')")
        if vacuum:
            store.conn.execute("VACUUM")
    store.checkpoint()
    store.bump_vec()


def run_retention(store: Store, cfg: dict[str, Any], clock: Clock, labels: dict[str, str], now: int | None = None) -> dict[str, Any]:
    now = now_ms() if now is None else now
    raw_days, meta_days = int(cfg["retention"]["raw_ttl_days"]), int(cfg["retention"]["meta_ttl_days"])
    cut = clock.day_start(now - raw_days * DAY)  # whole days only, so every digest is complete
    digests = 0
    with store.tx() as c:
        first = c.execute("SELECT MIN(ts) FROM events").fetchone()[0]
        row = c.execute("SELECT v FROM meta WHERE k='digest_through'").fetchone()
        day = int(row[0]) if row else (clock.day_start(first) if first else cut)
        built: set[int] = set()
        while day < cut:  # complete days that have no digest yet
            nxt = clock.next_day(day)
            body = build_digest(store, day, nxt, labels, clock)
            if body["notifications"]["total"] or body["dwell_s_by_app"]:
                digests += c.execute("INSERT OR IGNORE INTO digests(day,body) VALUES(?,?)",
                                     (body["day"], json.dumps(body, ensure_ascii=False))).rowcount
            built.add(day)
            day = nxt
        c.execute("INSERT INTO meta(k,v) VALUES('digest_through',?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (str(day),))
        rows = c.execute("SELECT seq, ts FROM events WHERE type='notification' AND purged=0 AND ts<?", (cut,)).fetchall()
        seqs = [r[0] for r in rows]
        for d0 in sorted({clock.day_start(r[1]) for r in rows} - built):  # late arrivals: refresh that day's digest
            body = build_digest(store, d0, clock.next_day(d0), labels, clock)
            c.execute("INSERT OR REPLACE INTO digests(day,body) VALUES(?,?)", (body["day"], json.dumps(body, ensure_ascii=False)))
            digests += 1
        for ch in _chunks(seqs):
            q = ",".join("?" * len(ch))
            c.execute(f"DELETE FROM events_fts WHERE rowid IN ({q})", ch)
            c.execute(f"DELETE FROM vectors WHERE event_seq IN ({q})", ch)
            c.execute(f"UPDATE edges SET evidence=NULL WHERE event_seq IN ({q})", ch)
            c.execute(f"UPDATE events SET text=NULL, data='{{}}', purged=1 WHERE seq IN ({q})", ch)
        old = [r[0] for r in c.execute("SELECT seq FROM events WHERE ts<?", (clock.day_start(now - meta_days * DAY),))]
        affected = _delete_events(c, old)
        for eid in affected:
            n = c.execute("SELECT COUNT(DISTINCT event_seq) FROM event_entities WHERE entity_id=?", (eid,)).fetchone()[0]
            c.execute("UPDATE entities SET mentions=?, dirty=1 WHERE id=?", (n, eid))
            if n == 0 and not c.execute("SELECT pinned FROM entities WHERE id=?", (eid,)).fetchone()[0]:
                _drop_entity(c, eid)
        if not cfg.get("neo4j", {}).get("enabled"):
            c.execute("DELETE FROM tombstones")  # nothing consumes them
    if seqs or old:
        _physical_cleanup(store, vacuum=False)
    return {"digests_built": digests, "events_purged": len(seqs), "events_deleted": len(old)}


def _resolve_forget_entity(store: Store, entity: Any) -> list[int]:
    if isinstance(entity, int) or (isinstance(entity, str) and entity.strip().isdigit()):
        return [int(entity)]
    cands = resolve.lookup(store, str(entity), limit=5)
    exact = [c for c in cands if c["score"] >= 0.999]
    near = [c for c in cands if c["score"] >= resolve.CANDIDATE]
    pick = exact or (near if len(near) == 1 else [])
    if not pick:
        names = ", ".join(f"{c['name']} (id {c['id']})" for c in cands) or "none"
        raise ValueError(f"entity {entity!r} is ambiguous or unknown; pass an exact name or id. Matches: {names}")
    return [c["id"] for c in pick]


def forget(store: Store, *, labels: dict[str, str], clock: Clock, app: str | None = None, entity: Any = None,
           since: Any = None, until: Any = None, event_ids: list[str] | None = None, everything: bool = False,
           dry_run: bool = False, vacuum: bool = True, now: int | None = None) -> dict[str, Any]:
    now = now_ms() if now is None else now
    conds, args, needles = ["1=1"], [], []
    scoped = False
    if app:
        pk = query.resolve_apps([app], labels)
        conds.append(f"e.app IN ({','.join('?' * len(pk))})")
        args += pk
        needles += [app, *[query._label(p, labels) for p in pk], *pk]
        scoped = True
    ent_ids: list[int] = []
    if entity not in (None, ""):
        ent_ids = _resolve_forget_entity(store, entity)
        conds.append(f"e.seq IN (SELECT event_seq FROM event_entities WHERE entity_id IN ({','.join('?' * len(ent_ids))}))")
        args += ent_ids
        needles += [r["name"] for r in store.q(f"SELECT name FROM entities WHERE id IN ({','.join('?' * len(ent_ids))})", ent_ids)]
        scoped = True
    start, end = query.resolve_range(since, until, now, clock)
    if start is not None:
        conds.append("e.ts>=?")
        args.append(start)
        scoped = True
    if end is not None:
        conds.append("e.ts<?")
        args.append(end)
        scoped = True
    if event_ids:
        conds.append(f"e.id IN ({','.join('?' * len(event_ids))})")
        args += list(event_ids)
        scoped = True
    if not scoped and not everything:
        raise ValueError("refusing to forget everything: pass at least one filter, or everything=True")
    where = " AND ".join(conds)
    if dry_run:
        n = store.scalar(f"SELECT COUNT(*) FROM events e WHERE {where}", args)
        return {"dry_run": True, "events_matched": n, "entities_targeted": len(ent_ids)}

    scrubbed = 0
    with store.tx() as c:
        seqs = [r[0] for r in c.execute(f"SELECT e.seq FROM events e WHERE {where}", args)]
        affected = _delete_events(c, seqs)
        dropped = 0
        for eid in affected | set(ent_ids):
            row = c.execute("SELECT pinned FROM entities WHERE id=?", (eid,)).fetchone()
            if row is None:
                continue
            n = c.execute("SELECT COUNT(DISTINCT event_seq) FROM event_entities WHERE entity_id=?", (eid,)).fetchone()[0]
            c.execute("UPDATE entities SET mentions=?, dirty=1 WHERE id=?", (n, eid))
            if eid in ent_ids or (n == 0 and not row[0]):
                _drop_entity(c, eid)
                dropped += 1
        for day, body in c.execute("SELECT day, body FROM digests").fetchall():
            low = body.casefold()
            in_window = False
            if start is not None or end is not None:
                d0 = clock.midnight(datetime.strptime(day, "%Y-%m-%d").date())
                in_window = (start is None or clock.next_day(d0) > start) and (end is None or d0 < end)
            if everything or in_window or any(n and n.casefold() in low for n in needles):
                c.execute("DELETE FROM digests WHERE day=?", (day,))
                scrubbed += 1
    _physical_cleanup(store, vacuum)
    return {"events_deleted": len(seqs), "entities_deleted": dropped, "digests_scrubbed": scrubbed, "vacuumed": vacuum}
