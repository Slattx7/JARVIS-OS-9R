"""Entity resolution: normalized aliases, conservative fuzzy auto-merge, review queue for ambiguous matches.

Auto-merge needs near-identical names (typos). A bare first name vs a full name ("Alice" / "Alice Johnson") is
only ever queued as a merge candidate, never merged automatically: two different Alices must stay separate.
"""
from __future__ import annotations

import difflib
import re
import sqlite3
import unicodedata
from typing import Any

from .store import Store

FUZZY_TYPES = ("person", "org", "place", "thread")
AUTO_MERGE, CANDIDATE, LOOKUP_MIN = 0.93, 0.72, 0.6
_PHONE_LIKE = re.compile(r"^\+?[\d\s\-().]{7,}$")


def norm_alias(name: str, etype: str) -> str:
    name = name.strip()
    if etype == "email":
        return name.lower()
    if etype == "domain":
        d = name.lower()
        return d[4:] if d.startswith("www.") else d
    if etype == "phone" or (etype == "person" and _PHONE_LIKE.match(name)):
        digits = re.sub(r"\D", "", name)
        return "tel:" + digits[-10:] if len(digits) >= 7 else ""
    s = unicodedata.normalize("NFKC", name).casefold()
    out = []
    for ch in s:  # keep letters/marks/numbers in any script (Devanagari etc.); drop emoji and punctuation
        if unicodedata.category(ch)[0] in "LMN":
            out.append(ch)
        elif ch.isspace() or ch in "-_.'":
            out.append(" ")
    return " ".join("".join(out).split())


def _similarity(a: str, b: str) -> float:
    r = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    inter = len(ta & tb)
    if inter:
        r = max(r, 0.85 * inter / min(len(ta), len(tb)))  # capped below AUTO_MERGE on purpose
    return r


def _fuzzy_candidates(conn: sqlite3.Connection, alias: str, etypes: tuple[str, ...]) -> dict[int, float]:
    scores: dict[int, float] = {}
    toks = [t for t in alias.split() if len(t) >= 3] or [alias[:3] or alias]
    for et in etypes:
        for t in toks[:3]:
            rows = conn.execute("SELECT alias, entity_id FROM aliases WHERE type=? AND instr(alias, ?) > 0 LIMIT 200",
                                (et, t[:3])).fetchall()
            for a, eid in rows:
                scores[eid] = max(scores.get(eid, 0.0), _similarity(alias, a))
    return scores


def _touch(conn: sqlite3.Connection, eid: int, ts: int) -> None:
    conn.execute("UPDATE entities SET last_seen=MAX(last_seen,?), created_ts=MIN(created_ts,?), dirty=1 WHERE id=?",
                 (ts, ts, eid))


def resolve(conn: sqlite3.Connection, etype: str, name: str, ts: int, create: bool = True) -> int | None:
    """Map a surface form to an entity id (call inside a Store.tx())."""
    alias = norm_alias(name, etype)
    if not alias:
        return None
    row = conn.execute("SELECT entity_id FROM aliases WHERE alias=? AND type=?", (alias, etype)).fetchone()
    if row:
        _touch(conn, row[0], ts)
        return int(row[0])
    best_id, best = None, 0.0
    if etype in FUZZY_TYPES:
        for eid, sc in _fuzzy_candidates(conn, alias, (etype,)).items():
            if sc > best:
                best_id, best = eid, sc
        if best_id is not None and best >= AUTO_MERGE:
            conn.execute("INSERT OR IGNORE INTO aliases(alias,type,entity_id) VALUES(?,?,?)", (alias, etype, best_id))
            _touch(conn, best_id, ts)
            return best_id
    if not create:
        return None
    eid = conn.execute("INSERT INTO entities(type,name,created_ts,last_seen) VALUES(?,?,?,?)",
                       (etype, name.strip()[:80], ts, ts)).lastrowid
    conn.execute("INSERT INTO aliases(alias,type,entity_id) VALUES(?,?,?)", (alias, etype, eid))
    if best_id is not None and best >= CANDIDATE:
        a, b = sorted((int(eid), int(best_id)))
        conn.execute("INSERT OR IGNORE INTO merge_candidates(a,b,score) VALUES(?,?,?)", (a, b, round(best, 3)))
    return int(eid)


def lookup(store: Store, name: str, etype: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
    """Read-only: exact alias hits (score 1.0) then fuzzy matches, best first."""
    types = (etype,) if etype else ("person", "org", "place", "thread", "domain", "email", "phone", "task", "app")
    found: dict[int, float] = {}
    with store.lock:
        conn = store.conn
        for t in types:
            a = norm_alias(name, t)
            if not a:
                continue
            for r in conn.execute("SELECT entity_id FROM aliases WHERE alias=? AND type=?", (a, t)).fetchall():
                found[r[0]] = 1.0
        fz = tuple(t for t in types if t in FUZZY_TYPES)
        if fz:
            a = norm_alias(name, "person")
            if a:
                for eid, sc in _fuzzy_candidates(conn, a, fz).items():
                    if sc >= LOOKUP_MIN:
                        found[eid] = max(found.get(eid, 0.0), sc)
        out = []
        for eid, sc in found.items():
            e = conn.execute("SELECT id,type,name,mentions,last_seen FROM entities WHERE id=?", (eid,)).fetchone()
            if e:
                out.append({"id": e["id"], "type": e["type"], "name": e["name"], "mentions": e["mentions"],
                            "last_seen": e["last_seen"], "score": round(sc, 3)})
    out.sort(key=lambda d: (-d["score"], -d["mentions"]))
    return out[:limit]


def merge_entities(store: Store, keep: int, drop: int) -> dict[str, Any]:
    if keep == drop:
        raise ValueError("cannot merge an entity into itself")
    with store.tx() as c:
        for e in (keep, drop):
            if not c.execute("SELECT 1 FROM entities WHERE id=?", (e,)).fetchone():
                raise ValueError(f"no such entity: {e}")
        c.execute("UPDATE OR IGNORE aliases SET entity_id=? WHERE entity_id=?", (keep, drop))
        c.execute("DELETE FROM aliases WHERE entity_id=?", (drop,))
        c.execute("UPDATE OR IGNORE event_entities SET entity_id=? WHERE entity_id=?", (keep, drop))
        c.execute("DELETE FROM event_entities WHERE entity_id=?", (drop,))
        lowest = c.execute("SELECT MIN(id) FROM edges WHERE src=? OR dst=?", (drop, drop)).fetchone()[0]
        if lowest is not None:  # let an optional Neo4j replica re-export the re-pointed edges
            c.execute("UPDATE meta SET v=CAST(MIN(CAST(v AS INTEGER), ?) AS TEXT) WHERE k='neo4j_edge_cursor'", (lowest - 1,))
        c.execute("UPDATE edges SET src=? WHERE src=?", (keep, drop))
        c.execute("UPDATE edges SET dst=? WHERE dst=?", (keep, drop))
        for (eid,) in c.execute("SELECT id FROM edges WHERE src=dst").fetchall():
            c.execute("INSERT INTO tombstones(kind,ref) VALUES('edge',?)", (eid,))
        c.execute("DELETE FROM edges WHERE src=dst")
        c.execute("DELETE FROM merge_candidates WHERE a IN (?,?) OR b IN (?,?)", (keep, drop, keep, drop))
        c.execute("""UPDATE entities SET dirty=1,
            mentions=(SELECT COUNT(DISTINCT event_seq) FROM event_entities WHERE entity_id=entities.id),
            created_ts=MIN(created_ts,(SELECT created_ts FROM entities WHERE id=?)),
            last_seen=MAX(last_seen,(SELECT last_seen FROM entities WHERE id=?)) WHERE id=?""", (drop, drop, keep))
        c.execute("DELETE FROM entities WHERE id=?", (drop,))
        c.execute("INSERT INTO tombstones(kind,ref) VALUES('entity',?)", (drop,))
        row = c.execute("SELECT id,type,name,mentions FROM entities WHERE id=?", (keep,)).fetchone()
    return {"kept": dict(row)}


def candidates(store: Store, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
    rows = store.q("""SELECT m.a, m.b, m.score, ea.name AS an, eb.name AS bn, ea.type AS t,
                      ea.mentions AS am, eb.mentions AS bm
                      FROM merge_candidates m JOIN entities ea ON ea.id=m.a JOIN entities eb ON eb.id=m.b
                      WHERE m.status=? ORDER BY m.score DESC LIMIT ?""", (status, limit))
    return [{"a": r["a"], "b": r["b"], "type": r["t"], "a_name": r["an"], "b_name": r["bn"],
             "a_mentions": r["am"], "b_mentions": r["bm"], "score": r["score"]} for r in rows]
