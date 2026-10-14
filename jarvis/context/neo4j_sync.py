"""Optional Neo4j read-replica of the graph. SQLite stays authoritative.

Only structure is exported: entity names/aliases, relations, and event ids as pointers -- never message text.
Deletions (forget/merge/retention) travel as tombstones so the replica cannot keep what the source forgot.

EXPERIMENTAL: unit-tested against a fake driver; not exercised against a live Neo4j in this build.
"""
from __future__ import annotations

import logging
from typing import Any

from .extract import ENTITY_TYPES, REL_TYPES
from .store import Store

log = logging.getLogger("recall.neo4j")
CURSOR_KEY = "neo4j_edge_cursor"
CONSTRAINT = "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
DELETE_ENTITIES = "UNWIND $ids AS i MATCH (n:Entity {id: i}) DETACH DELETE n"
DELETE_EDGES = "UNWIND $ids AS i MATCH ()-[r {id: i}]-() DELETE r"


class Neo4jUnavailable(RuntimeError):
    pass


def entity_cypher(etype: str) -> str:
    if etype not in ENTITY_TYPES:  # labels are interpolated, so they must come from the allowlist
        raise ValueError(f"unknown entity type {etype!r}")
    return ("UNWIND $rows AS r MERGE (n:Entity {id: r.id}) "
            f"SET n:`{etype.capitalize()}`, n.type = r.type, n.name = r.name, n.mentions = r.mentions, "
            "n.first_seen = r.first_seen, n.last_seen = r.last_seen, n.aliases = r.aliases")


def edge_cypher(rel: str) -> str:
    if rel not in REL_TYPES:
        raise ValueError(f"unknown relation {rel!r}")
    return ("UNWIND $rows AS r MATCH (a:Entity {id: r.src}), (b:Entity {id: r.dst}) "
            f"MERGE (a)-[e:`{rel}` {{id: r.id}}]->(b) SET e.ts = r.ts, e.event = r.event, e.source = r.source")


def _drain_tombstones(store: Store, session: Any, batch: int) -> int:
    n = 0
    while True:
        rows = store.q("SELECT id, kind, ref FROM tombstones ORDER BY id LIMIT ?", (batch,))
        if not rows:
            return n
        ents = [r["ref"] for r in rows if r["kind"] == "entity"]
        edges = [r["ref"] for r in rows if r["kind"] == "edge"]
        if ents:
            session.run(DELETE_ENTITIES, ids=ents)
        if edges:
            session.run(DELETE_EDGES, ids=edges)
        with store.tx() as c:
            c.executemany("DELETE FROM tombstones WHERE id=?", [(r["id"],) for r in rows])
        n += len(rows)


def _sync_entities(store: Store, session: Any, batch: int) -> int:
    n = 0
    while True:
        rows = store.q("SELECT id,type,name,mentions,created_ts,last_seen FROM entities WHERE dirty=1 ORDER BY id LIMIT ?", (batch,))
        if not rows:
            return n
        by_type: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            aliases = [a["alias"] for a in store.q("SELECT alias FROM aliases WHERE entity_id=? LIMIT 20", (r["id"],))]
            by_type.setdefault(r["type"], []).append({
                "id": r["id"], "type": r["type"], "name": r["name"], "mentions": r["mentions"],
                "first_seen": r["created_ts"], "last_seen": r["last_seen"], "aliases": aliases})
        for etype, batch_rows in by_type.items():
            session.run(entity_cypher(etype), rows=batch_rows)
        with store.tx() as c:  # stay dirty if the entity changed while we were exporting it
            c.executemany("UPDATE entities SET dirty=0 WHERE id=? AND mentions=? AND last_seen=?",
                          [(r["id"], r["mentions"], r["last_seen"]) for r in rows])
            still = c.execute(f"SELECT COUNT(*) FROM entities WHERE dirty=1 AND id IN ({','.join('?' * len(rows))})",
                              [r["id"] for r in rows]).fetchone()[0]
        n += len(rows)
        if still == len(rows):  # nothing cleared: avoid spinning on a hot entity
            return n


def _sync_edges(store: Store, session: Any, batch: int) -> int:
    n = 0
    while True:
        cur = int(store.scalar("SELECT v FROM meta WHERE k=?", (CURSOR_KEY,)) or 0)
        rows = store.q("SELECT e.id, e.src, e.rel, e.dst, e.ts, e.source, ev.id AS event FROM edges e "
                       "JOIN events ev ON ev.seq=e.event_seq WHERE e.id>? ORDER BY e.id LIMIT ?", (cur, batch))
        if not rows:
            return n
        by_rel: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_rel.setdefault(r["rel"], []).append({k: r[k] for k in ("id", "src", "dst", "ts", "source", "event")})
        for rel, batch_rows in by_rel.items():
            session.run(edge_cypher(rel), rows=batch_rows)
        with store.tx() as c:
            c.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                      (CURSOR_KEY, str(rows[-1]["id"])))
        n += len(rows)


def sync(store: Store, driver: Any, batch: int = 500) -> dict[str, int]:
    """Tombstones first (deletes), then entities, then edges (which need their endpoints to exist)."""
    with driver.session() as s:
        s.run(CONSTRAINT)
        deleted = _drain_tombstones(store, s, batch)
        entities = _sync_entities(store, s, batch)
        edges = _sync_edges(store, s, batch)
    return {"deleted": deleted, "entities": entities, "edges": edges}


def sync_from_config(store: Store, cfg: dict[str, Any]) -> dict[str, int]:
    n = cfg["neo4j"]
    if not n.get("enabled"):
        raise Neo4jUnavailable("neo4j.enabled is false in the server config")
    try:
        from neo4j import GraphDatabase  # type: ignore[import-not-found]
    except ImportError:
        raise Neo4jUnavailable("the neo4j driver is not installed: pip install 'recall-layer[neo4j]'") from None
    driver = GraphDatabase.driver(n["uri"], auth=(n["user"], n["password"]))
    try:
        driver.verify_connectivity()
        return sync(store, driver)
    except Exception as e:
        raise Neo4jUnavailable(f"{type(e).__name__}: {e}") from None
    finally:
        driver.close()
