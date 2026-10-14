"""SQLite store: raw events, FTS5, vectors, entity graph.

One connection guarded by an RLock, WAL, and secure_delete=ON so forgotten rows are overwritten in place.
SQLite is authoritative; Neo4j (optional) is a read replica.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np

DDL = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  device TEXT NOT NULL, type TEXT NOT NULL, ts INTEGER NOT NULL, app TEXT NOT NULL,
  data TEXT NOT NULL, text TEXT, flags TEXT NOT NULL DEFAULT '[]',
  ingested_at INTEGER NOT NULL,
  enriched INTEGER NOT NULL DEFAULT 0,   -- 0 pending, 1 done, 2 gave up
  attempts INTEGER NOT NULL DEFAULT 0,
  purged INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS ix_events_app_ts ON events(app, ts);
CREATE INDEX IF NOT EXISTS ix_events_type_ts ON events(type, ts);
CREATE INDEX IF NOT EXISTS ix_events_pending ON events(enriched) WHERE enriched=0;
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(text, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS vectors(event_seq INTEGER PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS entities(
  id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, name TEXT NOT NULL,
  created_ts INTEGER NOT NULL, last_seen INTEGER NOT NULL, mentions INTEGER NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0, dirty INTEGER NOT NULL DEFAULT 1);
CREATE INDEX IF NOT EXISTS ix_entities_type ON entities(type);
CREATE TABLE IF NOT EXISTS aliases(alias TEXT NOT NULL, type TEXT NOT NULL, entity_id INTEGER NOT NULL, PRIMARY KEY(alias, type));
CREATE INDEX IF NOT EXISTS ix_aliases_entity ON aliases(entity_id);
CREATE TABLE IF NOT EXISTS event_entities(
  event_seq INTEGER NOT NULL, entity_id INTEGER NOT NULL, role TEXT NOT NULL, source TEXT NOT NULL,
  PRIMARY KEY(event_seq, entity_id, role));
CREATE INDEX IF NOT EXISTS ix_ee_entity ON event_entities(entity_id);
CREATE TABLE IF NOT EXISTS edges(
  id INTEGER PRIMARY KEY AUTOINCREMENT, src INTEGER NOT NULL, rel TEXT NOT NULL, dst INTEGER NOT NULL,
  event_seq INTEGER NOT NULL, ts INTEGER NOT NULL, evidence TEXT, source TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS ix_edges_event ON edges(event_seq);
CREATE TABLE IF NOT EXISTS merge_candidates(
  a INTEGER NOT NULL, b INTEGER NOT NULL, score REAL NOT NULL, status TEXT NOT NULL DEFAULT 'open', PRIMARY KEY(a, b));
CREATE TABLE IF NOT EXISTS devices(
  device TEXT PRIMARY KEY, last_seen INTEGER NOT NULL DEFAULT 0, state_ts INTEGER NOT NULL DEFAULT 0, fg TEXT, screen TEXT);
CREATE TABLE IF NOT EXISTS digests(day TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tombstones(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, ref INTEGER NOT NULL);
"""


class Store:
    def __init__(self, path: str, *, secure_delete: bool = True):
        self.path = path
        self.lock = threading.RLock()
        self._depth = 0
        self.vec_gen = 0
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        c = self.conn
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        if secure_delete:
            c.execute("PRAGMA secure_delete=ON")
        try:
            c.executescript(DDL)
        except sqlite3.OperationalError as e:
            if "fts5" in str(e).lower():
                raise RuntimeError("this SQLite build lacks FTS5") from e
            raise

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Re-entrant write transaction (BEGIN IMMEDIATE ... COMMIT) holding the store lock."""
        with self.lock:
            if self._depth:
                self._depth += 1
                try:
                    yield self.conn
                finally:
                    self._depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")
            finally:
                self._depth = 0

    def q(self, sql: str, args: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, args).fetchone()

    def scalar(self, sql: str, args: Sequence[Any] = ()) -> Any:
        r = self.one(sql, args)
        return r[0] if r else None

    def bump_vec(self) -> None:
        self.vec_gen += 1

    def checkpoint(self) -> None:
        with self.lock:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def stats(self) -> dict[str, Any]:
        s: dict[str, Any] = {
            "events": self.scalar("SELECT COUNT(*) FROM events"),
            "purged": self.scalar("SELECT COUNT(*) FROM events WHERE purged=1"),
            "pending_enrich": self.scalar("SELECT COUNT(*) FROM events WHERE enriched=0 AND purged=0"),
            "vectors": self.scalar("SELECT COUNT(*) FROM vectors"),
            "entities": self.scalar("SELECT COUNT(*) FROM entities"),
            "edges": self.scalar("SELECT COUNT(*) FROM edges"),
            "open_merge_candidates": self.scalar("SELECT COUNT(*) FROM merge_candidates WHERE status='open'"),
            "digests": self.scalar("SELECT COUNT(*) FROM digests"),
        }
        s["by_type"] = {r[0]: r[1] for r in self.q("SELECT type, COUNT(*) FROM events GROUP BY type")}
        return s

    def close(self) -> None:
        with self.lock:
            self.conn.close()


class VectorIndex:
    """Brute-force cosine over an in-memory matrix, rebuilt when vectors change.
    Fine to ~200k events at 384 dims; swap for sqlite-vec/LanceDB beyond that."""

    def __init__(self, store: Store):
        self.store = store
        self._key: tuple[Any, ...] | None = None
        self._ids = np.zeros(0, dtype=np.int64)
        self._mat = np.zeros((0, 0), dtype=np.float32)

    def _refresh(self, model: str) -> None:
        r = self.store.one("SELECT COUNT(*) n, COALESCE(MAX(event_seq),0) m FROM vectors WHERE model=?", (model,))
        key = (model, r["n"], r["m"], self.store.vec_gen)
        if key == self._key:
            return
        rows = self.store.q("SELECT event_seq, dim, vec FROM vectors WHERE model=? ORDER BY event_seq", (model,))
        if rows:
            dim = rows[0]["dim"]
            rows = [x for x in rows if x["dim"] == dim]
            self._ids = np.array([x["event_seq"] for x in rows], dtype=np.int64)
            self._mat = np.vstack([np.frombuffer(x["vec"], dtype=np.float32) for x in rows])
        else:
            self._ids, self._mat = np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        self._key = key

    def search(self, qvec: np.ndarray, topk: int, allowed: set[int] | None, model: str,
               min_score: float = 0.0) -> list[tuple[int, float]]:
        self._refresh(model)
        if not len(self._ids) or self._mat.shape[1] != qvec.shape[0]:
            return []
        scores = self._mat @ qvec
        if allowed is not None:
            mask = np.isin(self._ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
            scores = np.where(mask, scores, -np.inf)
        k = min(topk, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(self._ids[i]), float(scores[i])) for i in idx if np.isfinite(scores[i]) and scores[i] >= min_score]
