"""Ingest + enrichment pipeline.

ingest(): validate -> privacy policy (again) -> store -> FTS -> deterministic entities/edges. Idempotent by event id.
enrich(): embeddings + optional LLM extraction for events not yet enriched; failures retry with an attempt cap.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any

import numpy as np

from . import extract, privacy, resolve, schema
from .embed import HashEmbedder, OpenAIEmbedder
from .store import Store, VectorIndex

log = logging.getLogger("recall.pipeline")
MAX_ATTEMPTS = 20


def _dumps(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


class Pipeline:
    def __init__(self, store: Store, deny: privacy.Denylist, labels: dict[str, str], *, redact: bool = True,
                 embedder: HashEmbedder | OpenAIEmbedder | None = None, extractor: extract.LLMExtractor | None = None):
        self.store, self.deny, self.labels, self.redact = store, deny, labels, redact
        self.embedder, self.extractor = embedder, extractor
        self.vindex = VectorIndex(store)

    # ---- ingest ----
    def _touch_device(self, c: sqlite3.Connection, ev: dict[str, Any]) -> None:
        dev, ts, t, d = ev["device"], ev["ts"], ev["type"], ev["data"]
        c.execute("INSERT INTO devices(device,last_seen) VALUES(?,?) "
                  "ON CONFLICT(device) DO UPDATE SET last_seen=MAX(last_seen, excluded.last_seen)", (dev, ts))
        fg = screen = None
        if t == "app_switch":
            fg = ev["app"]
        elif t == "screen":
            screen = d.get("state")
        elif t == "heartbeat":
            fg, screen = d.get("fg"), d.get("screen")
        if fg is not None or screen is not None:
            c.execute("UPDATE devices SET state_ts=?, fg=COALESCE(?,fg), screen=COALESCE(?,screen) "
                      "WHERE device=? AND state_ts<=?", (ts, fg, screen, dev, ts))

    def link(self, c: sqlite3.Connection, seq: int, ts: int, ex: extract.Extraction) -> None:
        ids: dict[tuple[str, str], int] = {}
        for e in ex.entities:
            eid = resolve.resolve(c, e.type, e.name, ts)
            if eid is None:
                continue
            ids[(e.type, e.name)] = eid
            fresh = not c.execute("SELECT 1 FROM event_entities WHERE event_seq=? AND entity_id=?", (seq, eid)).fetchone()
            c.execute("INSERT OR IGNORE INTO event_entities(event_seq,entity_id,role,source) VALUES(?,?,?,?)",
                      (seq, eid, e.role, e.source))
            if fresh:
                c.execute("UPDATE entities SET mentions=mentions+1, dirty=1 WHERE id=?", (eid,))
        for r in ex.relations:
            s, d = ids.get(r.src), ids.get(r.dst)
            if s and d and s != d:
                c.execute("INSERT INTO edges(src,rel,dst,event_seq,ts,evidence,source) VALUES(?,?,?,?,?,?,?)",
                          (s, r.rel, d, seq, ts, r.evidence or None, r.source))

    def ingest(self, events: list[Any]) -> dict[str, Any]:
        accepted: list[str] = []
        dups: list[str] = []
        rejected: list[dict[str, Any]] = []
        dropped = 0
        now = schema.now_ms()
        with self.store.tx() as c:
            for raw in events:
                rid = raw.get("id") if isinstance(raw, dict) else None
                try:
                    ev = schema.validate_event(raw, now)
                except ValueError as e:
                    rejected.append({"id": rid if isinstance(rid, str) else None, "reason": str(e)})
                    continue
                pol = privacy.apply_policy(ev, self.deny, redact=self.redact)
                if pol is None:
                    accepted.append(ev["id"])
                    dropped += 1
                    continue
                self._touch_device(c, pol)
                if pol["type"] == "heartbeat":
                    accepted.append(ev["id"])
                    continue
                text = schema.event_text(pol["type"], pol["data"])
                cur = c.execute(
                    "INSERT OR IGNORE INTO events(id,device,type,ts,app,data,text,flags,ingested_at,enriched) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (pol["id"], pol["device"], pol["type"], pol["ts"], pol["app"], _dumps(pol["data"]), text,
                     _dumps(pol["flags"]), now, 0 if text else 1))
                if cur.rowcount == 0:
                    dups.append(ev["id"])
                    continue
                seq = int(cur.lastrowid or 0)
                if text:
                    c.execute("INSERT INTO events_fts(rowid,text) VALUES(?,?)", (seq, text))
                self.link(c, seq, pol["ts"], extract.deterministic(pol, self.labels))
                accepted.append(ev["id"])
        return {"accepted": accepted, "duplicates": dups, "rejected": rejected, "dropped_by_policy": dropped}

    # ---- enrichment ----
    def enrich(self, limit: int = 64) -> tuple[int, int]:
        """Returns (completed, failed). Network/model calls happen outside the DB lock."""
        rows = self.store.q("SELECT seq,app,ts,data,text,flags FROM events WHERE enriched=0 AND purged=0 "
                            "ORDER BY seq LIMIT ?", (limit,))
        if not rows:
            return 0, 0
        vecs: np.ndarray | None = None
        embed_failed = False
        if self.embedder:
            try:
                vecs = self.embedder.embed([r["text"] for r in rows])
            except Exception as e:
                embed_failed = True
                log.warning("embedding failed: %s", e)
        extractions: dict[int, extract.Extraction] = {}
        llm_failed: set[int] = set()
        if self.extractor and not embed_failed:
            for r in rows:
                ev = {"type": "notification", "app": r["app"], "data": json.loads(r["data"]),
                      "flags": json.loads(r["flags"]), "text": r["text"]}
                if self.extractor.eligible(ev):
                    try:
                        extractions[r["seq"]] = self.extractor.extract(ev)
                    except Exception as e:
                        llm_failed.add(r["seq"])
                        log.warning("llm extraction failed: %s", e)
        done = failed = 0
        with self.store.tx() as c:
            for i, r in enumerate(rows):
                seq = r["seq"]
                if not c.execute("SELECT 1 FROM events WHERE seq=? AND purged=0", (seq,)).fetchone():
                    continue  # forgotten while we were away
                bad = embed_failed or seq in llm_failed
                if vecs is not None and self.embedder is not None:
                    c.execute("INSERT OR REPLACE INTO vectors(event_seq,model,dim,vec) VALUES(?,?,?,?)",
                              (seq, self.embedder.name, int(vecs.shape[1]), vecs[i].astype(np.float32).tobytes()))
                if seq in extractions:
                    self.link(c, seq, r["ts"], extractions[seq])
                if bad:
                    c.execute("UPDATE events SET attempts=attempts+1, "
                              "enriched=CASE WHEN attempts+1>=? THEN 2 ELSE 0 END WHERE seq=?", (MAX_ATTEMPTS, seq))
                    failed += 1
                else:
                    c.execute("UPDATE events SET enriched=1 WHERE seq=?", (seq,))
                    done += 1
        self.store.bump_vec()
        return done, failed


class EnrichWorker(threading.Thread):
    """Background enrichment. `pause()` lets the voice loop reclaim the GPU during a turn."""

    def __init__(self, pipeline: Pipeline, interval: float = 2.0):
        super().__init__(name="enrich", daemon=True)
        self.p, self.interval = pipeline, interval
        self._halt, self._wake = threading.Event(), threading.Event()
        self._paused_until = 0.0

    def kick(self) -> None:
        self._wake.set()

    def pause(self, seconds: float) -> None:
        self._paused_until = time.monotonic() + max(0.0, min(float(seconds), 3600.0))

    def stop(self) -> None:
        self._halt.set()
        self._wake.set()

    def run(self) -> None:
        backoff = 5.0
        while not self._halt.is_set():
            if time.monotonic() < self._paused_until:
                self._halt.wait(0.5)
                continue
            try:
                done, failed = self.p.enrich()
            except Exception:
                log.exception("enrich crashed")
                done, failed = 0, 1
            if failed and not done:
                self._halt.wait(backoff)
                backoff = min(backoff * 2, 60.0)
            else:
                backoff = 5.0
                if done == 0:
                    self._wake.wait(self.interval)
                    self._wake.clear()
