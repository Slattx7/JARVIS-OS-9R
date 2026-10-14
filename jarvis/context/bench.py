"""Synthetic benchmark: ingest throughput, retrieval quality, filter correctness, query latency.

Quality numbers are for the active embedder. The default `hash` embedder is lexical, so typo/paraphrase/semantic
queries are *expected* to be weak; they are reported as (info) so a real embedder's gain is measurable:
    recall bench --embed openai   (against a running /v1/embeddings server)
"""
from __future__ import annotations

import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from . import query as Q
from .config import load_config
from .runtime import build_runtime
from .schema import make_event_id, now_ms

PEOPLE = ["Aarav Mehta", "Priya Nair", "Rohan Kulkarni", "Sneha Iyer", "Vikram Rao", "Neha Joshi", "Karan Shah", "Meera Pillai"]
PLACES = ["Vaishali", "Sinhagad", "Koregaon Park", "Lonavala", "Panshet"]
TOPICS: dict[str, list[str]] = {
    "invoice": ["Can you send the invoice for {m} by Friday?", "Invoice for {m} attached, please review", "Reminder: {m} invoice is still unpaid"],
    "flight": ["Your flight to {c} is rescheduled to {t}", "Flight {k} boarding gate changed to {g}", "Check-in for your flight to {c} is open"],
    "dinner": ["Dinner at {p} on Saturday? {t}", "Booked a table at {p} for {t}", "Are we still on for dinner at {p}?"],
    "deploy": ["Deploy failed on staging, check the logs", "Production deploy finished with {n} warnings", "Rollback triggered for release {k}"],
    "gym": ["Gym membership renewal is due next week", "Your gym class starts at {t}", "Gym renewal payment received, thanks"],
    "standup": ["Standup moved to {t} tomorrow", "Skipping standup today, async update in thread", "Standup notes posted in the channel"],
    "trek": ["Trek to {p} this Sunday, bring water", "Carpool for the {p} trek leaves at {t}", "Trek cancelled due to rain at {p}"],
    "rent": ["Rent for {m} is due on the 5th", "Landlord confirmed the rent receipt for {m}", "Reminder to transfer rent before the 5th"],
    "passport": ["Passport appointment confirmed for {t}", "Bring original documents for the passport visit", "Passport status updated: ready for pickup"],
    "parcel": ["Your parcel {k} is out for delivery", "Parcel delivery attempted, nobody home", "Parcel {k} delivered to the front desk"],
}
APPS = [("com.whatsapp", "msg"), ("com.slack", "msg"), ("com.google.android.gm", "email"), ("com.google.android.apps.messaging", "msg")]
FILL = {"m": ["March", "April", "May"], "c": ["Delhi", "Goa", "Chennai"], "t": ["6:45 PM", "9:30 AM", "8pm"],
        "k": ["AI-202", "R-2041", "TRK-8841"], "g": ["B12", "A4"], "p": PLACES, "n": ["2", "5", "0"]}
HOUR = 3_600_000


def _gen(n: int, seed: int, days: int, now: int):
    """Returns events plus ground truth. 'lex' = the event text contains the topic word; the rest of a topic's events
    are paraphrases that only a semantic embedder could be expected to find."""
    rng = random.Random(seed)
    events: list[dict[str, Any]] = []
    by_topic, sem_topic = {t: set() for t in TOPICS}, {t: set() for t in TOPICS}
    by_pt: dict[tuple[str, str], set[str]] = {}
    for i in range(n):
        topic = rng.choice(list(TOPICS))
        person, (app, cat) = rng.choice(PEOPLE), rng.choice(APPS)
        text = rng.choice(TOPICS[topic]).format(**{k: rng.choice(v) for k, v in FILL.items()})
        ts = now - rng.randint(HOUR, days * 24 * HOUR)
        data: dict[str, Any] = {"title": person, "text": text, "category": cat}
        if cat == "msg":
            data["template"] = "Notification$MessagingStyle"
        eid = make_event_id("bench", "notification", str(i), ts, str(i))
        events.append({"id": eid, "device": "bench", "type": "notification", "ts": ts, "app": app, "data": data, "flags": []})
        if topic in text.lower():
            by_topic[topic].add(eid)
            by_pt.setdefault((person, topic), set()).add(eid)
        else:
            sem_topic[topic].add(eid)
    return events, by_topic, by_pt, sem_topic


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def run_bench(n_events: int = 2000, seed: int = 7, days: int = 30, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    now = now_ms()
    events, by_topic, by_pt, sem_topic = _gen(n_events, seed, days, now)
    with tempfile.TemporaryDirectory() as td:
        cfg = load_config(None, {"server": {"db_path": str(Path(td) / "bench.db")}, **(config_overrides or {})})
        rt = build_runtime(cfg)
        p = rt.pipeline
        t0 = time.perf_counter()
        for i in range(0, len(events), 200):
            p.ingest(events[i:i + 200])
        t_ingest = time.perf_counter() - t0
        t0 = time.perf_counter()
        while p.enrich(256) != (0, 0):
            pass
        t_enrich = time.perf_counter() - t0

        def q(text: str, **kw: Any) -> list[dict[str, Any]]:
            t = time.perf_counter()
            r = Q.search(p.store, p.embedder, p.vindex, text, labels=rt.labels, clock=rt.clock, now=now, **kw)
            lat.append((time.perf_counter() - t) * 1000)
            return r["hits"]

        lat: list[float] = []
        hit5, prec5, mrr = [], [], []
        for topic, rel in by_topic.items():
            hits = q(topic, limit=10)
            ids = [h["event_id"] for h in hits]
            hit5.append(any(i in rel for i in ids[:5]))
            prec5.append(sum(i in rel for i in ids[:5]) / 5)
            mrr.append(next((1 / (r + 1) for r, i in enumerate(ids) if i in rel), 0.0))
        ent_prec, ent_rec, violations = [], [], 0
        for (person, topic), rel in list(by_pt.items())[:40]:
            hits = q(topic, entity=person, limit=max(1, len(rel)))  # R-precision: ask for exactly |relevant| results
            ids = [h["event_id"] for h in hits]
            ent_prec.append(sum(i in rel for i in ids) / len(rel))
            ent_rec.append(len(set(ids) & rel) / len(rel))
            violations += sum(not any(e["name"] == person for e in h["entities"]) for h in hits)
        cutoff = now - 3 * 24 * HOUR
        windowed = [q(t, start=cutoff, limit=20) for t in TOPICS]
        window_violations = sum(h["ts"] < cutoff for hs in windowed for h in hs)
        typo = [any(i in by_topic["invoice"] for i in [h["event_id"] for h in q("invoise", limit=5)])]
        typo.append(any(i in by_topic["passport"] for i in [h["event_id"] for h in q("pasport", limit=5)]))
        para = [any(h["event_id"] in by_topic["flight"] for h in q("airplane delayed", limit=5)),
                any(h["event_id"] in by_topic["gym"] for h in q("fitness subscription", limit=5))]
        sem = [len({h["event_id"] for h in q(t, limit=20)} & ids) / min(20, len(ids)) for t, ids in sem_topic.items() if ids]
        stats = rt.store.stats()
        rt.close()
    return {
        "events": n_events, "entities": stats["entities"], "ingest_eps": round(n_events / t_ingest),
        "enrich_eps": round(n_events / t_enrich) if t_enrich else None,
        "topic_hit@5": round(statistics.mean(hit5), 3), "topic_precision@5": round(statistics.mean(prec5), 3),
        "topic_mrr": round(statistics.mean(mrr), 3),
        "entity_filter_R_precision": round(statistics.mean(ent_prec), 3),
        "entity_filter_violations": violations, "time_filter_violations": window_violations,
        "typo_hit@5 (info)": round(statistics.mean(typo), 2), "paraphrase_hit@5 (info)": round(statistics.mean(para), 2),
        "semantic_recall@20 (info)": round(statistics.mean(sem), 3) if sem else None,
        "query_ms_p50": round(_pct(lat, 0.5), 1), "query_ms_p95": round(_pct(lat, 0.95), 1),
        "embedder": rt.pipeline.embedder.name if rt.pipeline.embedder else None,
    }
