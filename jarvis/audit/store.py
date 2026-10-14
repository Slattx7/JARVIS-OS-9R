"""Append-only, hash-chained event store (SQLite WAL) + content-addressed blobs.

One table, `events`. A trace is a *view* folded from its events:
  trace_start, span_start, span_annot, span_end, trace_end, label
Spans are written write-ahead (span_start carries inputs + intent before the work runs), so a crash
mid-action leaves a span_start without a span_end = "outcome unknown", which audit surfaces.

Tamper evidence: every row's `chain` = sha256(prev_chain | ts | type | trace | span | body).
UPDATE/DELETE are blocked by triggers; verify() recomputes the chain. A process with root can still
rewrite the whole file, so anchor `store.head()` off-device if that matters.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import io
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .redact import Redactor
from .schema import (ERROR, OK, ORPHANED, RUNNING, SCHEMA_VERSION, NormCtx, Span, Trace, blob_ref, canonical)

GENESIS = hashlib.sha256(b"jarvis-trace/v1").hexdigest()
_BLOB_RE = re.compile(r'"__blob__":"(sha256:[0-9a-f]{64})"')
_REF_OK = re.compile(r"^sha256:[0-9a-f]{64}$")

_DDL = """
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ns INTEGER NOT NULL,
  trace_id TEXT NOT NULL,
  span_id TEXT,
  type TEXT NOT NULL,
  body TEXT NOT NULL,
  chain TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_trace ON events(trace_id, seq);
CREATE INDEX IF NOT EXISTS ix_events_type ON events(type, seq);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;
"""


class BlobCorrupt(RuntimeError):
    pass


class MemBlobs:
    def __init__(self) -> None:
        self._d: Dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        ref = blob_ref(data)
        self._d.setdefault(ref, data)
        return ref

    def has(self, ref: str) -> bool:
        return ref in self._d

    def get(self, ref: str) -> bytes:
        return self._d[ref]


class FileBlobs:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _p(self, ref: str) -> Path:
        if not _REF_OK.match(ref):
            raise ValueError(f"bad blob ref {ref!r}")
        h = ref.split(":", 1)[1]
        return self.root / h[:2] / h[2:]

    def put(self, data: bytes) -> str:
        ref = blob_ref(data)
        p = self._p(ref)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + f".tmp{os.getpid()}.{threading.get_ident()}")
            tmp.write_bytes(data)
            os.replace(tmp, p)
        return ref

    def has(self, ref: str) -> bool:
        try:
            return self._p(ref).exists()
        except ValueError:
            return False

    def get(self, ref: str) -> bytes:
        data = self._p(ref).read_bytes()
        if blob_ref(data) != ref:
            raise BlobCorrupt(ref)
        return data


def _load_key(p: Path) -> bytes:
    for _ in range(50):
        if p.exists():
            txt = p.read_text().strip()
            if txt:
                return bytes.fromhex(txt)
            time.sleep(0.02)
            continue
        key = os.urandom(32)
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            f.write(key.hex())
        return key
    raise RuntimeError(f"could not load or create key file {p}")


def _adopt_key(p: Path, key: bytes) -> bytes:
    """Persist a caller-supplied key next to the store (0600); refuse to silently disagree with an existing one."""
    if p.exists() and p.read_text().strip():
        if bytes.fromhex(p.read_text().strip()) != key:
            raise ValueError(f"{p}: existing key file differs from the key supplied")
        return key
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return key
    with os.fdopen(fd, "w") as f:
        f.write(key.hex())
    return key


def _chain(prev: str, ts_ns: int, typ: str, trace_id: str, span_id: Optional[str], body: str) -> str:
    h = hashlib.sha256()
    for part in (prev, str(ts_ns), typ, trace_id, span_id or "", body):
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


@dataclass
class VerifyReport:
    ok: bool
    events: int
    head: str
    problems: List[str] = field(default_factory=list)
    unfinished: List[Dict[str, Any]] = field(default_factory=list)  # spans with no span_end (stale/ended trace)
    rotation: Optional[Dict[str, Any]] = None                       # set when this store was created by rotate_to()


class Store:
    def __init__(self, path: str | os.PathLike = ":memory:", *, key: Optional[bytes] = None, redact: bool = True,
                 max_inline: int = 64 * 1024, extra_patterns: Iterable[Tuple[str, Any, int]] = ()) -> None:
        self.path = str(path) if str(path) == ":memory:" else os.path.expanduser(str(path))
        self.redact_enabled = redact
        self.max_inline = max_inline
        self._extra = tuple(extra_patterns)
        self._lock = threading.RLock()
        mem = self.path == ":memory:"
        if not mem:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        if not mem:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_DDL)
        if key is not None and not mem:
            key = _adopt_key(Path(self.path + ".key"), key)
        self.key: bytes = key if key is not None else (os.urandom(32) if mem else _load_key(Path(self.path + ".key")))
        self.blobs: Any = MemBlobs() if mem else FileBlobs(Path(self.path + ".blobs"))
        self.fallback: Optional["Store"] = None  # replay candidate stores resolve baseline blob refs through this
        self.redactor = Redactor(self.key, enabled=redact, extra_patterns=self._extra)

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()

    def clone_empty(self) -> "Store":
        """Fresh in-memory store sharing key + redaction config (so hashes are comparable). Used by replay."""
        s = Store(":memory:", key=self.key, redact=self.redact_enabled, max_inline=self.max_inline,
                  extra_patterns=self._extra)
        s.fallback = self
        return s

    # -- normalization ctx -------------------------------------------------- #
    def norm_ctx(self, *, store_blobs: bool = True) -> NormCtx:
        return NormCtx(redactor=self.redactor, put_blob=self.blobs.put if store_blobs else None,
                       max_inline=self.max_inline)

    def put_blob(self, data: bytes) -> Dict[str, Any]:
        return {"__blob__": self.blobs.put(bytes(data)), "len": len(data)}

    def get_blob(self, ref: str) -> bytes:
        if self.blobs.has(ref):
            return self.blobs.get(ref)
        if self.fallback is not None:
            return self.fallback.get_blob(ref)
        raise KeyError(ref)

    def rehydrate(self, obj: Any) -> Any:
        """Replace {"__blob__": ref} dicts with bytes (str for as=str). Missing blobs are left as refs."""
        if isinstance(obj, dict):
            if isinstance(obj.get("__blob__"), str):
                try:
                    data = self.get_blob(obj["__blob__"])
                except (KeyError, ValueError, FileNotFoundError):
                    return obj
                return data.decode("utf-8") if obj.get("as") == "str" else data
            return {k: self.rehydrate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.rehydrate(v) for v in obj]
        return obj

    # -- write path --------------------------------------------------------- #
    def append(self, typ: str, trace_id: str, span_id: Optional[str], body: Dict[str, Any], *,
               ts_ns: Optional[int] = None, durable: bool = False) -> int:
        ts_ns = time.time_ns() if ts_ns is None else ts_ns
        body_s = canonical(body)
        with self._lock:
            c = self._conn
            if durable:
                c.execute("PRAGMA synchronous=FULL")
            try:
                c.execute("BEGIN IMMEDIATE")
                try:
                    row = c.execute("SELECT chain FROM events ORDER BY seq DESC LIMIT 1").fetchone()
                    prev = row[0] if row else GENESIS
                    chain = _chain(prev, ts_ns, typ, trace_id, span_id, body_s)
                    cur = c.execute(
                        "INSERT INTO events(ts_ns,trace_id,span_id,type,body,chain) VALUES(?,?,?,?,?,?)",
                        (ts_ns, trace_id, span_id, typ, body_s, chain))
                    seq = int(cur.lastrowid)
                    c.execute("COMMIT")
                except BaseException:
                    c.execute("ROLLBACK")
                    raise
            finally:
                if durable:
                    c.execute("PRAGMA synchronous=NORMAL")
        return seq

    def label(self, trace_id: str, label: str, note: str = "", remove: bool = False) -> int:
        if not self.trace_exists(trace_id):
            raise KeyError(trace_id)
        return self.append("label", trace_id, None,
                           {"label": label, "note": note, "op": "remove" if remove else "add"})

    def trace_has_end(self, trace_id: str) -> bool:
        return bool(self._q("SELECT 1 FROM events WHERE trace_id=? AND type='trace_end' LIMIT 1", (trace_id,)))

    def close_trace(self, trace_id: str, reason: str = "process ended without finishing") -> bool:
        """Append an honest trace_end for a trace whose process died (status error, recovered=True). The log stays
        append-only: spans that never finished remain `orphaned`; nothing is rewritten."""
        if not self.trace_exists(trace_id) or self.trace_has_end(trace_id):
            return False
        self.append("trace_end", trace_id, None, {"dur_ns": None, "status": ERROR, "recovered": True,
                                                  "error": {"type": "ProcessDied", "msg": reason}})
        return True

    def head(self) -> str:
        r = self._q("SELECT chain FROM events ORDER BY seq DESC LIMIT 1")
        return r[0][0] if r else GENESIS

    # -- read path ---------------------------------------------------------- #
    def _q(self, sql: str, args: Iterable[Any] = ()) -> List[tuple]:
        with self._lock:
            return self._conn.execute(sql, tuple(args)).fetchall()

    def trace_exists(self, trace_id: str) -> bool:
        return bool(self._q("SELECT 1 FROM events WHERE trace_id=? AND type='trace_start' LIMIT 1", (trace_id,)))

    def count_events(self) -> int:
        return int(self._q("SELECT COUNT(*) FROM events")[0][0])

    def load_trace(self, trace_id: str, *, stale_s: float = 3600) -> Trace:
        rows = self._q("SELECT seq, ts_ns, span_id, type, body FROM events WHERE trace_id=? ORDER BY seq",
                       (trace_id,))
        if not rows:
            raise KeyError(trace_id)
        return _fold(trace_id, rows, stale_s)

    def list_traces(self, *, tag: Optional[str] = None, label: Optional[str] = None, name: Optional[str] = None,
                    since_ns: Optional[int] = None, until_ns: Optional[int] = None, status: Optional[str] = None,
                    limit: Optional[int] = None, newest_first: bool = True, stale_s: float = 3600) -> List[Trace]:
        """Trace headers (no spans), filtered. Use load_trace() for the full thing."""
        q = "SELECT trace_id, ts_ns, body FROM events WHERE type='trace_start'"
        args: List[Any] = []
        if since_ns is not None:
            q += " AND ts_ns>=?"
            args.append(since_ns)
        if until_ns is not None:
            q += " AND ts_ns<?"
            args.append(until_ns)
        q += " ORDER BY seq " + ("DESC" if newest_first else "ASC")
        starts = self._q(q, args)
        ends = {r[0]: json.loads(r[2]) for r in self._q("SELECT trace_id, ts_ns, body FROM events WHERE type='trace_end'")}
        labels: Dict[str, List[dict]] = collections.defaultdict(list)
        for tid, ts, body_s in self._q("SELECT trace_id, ts_ns, body FROM events WHERE type='label' ORDER BY seq"):
            b = json.loads(body_s)
            b["ts_ns"] = ts
            labels[tid].append(b)
        now = time.time_ns()
        out: List[Trace] = []
        for tid, ts, body_s in starts:
            b = json.loads(body_s)
            tr = Trace(trace_id=tid, name=b["name"], t_start_ns=ts, inputs=b.get("inputs"),
                       tags=list(b.get("tags", [])), env=b.get("env", {}), meta=b.get("meta", {}),
                       schema=b.get("schema", SCHEMA_VERSION), labels=labels.get(tid, []))
            e = ends.get(tid)
            if e is not None:
                tr.status = e.get("status", OK)
                tr.dur_ns = e.get("dur_ns")
                tr.outputs = e.get("outputs")
                tr.error = e.get("error")
            elif now - ts > stale_s * 1e9:
                tr.status = ORPHANED
            if tag is not None and tag not in tr.tags:
                continue
            if label is not None and label not in tr.label_set():
                continue
            if name is not None and tr.name != name:
                continue
            if status is not None and tr.status != status:
                continue
            out.append(tr)
            if limit and len(out) >= limit:
                break
        return out

    def _iter_rows(self, where: str = "", args: Iterable[Any] = (), batch: int = 5000) -> Iterator[tuple]:
        last = 0
        args = tuple(args)
        while True:
            rows = self._q(
                f"SELECT seq, ts_ns, trace_id, span_id, type, body, chain FROM events WHERE seq>? {where} "
                f"ORDER BY seq LIMIT ?", (last, *args, batch))
            if not rows:
                return
            for r in rows:
                yield r
            last = rows[-1][0]

    def unfinished_spans(self, stale_s: float = 3600) -> List[Dict[str, Any]]:
        rows = self._q("""
            SELECT s.trace_id, s.span_id, s.ts_ns, s.body,
                   EXISTS(SELECT 1 FROM events t WHERE t.type='trace_end' AND t.trace_id=s.trace_id)
            FROM events s
            WHERE s.type='span_start'
              AND NOT EXISTS (SELECT 1 FROM events e
                              WHERE e.type='span_end' AND e.trace_id=s.trace_id AND e.span_id=s.span_id)
            ORDER BY s.seq""")
        now = time.time_ns()
        out = []
        for tid, sid, ts, body_s, ended in rows:
            if ended or now - ts > stale_s * 1e9:
                b = json.loads(body_s)
                out.append({"trace_id": tid, "span_id": sid, "ts_ns": ts, "kind": b.get("kind"),
                            "name": b.get("name"), "trace_ended": bool(ended)})
        return out

    # -- integrity ---------------------------------------------------------- #
    def verify(self, *, stale_s: float = 3600, check_blobs: bool = True) -> VerifyReport:
        prev = GENESIS
        n = 0
        problems: List[str] = []
        seen_refs: Dict[str, bool] = {}
        for seq, ts, tid, sid, typ, body_s, chain in self._iter_rows():
            n += 1
            if _chain(prev, ts, typ, tid, sid, body_s) != chain:
                problems.append(f"chain break at seq {seq} (type={typ}, trace={tid})")
            prev = chain
            if check_blobs:
                for ref in _BLOB_RE.findall(body_s):
                    if ref in seen_refs:
                        continue
                    ok = True
                    try:
                        if not self.blobs.has(ref):
                            ok = False
                            problems.append(f"missing blob {ref[:19]}… (seq {seq})")
                        else:
                            self.blobs.get(ref)
                    except BlobCorrupt:
                        ok = False
                        problems.append(f"corrupt blob {ref[:19]}… (seq {seq})")
                    seen_refs[ref] = ok
        return VerifyReport(ok=not problems, events=n, head=prev, problems=problems,
                            unfinished=self.unfinished_spans(stale_s), rotation=self.rotation_info())

    # -- rotation ----------------------------------------------------------- #
    def rotation_info(self) -> Optional[Dict[str, Any]]:
        r = self._q("SELECT body FROM events WHERE type='rotation' ORDER BY seq LIMIT 1")
        return json.loads(r[0][0]) if r else None

    def rotate_to(self, new_path: str | os.PathLike, *, keep_s: Optional[float] = None,
                  keep_labels: Iterable[str] = ("golden",), now_ns: Optional[int] = None) -> Dict[str, Any]:
        """Start a fresh store at new_path (same key, so hashes/redaction stay comparable) carrying forward: traces
        newer than keep_s, traces labelled golden/incident*, and traces with unfinished spans. The first event of
        the new store records this store's final chain head, so the two files stay linked. This store is left
        untouched (archive it yourself). Run offline: a live writer would keep appending to the old file."""
        new_path = os.path.expanduser(str(new_path))
        if os.path.abspath(new_path) == os.path.abspath(self.path):
            raise ValueError("rotate target must be a different file")
        now = time.time_ns() if now_ns is None else now_ns
        cutoff = 0 if keep_s is None else now - int(keep_s * 1e9)
        keep_labels = set(keep_labels)
        pinned = {u["trace_id"] for u in self.unfinished_spans(stale_s=0)}
        keep: List[str] = []
        for h in self.list_traces(newest_first=False):
            labels = h.label_set()
            if (h.t_start_ns >= cutoff or labels & keep_labels or h.trace_id in pinned
                    or any(l.startswith("incident") for l in labels)):
                keep.append(h.trace_id)
        new = Store(new_path, key=self.key, redact=self.redact_enabled, max_inline=self.max_inline,
                    extra_patterns=self._extra)
        try:
            if new.count_events():
                raise ValueError(f"{new_path} is not empty")
            info = {"prev_path": os.path.basename(self.path), "prev_head": self.head(),
                    "prev_events": self.count_events(), "cutoff_ns": cutoff, "kept_traces": len(keep),
                    "rotated_ns": now}
            new.append("rotation", "_system", None, info)
            buf = io.StringIO()
            self.export_jsonl(buf, keep)
            buf.seek(0)
            ev, tr, bl = new.import_jsonl(buf)
            info.update({"imported_events": ev, "imported_traces": tr, "imported_blobs": bl,
                         "dropped_traces": len(self.list_traces()) - len(keep), "new_head": new.head()})
        finally:
            new.close()
        return info

    # -- export / import ---------------------------------------------------- #
    def export_jsonl(self, fp: Any, trace_ids: Optional[Iterable[str]] = None) -> int:
        ids = list(trace_ids) if trace_ids is not None else [r[0] for r in self._q(
            "SELECT trace_id FROM events WHERE type='trace_start' ORDER BY seq")]
        refs: set = set()
        n = 0
        for tid in ids:
            for seq, ts, sid, typ, body_s in self._q(
                    "SELECT seq, ts_ns, span_id, type, body FROM events WHERE trace_id=? ORDER BY seq", (tid,)):
                fp.write(json.dumps({"type": typ, "trace_id": tid, "span_id": sid, "ts_ns": ts,
                                     "body": json.loads(body_s)}, ensure_ascii=False) + "\n")
                refs.update(_BLOB_RE.findall(body_s))
                n += 1
        for ref in sorted(refs):
            if self.blobs.has(ref):
                fp.write(json.dumps({"type": "blob", "ref": ref,
                                     "b64": base64.b64encode(self.blobs.get(ref)).decode()}) + "\n")
        return n

    def import_jsonl(self, fp: Any) -> Tuple[int, int, int]:
        """Idempotent per trace: traces already present are skipped. Returns (events, traces, blobs)."""
        existing = {r[0] for r in self._q("SELECT trace_id FROM events WHERE type='trace_start'")}
        skip: set = set()
        ev = tr = bl = 0
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["type"] == "blob":
                data = base64.b64decode(rec["b64"])
                if blob_ref(data) != rec["ref"]:
                    raise BlobCorrupt(rec["ref"])
                self.blobs.put(data)
                bl += 1
                continue
            tid = rec["trace_id"]
            if tid in existing or tid in skip:
                skip.add(tid)
                continue
            if rec["type"] == "trace_start":
                tr += 1
            self.append(rec["type"], tid, rec.get("span_id"), rec["body"], ts_ns=rec["ts_ns"])
            ev += 1
        return ev, tr, bl


def _fold(trace_id: str, rows: List[tuple], stale_s: float) -> Trace:
    tr: Optional[Trace] = None
    spans: Dict[str, Span] = {}
    order: List[str] = []
    for seq, ts, sid, typ, body_s in rows:
        b = json.loads(body_s)
        if typ == "trace_start":
            tr = Trace(trace_id=trace_id, name=b["name"], t_start_ns=ts, inputs=b.get("inputs"),
                       tags=list(b.get("tags", [])), env=b.get("env", {}), meta=dict(b.get("meta", {})),
                       schema=b.get("schema", SCHEMA_VERSION))
        elif typ == "span_start":
            spans[sid] = Span(span_id=sid, trace_id=trace_id, parent_id=b.get("parent_id"), kind=b["kind"],
                              name=b["name"], t_start_ns=ts, inputs=b.get("inputs"), meta=dict(b.get("meta", {})),
                              input_hash=b.get("input_hash", ""), status=RUNNING, seq=seq)
            order.append(sid)
        elif typ == "span_annot":
            if sid in spans:
                spans[sid].meta.update(b.get("meta", {}))
        elif typ == "span_end":
            s = spans.get(sid)
            if s is not None:
                s.outputs = b.get("outputs")
                s.status = b.get("status", OK)
                s.error = b.get("error")
                s.dur_ns = b.get("dur_ns")
        elif typ == "trace_end" and tr is not None:
            tr.outputs = b.get("outputs")
            tr.status = b.get("status", OK)
            tr.error = b.get("error")
            tr.dur_ns = b.get("dur_ns")
            if "divergences" in b:
                tr.meta["divergences"] = b["divergences"]
        elif typ == "label" and tr is not None:
            b["ts_ns"] = ts
            tr.labels.append(b)
    if tr is None:
        raise ValueError(f"trace {trace_id} has no trace_start event")
    ended = tr.status != RUNNING
    now = time.time_ns()
    for sid in order:
        s = spans[sid]
        if s.status == RUNNING and (ended or now - s.t_start_ns > stale_s * 1e9):
            s.status = ORPHANED
    if tr.status == RUNNING and now - tr.t_start_ns > stale_s * 1e9:
        tr.status = ORPHANED
    tr.spans = [spans[i] for i in order]
    return tr
