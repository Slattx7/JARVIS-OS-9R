from __future__ import annotations
import hashlib, json, re, time
from pathlib import Path
from typing import Callable, Optional

from .refs import ModelRef
from .util import append_jsonl, atomic_write, file_lock, read_json, write_json

_VID = re.compile(r"[A-Za-z0-9._-]+")
_BIG = 64 * 1024 * 1024


def _hash_file(h, p: Path) -> None:
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        if size <= _BIG:                       # full hash for adapters; head+tail sample for multi-GB gguf
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(4 << 20))
            f.seek(size - (4 << 20))
            h.update(f.read(4 << 20))


def digest_path(artifact: str) -> Optional[str]:
    p = Path(artifact) if artifact else None
    if p is None or not p.exists():
        return None                            # e.g. a base-model id like "Qwen/..." -> nothing to hash
    h = hashlib.sha256()
    if p.is_file():
        _hash_file(h, p)
    else:
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(str(f.relative_to(p)).encode())
            _hash_file(h, f)
    return h.hexdigest()


class Registry:
    """versions/<id>/meta.json + append-only ledger (source of truth) + CURRENT pointer file (convenience).
    The live stack is replayed from the ledger, so rollback can walk back any number of promotions
    and a deleted/corrupt CURRENT is self-healing. Nothing is ever deleted."""

    def __init__(self, root, clock: Callable[[], float] = time.time):
        self.root, self.clock = Path(root), clock
        self.vdir = self.root / "versions"
        self.vdir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.root / "ledger.jsonl"
        self.cur = self.root / "CURRENT"
        self.pause_file = self.root / "PAUSE"
        self._lockp = self.root / ".lock"

    # ---- ledger
    def log(self, **ev) -> None:
        ev.setdefault("ts", self.clock())
        append_jsonl(self.ledger, ev)

    def history(self) -> list:
        if not self.ledger.exists():
            return []
        return [json.loads(l) for l in self.ledger.read_text(encoding="utf-8").splitlines() if l.strip()]

    def stack(self) -> list:
        st: list = []
        for e in self.history():
            if e["event"] in ("bootstrap", "promote"):
                st.append(e["version"])
            elif e["event"] == "rollback" and len(st) > 1:
                st.pop()
        return st

    def current(self) -> Optional[str]:
        st = self.stack()
        return st[-1] if st else None

    def _sync(self) -> None:
        atomic_write(self.cur, self.current() or "")

    # ---- versions
    def versions(self) -> list:
        return sorted(d.name for d in self.vdir.iterdir() if d.is_dir())

    def meta(self, vid: str) -> Optional[dict]:
        return read_json(self.vdir / vid / "meta.json")

    def ref(self, vid: Optional[str] = None) -> ModelRef:
        vid = vid or self.current()
        m = self.meta(vid) if vid else None
        if m is None:
            raise KeyError(vid)
        return ModelRef(vid, m.get("artifact", ""))

    def _register(self, vid: str, artifact: str, meta: Optional[dict]) -> dict:
        if not _VID.fullmatch(vid):
            raise ValueError(f"bad version id {vid!r}")
        d = self.vdir / vid
        if d.exists():
            raise FileExistsError(vid)
        m = {"artifact": artifact, "digest": digest_path(artifact), "created": self.clock(), **(meta or {})}
        d.mkdir()
        write_json(d / "meta.json", m)
        self.log(event="register", version=vid, digest=m["digest"])
        return m

    def register(self, vid: str, artifact: str, meta: Optional[dict] = None) -> dict:
        with file_lock(self._lockp):
            return self._register(vid, artifact, meta)

    def verify(self, vid: str) -> bool:
        m = self.meta(vid)
        if m is None:
            raise KeyError(vid)
        return m.get("digest") is None or digest_path(m["artifact"]) == m["digest"]

    # ---- state changes
    def bootstrap(self, vid: str, artifact: str = "", meta: Optional[dict] = None) -> dict:
        """Human-authorised initial baseline (no gate)."""
        with file_lock(self._lockp):
            if self.stack():
                raise RuntimeError("already bootstrapped")
            m = self._register(vid, artifact, meta)
            self.log(event="bootstrap", version=vid)
            self._sync()
            return m

    def promote(self, vid: str, verdict) -> None:
        if verdict.decision != "promote":
            raise PermissionError(f"gate said {verdict.decision}: {verdict.reasons}")
        with file_lock(self._lockp):
            why = self.paused()
            if why is not None:
                raise PermissionError(f"paused: {why}")
            if self.meta(vid) is None:
                raise KeyError(vid)
            if not self.verify(vid):
                raise ValueError(f"artifact digest mismatch for {vid}")
            self.log(event="promote", version=vid, prev=self.current(), stats=verdict.stats, reasons=verdict.reasons)
            self._sync()

    def rollback(self, reason: str = "") -> str:
        with file_lock(self._lockp):
            st = self.stack()
            if len(st) < 2:
                raise RuntimeError("nothing to roll back to")
            target = st[-2]
            if not self.verify(target):
                raise ValueError(f"artifact digest mismatch for rollback target {target}")
            self.log(event="rollback", version=target, from_=st[-1], reason=reason)
            self._sync()
            return target

    # ---- kill switch (blocks promotion; rollback always allowed)
    def paused(self) -> Optional[str]:
        if not self.pause_file.exists():
            return None
        return self.pause_file.read_text(encoding="utf-8").strip() or "paused"

    def pause(self, reason: str = "") -> None:
        atomic_write(self.pause_file, reason or "paused")
        self.log(event="pause", reason=reason)

    def resume(self) -> None:
        if self.pause_file.exists():
            self.pause_file.unlink()
        self.log(event="resume")
