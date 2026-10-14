from __future__ import annotations
import json, os, time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path, timeout: float = 10.0, stale: float = 300.0):
    """O_EXCL lockfile. Stale locks (mtime older than `stale`s) are stolen. timeout=0 -> fail immediately."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - p.stat().st_mtime > stale:
                    p.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.time() >= deadline:
                raise TimeoutError(f"locked: {p}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def atomic_write(path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def append_jsonl(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
