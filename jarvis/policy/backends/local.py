"""Host-side (Nitro) backends: confined filesystem + audio. Paths are realpath-checked against roots."""
from __future__ import annotations
import os, re, shutil, time
from typing import Optional, Sequence

from ..runner import Runner

MAX_READ = 64 * 1024
MAX_LIST = 200
NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")


class FsError(RuntimeError):
    pass


def _within(path: str, root: str) -> bool:
    return os.path.commonpath([path, root]) == root


class FsSandbox:
    def __init__(self, read_roots: Sequence[str], scratch_root: str, trash_root: Optional[str] = None,
                 delete_roots: Optional[Sequence[str]] = None):
        self.read_roots = [os.path.realpath(r) for r in read_roots]
        self.scratch = os.path.realpath(scratch_root)
        self.trash = os.path.realpath(trash_root or os.path.join(self.scratch, ".trash"))
        self.delete_roots = [os.path.realpath(r) for r in (delete_roots or [self.scratch])]
        os.makedirs(self.scratch, exist_ok=True); os.makedirs(self.trash, exist_ok=True)

    def resolve(self, path: str, roots: Optional[Sequence[str]] = None) -> str:
        p = os.path.realpath(os.path.expanduser(path))
        for r in (roots or self.read_roots + [self.scratch]):
            if _within(p, r): return p
        raise FsError("path outside allowed roots")

    def in_scratch(self, path: str) -> bool:
        try: return _within(os.path.realpath(os.path.expanduser(path)), self.scratch)
        except ValueError: return False

    def list(self, path: str) -> list:
        p = self.resolve(path)
        return sorted(os.listdir(p))[:MAX_LIST]

    def read(self, path: str, max_bytes: int = MAX_READ) -> str:
        p = self.resolve(path)
        with open(p, "rb") as f: return f.read(min(max_bytes, MAX_READ)).decode("utf-8", "replace")

    def write_scratch(self, name: str, content: str):
        """returns undo callable"""
        if not NAME_RE.fullmatch(name) or name.startswith("."): raise FsError("bad scratch filename")
        p = os.path.join(self.scratch, name)
        if os.path.islink(p): raise FsError("refusing symlink")
        prev = open(p, "rb").read() if os.path.exists(p) else None
        with open(p, "w") as f: f.write(content)
        def undo():
            if prev is None: os.remove(p)
            else: open(p, "wb").write(prev)
        return undo

    def delete(self, path: str):
        """move to trash (never unlink). returns undo callable"""
        p = self.resolve(path, self.delete_roots)
        if p in self.delete_roots or p == self.trash: raise FsError("refusing to delete a root")
        dest = os.path.join(self.trash, f"{int(time.time() * 1000)}-{os.path.basename(p)}")
        shutil.move(p, dest)
        def undo():
            if os.path.exists(p): raise FsError("original path now occupied")
            shutil.move(dest, p)
        return undo


class LinuxAudio:
    """pactl preferred, amixer fallback. Volume 0-100."""
    def __init__(self, runner: Runner):
        self.r = runner

    def get(self) -> int:
        res = self.r.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
        if res.ok:
            m = re.search(r"(\d+)%", res.stdout)
            if m: return int(m.group(1))
        res = self.r.run(["amixer", "get", "Master"])
        m = re.search(r"\[(\d+)%\]", res.stdout) if res.ok else None
        if m: return int(m.group(1))
        raise RuntimeError("no audio backend (pactl/amixer)")

    def set(self, level: int):
        if not 0 <= level <= 100: raise ValueError("0-100")
        res = self.r.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"])
        if not res.ok:
            res = self.r.run(["amixer", "set", "Master", f"{level}%"])
        if not res.ok: raise RuntimeError("set volume failed")
