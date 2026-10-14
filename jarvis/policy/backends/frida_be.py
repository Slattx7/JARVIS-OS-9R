"""Frida backend: runs ONLY vetted scripts from a hash-pinned manifest. Arbitrary JS is never accepted.

manifest.json: {"name": {"file": "hello.js", "sha256": "...", "description": "..."}}
Build/refresh with `python -m jarvis_actions frida-manifest <dir>` after reviewing scripts.
CLI flags (-D, -f/-n, -l, -q, -t) follow frida-tools; ASSUMPTION -- confirm against your installed version.
"""
from __future__ import annotations
import hashlib, json, os, re
from typing import Optional

from ..runner import Runner

TARGET_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")


class ScriptError(RuntimeError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""): h.update(b)
    return h.hexdigest()


def build_manifest(script_dir: str) -> dict:
    m = {}
    for fn in sorted(os.listdir(script_dir)):
        if fn.endswith(".js"):
            m[fn[:-3]] = {"file": fn, "sha256": sha256_file(os.path.join(script_dir, fn)), "description": ""}
    return m


class FridaScripts:
    def __init__(self, script_dir: str, manifest_path: Optional[str] = None):
        self.dir = os.path.realpath(script_dir)
        mp = manifest_path or os.path.join(self.dir, "manifest.json")
        try:
            with open(mp) as f: self.manifest = json.load(f)
        except FileNotFoundError:
            self.manifest = {}

    def names(self): return sorted(self.manifest)

    def path(self, name: str) -> str:
        ent = self.manifest.get(name)
        if not ent: raise ScriptError(f"script '{name}' not in manifest")
        p = os.path.realpath(os.path.join(self.dir, ent["file"]))
        if os.path.dirname(p) != self.dir: raise ScriptError("script escapes script dir")
        if sha256_file(p) != ent["sha256"]: raise ScriptError(f"script '{name}' hash mismatch (modified?)")
        return p


class Frida:
    def __init__(self, runner: Runner, scripts: FridaScripts, serial: Optional[str] = None,
                 frida_bin: str = "frida", mode: str = "spawn"):
        if mode not in ("spawn", "attach"): raise ValueError("mode")
        self.r, self.scripts, self.serial, self.bin, self.mode = runner, scripts, serial, frida_bin, mode

    def argv(self, target: str, script: str, duration_s: int) -> list:
        if not TARGET_RE.fullmatch(target): raise ValueError("bad target")
        p = self.scripts.path(script)
        dev = ["-D", self.serial] if self.serial else ["-U"]
        flag = "-f" if self.mode == "spawn" else "-n"
        return [self.bin, *dev, flag, target, "-l", p, "-q", "-t", str(int(duration_s))]

    def run(self, target: str, script: str, duration_s: int = 10) -> str:
        res = self.r.run(self.argv(target, script, duration_s), timeout=duration_s + 20)
        if res.timed_out: raise ScriptError("frida timed out")
        return (res.stdout + (("\n" + res.stderr) if res.stderr.strip() else "")).strip()
