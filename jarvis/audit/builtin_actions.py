"""Reference handlers that show the contract (snapshot before change, undo spec after) and are safe to ship:

  fs.read fs.write fs.restore fs.remove fs.delete fs.move     files confined to `roots` (realpath, symlink-safe)
  kv.get kv.set kv.unset                                      a JSON settings file
  shell.read                                                  allowlisted read-only commands, no shell, no metachars
  shell.exec                                                  argv exec (never `sh -c`), tier 3, deny-patterns in policy

Device-specific actions (adb, Frida, root shell, ...) follow the same shape: params, tier, snapshot(), handler(),
undo(). The policy tier map in policy.py already covers these ids.
Caveat: root confinement is a check-then-use on paths (TOCTOU); it is a seatbelt against a confused model, not a
sandbox against an attacker who can race the filesystem.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from .actions import ActionRegistry, ActionSpec, ArgError, Param

MAX_SNAPSHOT = 8 * 1024 * 1024
READ_ONLY_CMDS = {"ls", "cat", "head", "tail", "pwd", "id", "whoami", "uname", "df", "du", "ps", "date", "uptime",
                  "echo", "stat", "wc", "which", "getprop"}
PATH_ARG_CMDS = {"ls", "cat", "head", "tail", "stat", "wc", "du"}
_META = set(";|&><`$()\n\\")


def _atomic_write(path: str, data: bytes) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".jt-", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _Fs:
    def __init__(self, roots: Sequence[str]):
        self.roots = [os.path.realpath(r) for r in roots]

    def p(self, path: str) -> str:
        if not isinstance(path, str) or not os.path.isabs(path):
            raise ArgError("path must be absolute")
        real = os.path.realpath(path)
        if not any(real == r or real.startswith(r.rstrip(os.sep) + os.sep) for r in self.roots):
            raise PermissionError(f"{path} resolves outside the allowed roots")
        return real

    def read_bytes(self, real: str) -> Optional[bytes]:
        if not os.path.exists(real):
            return None
        if not os.path.isfile(real):
            raise ArgError(f"{real} is not a regular file")
        if os.path.getsize(real) > MAX_SNAPSHOT:
            raise ValueError(f"{real} larger than {MAX_SNAPSHOT} bytes: refusing to modify without a snapshot")
        with open(real, "rb") as f:
            return f.read()

    # handlers ------------------------------------------------------------------------------------------------ #
    def read(self, path: str, max_bytes: int = 65536) -> Dict[str, Any]:
        real = self.p(path)
        if not os.path.isfile(real):
            raise FileNotFoundError(path)
        with open(real, "rb") as f:
            data = f.read(max_bytes + 1)
        return {"status": "ok", "text": data[:max_bytes].decode("utf-8", "replace"), "truncated": len(data) > max_bytes}

    def write(self, path: str, content: Any, create_only: bool = False) -> Dict[str, Any]:
        real = self.p(path)
        existed = os.path.exists(real)
        if existed and create_only:
            raise FileExistsError(path)
        os.makedirs(os.path.dirname(real), exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        _atomic_write(real, data)
        return {"status": "ok", "path": path, "bytes": len(data), "existed": existed, "sha256": _sha(data)}

    def restore(self, path: str, snapshot: bytes) -> Dict[str, Any]:
        real = self.p(path)
        os.makedirs(os.path.dirname(real), exist_ok=True)
        _atomic_write(real, bytes(snapshot))
        return {"status": "ok", "path": path, "bytes": len(snapshot), "sha256": _sha(bytes(snapshot))}

    def remove(self, path: str, expect_sha256: Optional[str] = None) -> Dict[str, Any]:
        real = self.p(path)
        data = self.read_bytes(real)
        if data is None:
            return {"status": "ok", "path": path, "removed": False, "reason": "already absent"}
        if expect_sha256 and _sha(data) != expect_sha256:
            raise RuntimeError(f"{path} changed since it was created (sha mismatch): refusing to remove")
        os.unlink(real)
        return {"status": "ok", "path": path, "removed": True}

    def delete(self, path: str) -> Dict[str, Any]:
        real = self.p(path)
        if not os.path.isfile(real):
            raise FileNotFoundError(path)
        os.unlink(real)
        return {"status": "ok", "path": path, "removed": True}

    def move(self, src: str, dst: str) -> Dict[str, Any]:
        rs, rd = self.p(src), self.p(dst)
        if not os.path.exists(rs):
            raise FileNotFoundError(src)
        if os.path.exists(rd):
            raise FileExistsError(dst)
        os.makedirs(os.path.dirname(rd), exist_ok=True)
        os.rename(rs, rd)
        return {"status": "ok", "src": src, "dst": dst}

    # snapshots (pre-state) ------------------------------------------------------------------------------------ #
    def snap_path(self, path: str, **_: Any) -> Optional[bytes]:
        return self.read_bytes(self.p(path))


def _kv_load(path: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}


def _kv_save(path: str, d: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write(path, json.dumps(d, sort_keys=True).encode())


def _shell_argv(cmd: str) -> List[str]:
    try:
        argv = shlex.split(cmd)
    except ValueError as e:
        raise ArgError(f"cannot parse command: {e}") from e
    if not argv:
        raise ArgError("empty command")
    return argv


def build_registry(roots: Sequence[str], kv_path: str, registry: Optional[ActionRegistry] = None) -> ActionRegistry:
    reg = registry or ActionRegistry()
    fs = _Fs(roots)
    S, B = str, bool

    def undo_restore(path_key: str = "path"):
        def undo(args, result, snap_ref):
            if snap_ref:
                return {"action": "fs.restore", "args": {"path": args[path_key], "snapshot": snap_ref}}
            return {"action": "fs.remove", "args": {"path": args[path_key], "expect_sha256": result.get("sha256")}}
        return undo

    reg.register(ActionSpec("fs.read", fs.read, 0, {"path": S, "max_bytes": Param(int, False)},
                            describe=lambda a: f"read {a['path']}"))
    reg.register(ActionSpec("fs.write", fs.write, 2, {"path": S, "content": (str, bytes), "create_only": Param(B, False)},
                            describe=lambda a: f"write {len(a['content'])} bytes to {a['path']}",
                            snapshot=fs.snap_path, undo=undo_restore()))
    reg.register(ActionSpec("fs.restore", fs.restore, 2, {"path": S, "snapshot": bytes},
                            describe=lambda a: f"restore {a['path']} from a snapshot ({len(a['snapshot'])} bytes)",
                            snapshot=fs.snap_path, undo=undo_restore()))
    reg.register(ActionSpec("fs.remove", fs.remove, 2, {"path": S, "expect_sha256": Param((str, type(None)), False)},
                            describe=lambda a: f"remove {a['path']}", snapshot=fs.snap_path,
                            undo=lambda args, result, snap: ({"action": "fs.restore",
                                                              "args": {"path": args["path"], "snapshot": snap}}
                                                             if snap else None)))
    reg.register(ActionSpec("fs.delete", fs.delete, 3, {"path": S}, describe=lambda a: f"DELETE {a['path']}",
                            snapshot=fs.snap_path,
                            undo=lambda args, result, snap: {"action": "fs.restore",
                                                             "args": {"path": args["path"], "snapshot": snap}}))
    reg.register(ActionSpec("fs.move", fs.move, 2, {"src": S, "dst": S},
                            describe=lambda a: f"move {a['src']} -> {a['dst']}",
                            undo=lambda args, result, snap: {"action": "fs.move",
                                                             "args": {"src": args["dst"], "dst": args["src"]}}))

    def kv_get(key: str) -> Dict[str, Any]:
        d = _kv_load(kv_path)
        return {"status": "ok", "key": key, "found": key in d, "value": d.get(key)}

    def kv_set(key: str, value: Any) -> Dict[str, Any]:
        d = _kv_load(kv_path)
        had, prev = key in d, d.get(key)
        d[key] = value
        _kv_save(kv_path, d)
        return {"status": "ok", "key": key, "had": had, "prev": prev}

    def kv_unset(key: str) -> Dict[str, Any]:
        d = _kv_load(kv_path)
        had, prev = key in d, d.pop(key, None)
        _kv_save(kv_path, d)
        return {"status": "ok", "key": key, "had": had, "prev": prev}

    def kv_undo(args, result, snap):
        if result.get("had"):
            return {"action": "kv.set", "args": {"key": args["key"], "value": result.get("prev")}}
        return {"action": "kv.unset", "args": {"key": args["key"]}}

    reg.register(ActionSpec("kv.get", kv_get, 0, {"key": S}, describe=lambda a: f"read setting {a['key']}"))
    reg.register(ActionSpec("kv.set", kv_set, 1, {"key": S, "value": object},
                            describe=lambda a: f"set {a['key']} = {json.dumps(a['value'])[:60]}", undo=kv_undo))
    reg.register(ActionSpec("kv.unset", kv_unset, 1, {"key": S}, describe=lambda a: f"unset {a['key']}", undo=kv_undo))

    def shell_read(cmd: str) -> Dict[str, Any]:
        if any(c in _META for c in cmd):
            raise ArgError("shell metacharacters are not allowed in shell.read")
        argv = _shell_argv(cmd)
        name = os.path.basename(argv[0])
        if name not in READ_ONLY_CMDS or argv[0] != name:
            raise PermissionError(f"{argv[0]!r} is not in the read-only allowlist")
        if name in PATH_ARG_CMDS:
            for a in argv[1:]:
                if not a.startswith("-"):
                    fs.p(a if os.path.isabs(a) else os.path.join(os.getcwd(), a))
        r = subprocess.run(argv, capture_output=True, timeout=15)
        return {"status": "ok" if r.returncode == 0 else "failed", "returncode": r.returncode,
                "stdout": r.stdout[:65536].decode("utf-8", "replace"), "stderr": r.stderr[:4096].decode("utf-8", "replace")}

    def shell_exec(cmd: str, timeout_s: float = 30.0) -> Dict[str, Any]:
        argv = _shell_argv(cmd)
        r = subprocess.run(argv, capture_output=True, timeout=timeout_s)
        return {"status": "ok" if r.returncode == 0 else "failed", "returncode": r.returncode,
                "stdout": r.stdout[:65536].decode("utf-8", "replace"), "stderr": r.stderr[:4096].decode("utf-8", "replace")}

    reg.register(ActionSpec("shell.read", shell_read, 0, {"cmd": S}, describe=lambda a: f"run (read-only) {a['cmd']}"))
    reg.register(ActionSpec("shell.exec", shell_exec, 3, {"cmd": S, "timeout_s": Param((int, float), False)},
                            describe=lambda a: f"EXEC {a['cmd']}"))
    return reg
