"""Subprocess runner: argv lists only (never shell=True), timeouts, output caps. FakeRunner for tests."""
from __future__ import annotations
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

MAX_OUT = 256 * 1024


@dataclass
class RunResult:
    argv: list
    returncode: int
    out: bytes = b""
    err: bytes = b""
    timed_out: bool = False

    @property
    def stdout(self) -> str: return self.out.decode("utf-8", "replace")
    @property
    def stderr(self) -> str: return self.err.decode("utf-8", "replace")
    @property
    def ok(self) -> bool: return self.returncode == 0 and not self.timed_out


class Runner:
    def run(self, argv: Sequence[str], timeout: float = 30.0, input: Optional[bytes] = None,
            env: Optional[dict] = None) -> RunResult:
        try:
            p = subprocess.run(list(argv), capture_output=True, timeout=timeout, input=input, env=env, shell=False)
            return RunResult(list(argv), p.returncode, p.stdout[:MAX_OUT], p.stderr[:MAX_OUT])
        except subprocess.TimeoutExpired as e:
            return RunResult(list(argv), 124, (e.stdout or b"")[:MAX_OUT], (e.stderr or b"")[:MAX_OUT], True)
        except FileNotFoundError as e:
            return RunResult(list(argv), 127, b"", str(e).encode())


class FakeRunner(Runner):
    """handler(argv)->RunResult|str|None. Records every call."""
    def __init__(self, handler: Optional[Callable] = None, default_out: str = ""):
        self.calls: list = []
        self.handler, self.default_out = handler, default_out

    def run(self, argv, timeout=30.0, input=None, env=None) -> RunResult:
        argv = list(argv); self.calls.append(argv)
        r = self.handler(argv) if self.handler else None
        if isinstance(r, RunResult): return r
        return RunResult(argv, 0, (r if isinstance(r, str) else self.default_out).encode())
