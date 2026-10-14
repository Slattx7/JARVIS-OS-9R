"""Root shell on the 9R via `su`. Only ever reached through T3 policy. Configurable su dialect."""
from __future__ import annotations
import shlex

from .adb import Adb, AdbError


class RootShell:
    def __init__(self, adb: Adb, su_style: str = "-c"):
        if su_style not in ("-c", "0"): raise ValueError("su_style must be '-c' or '0'")
        self.adb, self.style = adb, su_style

    def run(self, cmd: str, timeout=30.0) -> str:
        inner = shlex.quote(cmd)
        line = f"su -c {inner}" if self.style == "-c" else f"su 0 sh -c {inner}"
        res = self.adb.shell_raw(line, timeout=timeout)
        if not res.ok:
            raise AdbError(f"root exec rc={res.returncode}: {res.stderr.strip()[:200]}")
        return res.stdout
