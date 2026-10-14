"""On-device capture daemon for a rooted Android phone. Stdlib-only.

Run as root from a Magisk/KernelSU service.d script (robust: not an app process, so not frozen/killed by the
app freezer) or from Termux via `su -c` for testing.

Sensors:
  focus + screen  `dumpsys window|power` polled; optionally triggered instantly by logcat event-buffer tags
  notifications   `dumpsys notification --noredact` diffed against a persisted seen-set

Every event passes privacy.apply_policy BEFORE it is queued (blocked apps never touch disk). Events wait in a
local SQLite outbox until the server acks them; acked payloads are blanked immediately.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import logging.handlers
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from .. import privacy, schema
from . import parsers

log = logging.getLogger("recall.capture")

FOCUS_CMDS = (
    "dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'",
    "dumpsys activity activities | grep -E 'topResumedActivity|ResumedActivity'",
)
POWER_CMD = "dumpsys power | grep -E 'mWakefulness='"
POWER_MARK = "===POWER==="
NOTIF_CMD = "dumpsys notification --noredact"
LOGCAT_CMD = "logcat -b events -T 1 -s notification_enqueue wm_set_resumed_activity"
DEFAULT_CONFIG_PATH = "/data/local/tmp/recall/device.toml"

DEFAULTS: dict[str, Any] = {
    "device_id": "9r",
    "db_path": "/data/local/tmp/recall/outbox.db",
    "server": {"url": "http://127.0.0.1:8787", "token": "", "batch": 200, "timeout_s": 15, "gzip": True},
    "poll": {"focus_s": 2.0, "notif_s": 8.0, "idle_mult": 3, "heartbeat_s": 300},
    "capture": {"emit_initial": True, "skip_ongoing": True, "skip_group_summary": True,
                "ignore_pkgs": ["com.android.systemui"], "logcat_triggers": True},
    "privacy": {"block": [], "meta": [], "allow": [], "redact": True},
    "limits": {"max_unsent": 200000, "keep_sent": 5000},
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    if overrides:
        _merge(cfg, overrides)
    return cfg


class Priv:
    """Runs shell commands with root: directly if already uid 0, else through `su -c`."""

    def __init__(self, use_su: bool | None = None):
        self.use_su = (os.geteuid() != 0) if use_su is None else use_su

    def _argv(self, cmd: str) -> list[str]:
        return ["su", "-c", cmd] if self.use_su else ["sh", "-c", cmd]

    def run(self, cmd: str, timeout: float = 20.0) -> str | None:
        try:
            p = subprocess.run(self._argv(cmd), capture_output=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as e:
            log.warning("command failed (%s): %s", type(e).__name__, cmd[:70])
            return None
        if p.stderr:
            log.debug("stderr from %r: %s", cmd[:40], p.stderr.decode("utf-8", "replace")[:200])
        return p.stdout.decode("utf-8", "replace")

    def stream(self, cmd: str) -> subprocess.Popen[str]:
        return subprocess.Popen(self._argv(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, errors="replace", bufsize=1)


_OUTBOX_DDL = """
PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS outbox(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
  payload TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0, note TEXT);
CREATE INDEX IF NOT EXISTS ix_outbox_sent ON outbox(sent, seq);
CREATE TABLE IF NOT EXISTS notif_seen(key TEXT PRIMARY KEY, chash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""


class Outbox:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(_OUTBOX_DDL)

    def put(self, ev: dict[str, Any]) -> bool:
        payload = json.dumps(ev, separators=(",", ":"), ensure_ascii=False)
        with self.lock:
            return self.db.execute("INSERT OR IGNORE INTO outbox(id,payload) VALUES(?,?)", (ev["id"], payload)).rowcount == 1

    def pending(self, limit: int) -> list[tuple[str, str]]:
        with self.lock:
            return self.db.execute("SELECT id,payload FROM outbox WHERE sent=0 ORDER BY seq LIMIT ?", (limit,)).fetchall()

    def done(self, ids: Iterable[str], note: str | None = None) -> None:
        ids = [i for i in ids if isinstance(i, str)]
        with self.lock:
            for i in range(0, len(ids), 400):
                chunk = ids[i:i + 400]
                self.db.execute(f"UPDATE outbox SET sent=1, payload='', note=? WHERE id IN ({','.join('?' * len(chunk))})",
                                (note, *chunk))

    def stats(self) -> dict[str, int]:
        with self.lock:
            p = self.db.execute("SELECT COUNT(*) FROM outbox WHERE sent=0").fetchone()[0]
            s = self.db.execute("SELECT COUNT(*) FROM outbox WHERE sent=1").fetchone()[0]
        return {"pending": p, "sent": s}

    def prune(self, keep_sent: int, max_unsent: int) -> int:
        with self.lock:
            n = self.db.execute("DELETE FROM outbox WHERE sent=1 AND seq <= "
                                "(SELECT COALESCE(MAX(seq),0) FROM outbox WHERE sent=1) - ?", (keep_sent,)).rowcount
            over = self.db.execute("SELECT COUNT(*) FROM outbox WHERE sent=0").fetchone()[0] - max_unsent
            if over > 0:
                n += self.db.execute("DELETE FROM outbox WHERE seq IN "
                                     "(SELECT seq FROM outbox WHERE sent=0 ORDER BY seq LIMIT ?)", (over,)).rowcount
            return n

    def seen_load(self) -> dict[str, str]:
        with self.lock:
            return dict(self.db.execute("SELECT key,chash FROM notif_seen").fetchall())

    def seen_save(self, upserts: dict[str, str], deletes: Iterable[str]) -> None:
        with self.lock:
            self.db.execute("BEGIN")
            self.db.executemany("INSERT INTO notif_seen(key,chash) VALUES(?,?) "
                                "ON CONFLICT(key) DO UPDATE SET chash=excluded.chash", list(upserts.items()))
            self.db.executemany("DELETE FROM notif_seen WHERE key=?", [(k,) for k in deletes])
            self.db.execute("COMMIT")

    def kv_get(self, k: str) -> str | None:
        with self.lock:
            r = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def kv_set(self, k: str, v: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))


class UploadError(Exception):
    def __init__(self, msg: str, backoff: float):
        super().__init__(msg)
        self.backoff = backoff


class Uploader:
    """Ships outbox batches to the server. Idempotent: event ids are stable, so retries are safe."""

    def __init__(self, outbox: Outbox, cfg: dict[str, Any]):
        s = cfg["server"]
        self.outbox, self.device = outbox, cfg["device_id"]
        self.base = s["url"].rstrip("/")
        self.token, self.batch = s["token"], int(s["batch"])
        self.timeout, self.gzip = float(s["timeout_s"]), bool(s["gzip"])
        self.wake = threading.Event()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            if self.gzip and len(body) > 512:
                body, headers["Content-Encoding"] = gzip.compress(body, 5), "gzip"
        req = urllib.request.Request(self.base + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise UploadError(f"auth rejected ({e.code}); check [server].token", 300) from None
            if e.code == 413:
                self.batch = max(1, self.batch // 2)
                raise UploadError(f"payload too large; batch now {self.batch}", 1) from None
            raise UploadError(f"http {e.code}", 30) from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise UploadError(f"{type(e).__name__}: {e}", 30) from None

    def flush_once(self) -> int:
        rows = self.outbox.pending(self.batch)
        if not rows:
            return 0
        resp = self._request("POST", "/v1/events", {"device": self.device, "events": [json.loads(p) for _, p in rows]})
        acked = [*resp.get("accepted", []), *resp.get("duplicates", [])]
        self.outbox.done(acked)
        for r in resp.get("rejected", []):
            self.outbox.done([r.get("id")], note=str(r.get("reason"))[:200])  # permanent failure: don't retry forever
        return len(acked) + len(resp.get("rejected", []))

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                n = self.flush_once()
            except UploadError as e:
                log.warning("upload: %s (retry in %ss)", e, e.backoff)
                stop.wait(e.backoff)
                continue
            if n == 0:
                self.wake.wait(5.0)
                self.wake.clear()
                stop.wait(0.5)  # coalesce bursts into one batch


class Collector:
    def __init__(self, cfg: dict[str, Any], priv: Any, outbox: Outbox, wake: threading.Event | None = None):
        self.cfg, self.priv, self.outbox = cfg, priv, outbox
        self.wake = wake or threading.Event()
        self.device = cfg["device_id"]
        self.deny = privacy.Denylist.from_config(cfg["privacy"])
        self.redact = bool(cfg["privacy"]["redact"])
        self.ignore = set(cfg["capture"]["ignore_pkgs"])
        self.screen: str | None = None
        self.fg: str | None = None
        self.seen = outbox.seen_load()
        self.focus_idx, self._focus_miss = 0, 0
        self._lock = threading.Lock()
        self._pending: dict[str, float] = {}
        self._kick = threading.Event()

    def _emit(self, etype: str, app: str, data: dict[str, Any], key: str = "", chash: str = "") -> bool:
        ts = schema.now_ms()
        ev = schema.Event(schema.make_event_id(self.device, etype, key or app, ts, chash),
                          self.device, etype, ts, app, data).to_dict()
        ev = privacy.apply_policy(ev, self.deny, redact=self.redact)
        if ev is None:
            return False
        if self.outbox.put(ev):
            self.wake.set()
        return True

    def tick_focus(self) -> None:
        out = self.priv.run(f"{FOCUS_CMDS[self.focus_idx]}; echo {POWER_MARK}; {POWER_CMD}", timeout=10)
        if not out:
            return
        focus_part, _, power_part = out.partition(POWER_MARK)
        screen = parsers.parse_screen(power_part)
        if screen and screen != self.screen:
            self.screen = screen
            self._emit("screen", "android", {"state": screen}, key="screen", chash=screen)
            if screen == "off":
                self.fg = None  # re-emit app_switch on wake so the server sees a session start
        if self.screen == "off":
            return
        f = parsers.parse_focus(focus_part)
        if f is None:
            self._focus_miss += 1
            if self._focus_miss >= 3 and len(FOCUS_CMDS) > 1:
                self.focus_idx = (self.focus_idx + 1) % len(FOCUS_CMDS)
                self._focus_miss = 0
                log.warning("focus parse failing; switching to focus command #%d", self.focus_idx)
            return
        self._focus_miss = 0
        if f.pkg in self.ignore or f.pkg == self.fg:
            return
        data: dict[str, Any] = {"activity": f.activity, "user": f.user}
        if self.fg:
            data["from"] = self.fg
        self._emit("app_switch", f.pkg, data, key=f.pkg, chash=self.fg or "")
        self.fg = f.pkg

    def tick_notifications(self) -> int:
        out = self.priv.run(NOTIF_CMD, timeout=25)
        if not out or not parsers.is_valid_notification_dump(out):
            log.warning("notification dump unusable (%d bytes); skipping cycle", len(out or ""))
            return 0  # never treat a failed dump as 'everything dismissed'
        cap = self.cfg["capture"]
        first_run = self.outbox.kv_get("notif_init") is None
        emitted, active, upserts = 0, set(), {}
        for r in parsers.parse_notifications(out):
            if (cap["skip_ongoing"] and r.ongoing) or (cap["skip_group_summary"] and r.group_summary):
                continue
            active.add(r.key)
            ch = r.chash
            if self.seen.get(r.key) == ch:
                continue
            self.seen[r.key] = upserts[r.key] = ch
            if first_run and not cap["emit_initial"]:
                continue
            data = r.to_data()
            data.pop("key", None)
            data.pop("tag", None)
            if first_run:
                data["initial"] = True
            if self._emit("notification", r.pkg, data, key=r.key, chash=ch):
                emitted += 1
        gone = [k for k in self.seen if k not in active]
        for k in gone:
            del self.seen[k]
        self.outbox.seen_save(upserts, gone)
        if first_run:
            self.outbox.kv_set("notif_init", "1")
        return emitted

    def heartbeat(self) -> None:
        self._emit("heartbeat", "recall", {"screen": self.screen, "fg": self.fg}, key="hb", chash=str(time.time()))

    def kick(self, kind: str, delay: float = 0.0) -> None:
        with self._lock:
            self._pending[kind] = min(self._pending.get(kind, 1e18), time.monotonic() + delay)
        self._kick.set()

    def _due_kicks(self) -> set[str]:
        now = time.monotonic()
        with self._lock:
            due = {k for k, t in self._pending.items() if t <= now}
            for k in due:
                del self._pending[k]
        return due

    @staticmethod
    def _safe(fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception:  # a sensor bug must never kill the daemon
            log.exception("tick failed")

    def run(self, stop: threading.Event) -> None:
        poll, lim, mono = self.cfg["poll"], self.cfg["limits"], time.monotonic
        nxt = {"focus": 0.0, "notif": 0.0, "hb": 0.0}
        while not stop.is_set():
            now = mono()
            kicks = self._due_kicks()
            if "focus" in kicks or now >= nxt["focus"]:
                self._safe(self.tick_focus)
                nxt["focus"] = mono() + poll["focus_s"] * (poll["idle_mult"] if self.screen == "off" else 1)
            if "notif" in kicks or now >= nxt["notif"]:
                self._safe(self.tick_notifications)
                nxt["notif"] = mono() + poll["notif_s"]
            if now >= nxt["hb"]:
                self._safe(self.heartbeat)
                self._safe(lambda: self.outbox.prune(lim["keep_sent"], lim["max_unsent"]))
                nxt["hb"] = mono() + poll["heartbeat_s"]
            with self._lock:
                pend = min(self._pending.values(), default=1e18)
            self._kick.wait(timeout=min(max(0.05, min(min(nxt.values()), pend) - mono()), 1.0))
            self._kick.clear()


class LogcatTrigger(threading.Thread):
    """Optional low-latency triggers from the events log. Failure is harmless: polling still works."""

    def __init__(self, priv: Any, collector: Collector, stop: threading.Event):
        super().__init__(name="logcat-trigger", daemon=True)
        self.priv, self.col, self.stop = priv, collector, stop

    def handle_line(self, line: str) -> None:
        if "notification_enqueue" in line:
            self.col.kick("notif", 0.7)  # let NotificationManager finish posting before we dump
        elif "wm_set_resumed_activity" in line:
            self.col.kick("focus", 0.15)

    def run(self) -> None:
        while not self.stop.is_set():
            proc = None
            try:
                proc = self.priv.stream(LOGCAT_CMD)
                for line in proc.stdout:  # type: ignore[union-attr]
                    self.handle_line(line)
                    if self.stop.is_set():
                        break
            except Exception:
                log.exception("logcat trigger crashed")
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
            self.stop.wait(10.0)


def _healthcheck(up: Uploader) -> tuple[bool, str]:
    try:
        h = up._request("GET", "/healthz")
        r = up._request("POST", "/v1/events", {"device": up.device, "events": []})  # empty batch verifies the token
        return True, f"server ok (v{h.get('version')}), ingest token accepted ({r})"
    except UploadError as e:
        return False, str(e)


def doctor(cfg: dict[str, Any], show_content: bool = False) -> int:
    priv, fails = Priv(), 0

    def line(status: str, msg: str) -> None:
        nonlocal fails
        fails += status == "FAIL"
        print(f"[{status}] {msg}")

    ident = (priv.run("id") or "").strip()
    line("PASS" if "uid=0" in ident else "FAIL", f"root shell: {ident[:70] or 'no output (is su granted?)'}")
    for i, cmd in enumerate(FOCUS_CMDS):
        out = priv.run(f"{cmd}; echo {POWER_MARK}; {POWER_CMD}", timeout=15) or ""
        fp, _, pp = out.partition(POWER_MARK)
        f, scr = parsers.parse_focus(fp), parsers.parse_screen(pp)
        if f:
            line("PASS", f"focus cmd #{i}: {f.pkg} / {f.activity}; screen={scr}")
        else:
            line("WARN", f"focus cmd #{i}: no parse; screen={scr} (expected while the screen is off)")
    dump = priv.run(NOTIF_CMD, timeout=30) or ""
    if not parsers.is_valid_notification_dump(dump):
        line("FAIL", f"notification dump unusable ({len(dump)} bytes): {dump[:80]!r}")
    else:
        recs = parsers.parse_notifications(dump)
        n_with_text = sum(1 for r in recs if r.title or r.text)
        line("PASS" if recs and n_with_text else "WARN",
             f"notifications: {len(recs)} records, {n_with_text} with text, "
             f"{sum(r.ongoing for r in recs)} ongoing, {sum(r.group_summary for r in recs)} group summaries")
        if recs and not n_with_text:
            line("FAIL", "records parsed but no title/text: dump is probably redacted; check `--noredact` support")
        for r in recs[:3]:
            body = f"title={r.title[:40]!r} text={r.text[:40]!r}" if show_content else f"len(title)={len(r.title)} len(text)={len(r.text)}"
            print(f"       - {r.pkg} cat={r.category} {body}")
    ev = priv.run("logcat -b events -d -t 300 -s notification_enqueue wm_set_resumed_activity", timeout=15) or ""
    n = sum(1 for ln in ev.splitlines() if "notification_enqueue" in ln or "wm_set_resumed_activity" in ln)
    line("PASS" if n else "WARN", f"logcat event triggers: {n} matching lines (0 => polling only; still fine)")
    if cfg["server"]["token"]:
        ok, msg = _healthcheck(Uploader(Outbox(":memory:"), cfg))
        line("PASS" if ok else "FAIL", msg)
    else:
        line("WARN", "no [server].token configured: events will queue locally")
    print("doctor: " + ("OK" if not fails else f"{fails} failure(s)"))
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recall-capture")
    ap.add_argument("cmd", choices=["run", "doctor", "status", "once"])
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None)
    ap.add_argument("--log-file")
    ap.add_argument("--show-content", action="store_true", help="doctor: print notification text (default: lengths only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    handler: logging.Handler = (logging.handlers.RotatingFileHandler(a.log_file, maxBytes=1_000_000, backupCount=3)
                                if a.log_file else logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    if a.cmd == "doctor":
        return doctor(cfg, a.show_content)
    if a.cmd == "status":
        print(json.dumps(Outbox(cfg["db_path"]).stats()))
        return 0
    if a.cmd == "once":
        ob = Outbox(":memory:")
        col = Collector(cfg, Priv(), ob)
        col.tick_focus()
        col.tick_notifications()
        for _id, payload in ob.pending(10_000):
            print(payload)
        return 0

    outbox = Outbox(cfg["db_path"])
    wake = threading.Event()
    priv, stop = Priv(), threading.Event()
    col, up = Collector(cfg, priv, outbox, wake), Uploader(outbox, cfg)
    up.wake = wake
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    threads = [threading.Thread(target=up.run, args=(stop,), name="uploader", daemon=True)]
    if cfg["capture"]["logcat_triggers"]:
        threads.append(LogcatTrigger(priv, col, stop))
    if not cfg["server"]["token"]:
        log.warning("no server token configured: capturing to the local outbox only")
    for t in threads:
        t.start()
    log.info("capture started: device=%s root=%s outbox=%s", cfg["device_id"], not priv.use_su, outbox.stats())
    col.run(stop)
    log.info("capture stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
