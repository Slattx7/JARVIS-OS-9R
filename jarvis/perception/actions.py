"""Action registry: every action the agent can take, with tier, strict arg validation, and an executor.

Single source of truth: the broker enforces from it, the LLM system prompt catalog is generated from it.
Executors only ever receive *validated* args and build argv lists (never shell strings from user text).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable


class Tier(IntEnum):
    T0 = 0  # read-only
    T1 = 1  # reversible / low blast radius
    T2 = 2  # state-changing or privacy-relevant: spoken confirmation
    T3 = 3  # root / arbitrary: out-of-band cryptographic approval


class ArgError(ValueError):
    pass


# ---------- validators ----------
def v_int(lo: int, hi: int):
    def f(x):
        if isinstance(x, bool):
            raise ArgError("expected integer")
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        if not isinstance(x, int):
            raise ArgError("expected integer")
        if not lo <= x <= hi:
            raise ArgError(f"out of range {lo}..{hi}")
        return x
    f.doc = f"int {lo}..{hi}"
    return f


def v_enum(*vals: str):
    def f(x):
        if x not in vals:
            raise ArgError(f"must be one of {list(vals)}")
        return x
    f.doc = "|".join(vals)
    return f


_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def v_text(maxlen: int):
    def f(x):
        if not isinstance(x, str) or not x.strip():
            raise ArgError("expected non-empty string")
        if len(x) > maxlen:
            raise ArgError(f"too long (max {maxlen})")
        if _CTRL.search(x):
            raise ArgError("control characters not allowed")
        return x.strip()
    f.doc = f"text<= {maxlen}"
    return f


_PKG = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z0-9_]+)+$")


def v_pkg():
    def f(x):
        if not isinstance(x, str) or len(x) > 255 or not _PKG.match(x):
            raise ArgError("invalid Android package name")
        return x
    f.doc = "android.package.name"
    return f


DEVICE = v_enum("pro14", "r9")
DEVNAME = {"pro14": "the OnePlus 14 Pro", "r9": "the OnePlus nine R"}


def _dn(d: str) -> str:
    return DEVNAME.get(d, d)


# ---------- exec plumbing ----------
@dataclass
class CmdResult:
    rc: int
    out: str
    err: str = ""


class Transport:
    def run(self, argv: list[str], timeout: float = 10.0) -> CmdResult:
        raise NotImplementedError


class AdbTransport(Transport):
    """Runs a device-side argv through `adb shell`, quoting every token; root wraps in `su -c`."""

    def __init__(self, adb_path: str, serial: str | None, root: bool = False, runner=subprocess.run):
        self.adb, self.serial, self.root, self._run = adb_path, serial, root, runner

    def build(self, argv: list[str]) -> list[str]:
        remote = " ".join(shlex.quote(a) for a in argv)
        if self.root:
            remote = "su -c " + shlex.quote(remote)
        cmd = [self.adb]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd + ["shell", remote]

    def run(self, argv, timeout=10.0):
        p = self._run(self.build(argv), capture_output=True, text=True, timeout=timeout)
        return CmdResult(p.returncode, p.stdout, p.stderr)


class FakeTransport(Transport):
    def __init__(self, responses: dict[str, CmdResult] | None = None):
        self.responses = responses or {}
        self.calls: list[list[str]] = []

    def run(self, argv, timeout=10.0):
        self.calls.append(argv)
        key = " ".join(argv)
        for prefix, res in self.responses.items():
            if key.startswith(prefix):
                return res
        return CmdResult(0, "")


@dataclass
class ExecResult:
    ok: bool
    output: str = ""
    data: dict = field(default_factory=dict)
    dry_run: bool = False


def fmt_duration(sec: int) -> str:
    h, rem = divmod(int(sec), 3600)
    m, sc = divmod(rem, 60)
    parts = [f"{n} {u}{'' if n == 1 else 's'}" for n, u in ((h, "hour"), (m, "minute"), (sc, "second")) if n]
    return " ".join(parts) or "0 seconds"


class TimerService:
    def __init__(self, scheduler=threading.Timer):
        self._sched = scheduler
        self._timers: dict[int, tuple[float, str, Any]] = {}
        self._n = 0
        self._lock = threading.Lock()

    def set(self, seconds: int, label: str, fire: Callable[[str], None]) -> int:
        with self._lock:
            self._n += 1
            tid = self._n

        def done():
            with self._lock:
                self._timers.pop(tid, None)
            fire(f"Timer done: {label}" if label else "Your timer is done.")

        t = self._sched(seconds, done)
        if hasattr(t, "daemon"):
            t.daemon = True
        with self._lock:
            self._timers[tid] = (time.time() + seconds, label, t)
        t.start()
        return tid

    def list(self) -> list[tuple[int, int, str]]:
        now = time.time()
        with self._lock:
            return [(i, max(0, round(end - now)), label) for i, (end, label, _) in sorted(self._timers.items())]

    def cancel_all(self) -> None:
        with self._lock:
            for _, _, t in self._timers.values():
                if hasattr(t, "cancel"):
                    t.cancel()
            self._timers.clear()


@dataclass
class ExecContext:
    cfg: Any
    transports: dict[tuple[str, bool], Transport] = field(default_factory=dict)
    say: Callable[[str], None] = lambda text: None
    timers: TimerService = field(default_factory=TimerService)
    clock: Callable[[], datetime] = datetime.now

    def transport(self, device: str, root: bool = False) -> Transport:
        key = (device, root)
        if key not in self.transports:
            dcfg = getattr(self.cfg, device)
            if root and not dcfg.root:
                raise RuntimeError(f"device {device} is not configured as rooted")
            self.transports[key] = AdbTransport(self.cfg.adb_path, dcfg.serial, root)
        return self.transports[key]


# ---------- registry ----------
@dataclass
class ActionSpec:
    name: str
    tier: Tier
    args: dict[str, tuple[Callable, bool]]   # name -> (validator, required)
    doc: str
    run: Callable[[dict, ExecContext], ExecResult]
    describe: Callable[[dict], str] = lambda a: ""
    mutating: bool = True                    # False => runs even in dry-run mode


class Registry:
    def __init__(self) -> None:
        self._specs: dict[str, ActionSpec] = {}

    def register(self, spec: ActionSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ActionSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def validate(self, name: str, args: dict) -> tuple[ActionSpec, dict]:
        spec = self.get(name)
        if spec is None:
            raise ArgError(f"unknown action {name!r}")
        if not isinstance(args, dict):
            raise ArgError("args must be an object")
        extra = set(args) - set(spec.args)
        if extra:
            raise ArgError(f"unexpected args: {sorted(extra)}")
        clean = {}
        for k, (validator, required) in spec.args.items():
            if k not in args:
                if required:
                    raise ArgError(f"missing arg {k!r}")
                continue
            try:
                clean[k] = validator(args[k])
            except ArgError as e:
                raise ArgError(f"{k}: {e}") from None
        return spec, clean

    def catalog(self) -> str:
        lines = []
        for n in self.names():
            s = self._specs[n]
            sig = ", ".join(f"{k}{'' if req else '?'}:{getattr(v, 'doc', '?')}" for k, (v, req) in s.args.items())
            note = {Tier.T2: " [needs user confirmation]", Tier.T3: " [needs out-of-band approval]"}.get(s.tier, "")
            lines.append(f"- {n}({sig}){note}: {s.doc}")
        return "\n".join(lines)


# ---------- executors ----------
def _fail(msg: str) -> ExecResult:
    return ExecResult(False, msg)


def _sh(ctx, device, argv, root=False, timeout=10.0) -> CmdResult:
    return ctx.transport(device, root).run(argv, timeout)


def x_time(args, ctx):
    now = ctx.clock()
    return ExecResult(True, now.strftime("%A %d %B %Y, %H:%M"),
                      {"hour": now.hour, "minute": now.minute, "weekday": now.strftime("%A"),
                       "month": now.strftime("%B"), "day": now.day, "year": now.year})


def x_timer_set(args, ctx):
    tid = ctx.timers.set(args["seconds"], args.get("label", ""), ctx.say)
    return ExecResult(True, f"timer {tid} set for {fmt_duration(args['seconds'])}", {"id": tid, "seconds": args["seconds"]})


def x_timer_list(args, ctx):
    ts = ctx.timers.list()
    return ExecResult(True, "; ".join(f"{label or 'timer'}: {fmt_duration(rem)} left" for _, rem, label in ts) or "no timers running",
                      {"count": len(ts)})


def x_note(args, ctx):
    p = Path(ctx.cfg.broker.notes_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(f"- {ctx.clock().isoformat(timespec='seconds')} {args['text']}\n")
    return ExecResult(True, "noted")


def x_timer_cancel(args, ctx):
    n = len(ctx.timers.list())
    ctx.timers.cancel_all()
    return ExecResult(True, f"cancelled {n} timer{'s' if n != 1 else ''}", {"count": n})


def x_volume_step(args, ctx):
    key = "KEYCODE_VOLUME_UP" if args["direction"] == "up" else "KEYCODE_VOLUME_DOWN"
    r = _sh(ctx, args["device"], ["input", "keyevent"] + [key] * args.get("steps", 2))
    return ExecResult(r.rc == 0, f"volume {args['direction']}" if r.rc == 0 else r.err.strip())


_BAT = re.compile(r"^\s*(level|status|powered|AC powered|USB powered|temperature):\s*(\S+)", re.M)
_STATUS = {"2": "charging", "3": "discharging", "4": "not charging", "5": "full"}


def x_battery(args, ctx):
    r = _sh(ctx, args["device"], ["dumpsys", "battery"])
    if r.rc != 0:
        return _fail(r.err.strip() or "dumpsys failed")
    f = dict(_BAT.findall(r.out))
    if "level" not in f:
        return _fail("could not parse battery info")
    temp = f"{int(f['temperature']) / 10:.1f}C" if f.get("temperature", "").lstrip("-").isdigit() else "?"
    status = _STATUS.get(f.get("status", ""), "unknown")
    return ExecResult(True, f"{args['device']} battery {f['level']} percent, {status}, {temp}",
                      {"device": args["device"], "level": int(f["level"]), "status": status})


def x_volume(args, ctx):
    idx = round(args["percent"] / 100 * 15)  # media stream max index is 15 on stock/LineageOS builds
    r = _sh(ctx, args["device"], ["cmd", "media_session", "volume", "--stream", "3", "--set", str(idx)])
    return ExecResult(r.rc == 0, f"volume {args['percent']} percent" if r.rc == 0 else r.err.strip())


_KEYS = {"play_pause": "KEYCODE_MEDIA_PLAY_PAUSE", "next": "KEYCODE_MEDIA_NEXT",
         "previous": "KEYCODE_MEDIA_PREVIOUS", "stop": "KEYCODE_MEDIA_STOP"}


def x_media(args, ctx):
    r = _sh(ctx, args["device"], ["input", "keyevent", _KEYS[args["command"]]])
    return ExecResult(r.rc == 0, args["command"] if r.rc == 0 else r.err.strip())


def x_notify(args, ctx):
    r = _sh(ctx, args["device"], ["cmd", "notification", "post", "-t", args.get("title", "Jarvis"), "jarvis", args["text"]])
    return ExecResult(r.rc == 0, "notification posted" if r.rc == 0 else r.err.strip())


def x_launch(args, ctx):
    r = _sh(ctx, args["device"], ["monkey", "-p", args["package"], "-c", "android.intent.category.LAUNCHER", "1"])
    ok = r.rc == 0 and "No activities found" not in r.out
    return ExecResult(ok, f"launched {args['package']}" if ok else (r.err.strip() or "app not found"))


def x_force_stop(args, ctx):
    r = _sh(ctx, args["device"], ["am", "force-stop", args["package"]])
    return ExecResult(r.rc == 0, f"stopped {args['package']}" if r.rc == 0 else r.err.strip())


def x_wifi(args, ctx):
    r = _sh(ctx, args["device"], ["svc", "wifi", "enable" if args["state"] == "on" else "disable"])
    return ExecResult(r.rc == 0, f"wifi {args['state']}" if r.rc == 0 else r.err.strip())


def x_screenshot(args, ctx):
    path = f"/sdcard/jarvis_{int(time.time())}.png"
    r = _sh(ctx, args["device"], ["screencap", "-p", path], timeout=15.0)
    return ExecResult(r.rc == 0, f"saved on device at {path}" if r.rc == 0 else r.err.strip(), {"path": path})


def x_frida_ps(args, ctx):
    serial = getattr(ctx.cfg, args["device"]).serial
    argv = [ctx.cfg.frida_ps_path] + (["-D", serial] if serial else ["-U"])
    p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    return ExecResult(p.returncode == 0, p.stdout[:1500] if p.returncode == 0 else p.stderr.strip()[:300])


def x_root_shell(args, ctx):
    r = ctx.transport("r9", root=True).run(["sh", "-c", args["command"]], timeout=30.0)
    out = (r.out + r.err).strip()
    return ExecResult(r.rc == 0, out[:1500] or f"exit {r.rc}", {"rc": r.rc})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None   # a webhook that redirects is misconfigured (or hostile): never follow it off-host


def post_json(url: str, payload: dict, timeout: float, headers: dict | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(300).decode("utf-8", "replace")


_WF_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_WF_PATH = re.compile(r"^/[A-Za-z0-9._~/-]{1,200}$")


def n8n_actions(cfg) -> list[ActionSpec]:
    """Validate the [n8n] config strictly (a typo here must fail at startup, not at 2 a.m.) and build the specs."""
    n = cfg.n8n
    specs = []
    for name, wf in n.workflows.items():
        if not _WF_NAME.match(name):
            raise ValueError(f"n8n workflow name {name!r}: use lowercase letters, digits, underscore (max 32)")
        if not isinstance(wf, dict) or "path" not in wf:
            raise ValueError(f"n8n.workflows.{name}: needs a table with at least `path`")
        extra = set(wf) - {"path", "tier", "description"}
        if extra:
            raise ValueError(f"n8n.workflows.{name}: unknown keys {sorted(extra)}")
        path, tier = wf["path"], wf.get("tier", 2)
        if not isinstance(path, str) or not _WF_PATH.match(path) or ".." in path:
            raise ValueError(f"n8n.workflows.{name}.path {path!r}: must be an absolute path like /webhook/abc123")
        if isinstance(tier, bool) or tier not in (0, 1, 2, 3):
            raise ValueError(f"n8n.workflows.{name}.tier must be 0..3")
        url = n.base_url.rstrip("/") + path
        label = wf.get("description") or f"trigger the {name} n8n workflow"

        def run(args, ctx, url=url, name=name):
            headers = {}
            if ctx.cfg.n8n.auth_header and ctx.cfg.n8n.auth_env:
                val = os.environ.get(ctx.cfg.n8n.auth_env)
                if not val:
                    return ExecResult(False, f"env var {ctx.cfg.n8n.auth_env} is not set")
                headers[ctx.cfg.n8n.auth_header] = val
            code, body = post_json(url, {"source": "jarvis", "workflow": name, "text": args.get("text", "")},
                                   ctx.cfg.n8n.timeout_s, headers)
            body = " ".join(body.split())[:300]
            return ExecResult(200 <= code < 300, body or f"http {code}", {"status": code})

        specs.append(ActionSpec(f"n8n.{name}", Tier(tier), {"text": (v_text(500), False)}, label, run,
                                (lambda a, name=name: f"run the {name.replace('_', ' ')} workflow"), mutating=tier > 0))
    return specs


def build_registry(cfg=None) -> Registry:
    R = Registry()
    if cfg is not None and cfg.n8n.enabled:
        for spec in n8n_actions(cfg):
            R.register(spec)
    add = R.register
    dev = (DEVICE, True)
    # T0 - read-only
    add(ActionSpec("host.time", Tier.T0, {}, "current local date and time", x_time, lambda a: "read the time", False))
    add(ActionSpec("host.timer.list", Tier.T0, {}, "list running timers", x_timer_list, lambda a: "list timers", False))
    add(ActionSpec("device.battery", Tier.T0, {"device": dev}, "battery level and charging state", x_battery,
                   lambda a: f"read {_dn(a['device'])} battery", False))
    # T1 - reversible
    add(ActionSpec("host.timer.set", Tier.T1, {"seconds": (v_int(1, 86400), True), "label": (v_text(60), False)},
                   "start a countdown timer; speaks when done", x_timer_set, lambda a: f"set a {a['seconds']} second timer"))
    add(ActionSpec("host.timer.cancel", Tier.T1, {}, "cancel all running timers", x_timer_cancel, lambda a: "cancel all timers"))
    add(ActionSpec("device.volume.step", Tier.T1, {"device": dev, "direction": (v_enum("up", "down"), True),
                                                    "steps": (v_int(1, 5), False)},
                   "nudge media volume up or down", x_volume_step,
                   lambda a: f"turn {_dn(a['device'])} volume {a['direction']}"))
    add(ActionSpec("host.note.append", Tier.T1, {"text": (v_text(500), True)}, "append a line to the notes file",
                   x_note, lambda a: "add a note"))
    add(ActionSpec("device.volume.set", Tier.T1, {"device": dev, "percent": (v_int(0, 100), True)},
                   "set media volume", x_volume, lambda a: f"set {_dn(a['device'])} volume to {a['percent']} percent"))
    add(ActionSpec("device.media.control", Tier.T1, {"device": dev, "command": (v_enum(*_KEYS), True)},
                   "media transport control", x_media, lambda a: f"{a['command']} media on {_dn(a['device'])}"))
    add(ActionSpec("device.notify", Tier.T1, {"device": dev, "text": (v_text(300), True), "title": (v_text(80), False)},
                   "post a notification", x_notify, lambda a: f"post a notification on {_dn(a['device'])}"))
    # T2 - confirm
    add(ActionSpec("device.app.launch", Tier.T2, {"device": dev, "package": (v_pkg(), True)}, "launch an app",
                   x_launch, lambda a: f"launch {a['package']} on {_dn(a['device'])}"))
    add(ActionSpec("device.app.force_stop", Tier.T2, {"device": dev, "package": (v_pkg(), True)}, "force-stop an app",
                   x_force_stop, lambda a: f"force stop {a['package']} on {_dn(a['device'])}"))
    add(ActionSpec("device.wifi.set", Tier.T2, {"device": dev, "state": (v_enum("on", "off"), True)}, "toggle wifi",
                   x_wifi, lambda a: f"turn wifi {a['state']} on {_dn(a['device'])}"))
    add(ActionSpec("device.screenshot", Tier.T2, {"device": dev}, "capture the screen to device storage",
                   x_screenshot, lambda a: f"take a screenshot on {_dn(a['device'])}"))
    add(ActionSpec("device.frida.ps", Tier.T2, {"device": dev}, "list processes via Frida (frida-server must be running)",
                   x_frida_ps, lambda a: f"list processes on {_dn(a['device'])} with Frida"))
    # T3 - out-of-band approval
    add(ActionSpec("root.shell", Tier.T3, {"command": (v_text(500), True)}, "run a shell command as root on the OnePlus 9R",
                   x_root_shell, lambda a: f"run as root on the OnePlus nine R: {a['command']}"))
    return R
