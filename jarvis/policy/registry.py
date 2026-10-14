"""Default action set wired to real backends. build_dispatcher() is the usual entry point."""
from __future__ import annotations
import re
from typing import Optional

from .backends.adb import Adb
from .backends.frida_be import Frida, FridaScripts
from .backends.local import FsSandbox, LinuxAudio
from .backends.root import RootShell
from .config import Config
from .core import Action, Audit, Dispatcher, Registry, Tier
from .runner import Runner
from .schema import Arg
from .taint import TaintLedger
from .trace import JsonlSink

DESTRUCTIVE = re.compile(
    r"\b(rm|rmdir|dd|mkfs|chmod|chown|su|sudo|reboot|poweroff|shutdown|wipe|format|kill|killall|pkill|"
    r"mv|truncate|shred|iptables|setenforce|mount|umount|insmod|rmmod|curl|wget|nc|ncat|"
    r"pm\s+(uninstall|clear|disable)|settings\s+put|am\s+force-stop)\b")
METACHARS = re.compile(r"[;&|`$<>\n\\()]")


def shell_tier(a: Action) -> Tier:
    cmd = a.args.get("cmd", "")
    if a.args.get("root"): return Tier.T3_DANGEROUS
    if DESTRUCTIVE.search(cmd) or METACHARS.search(cmd): return Tier.T3_DANGEROUS
    return Tier.T2_STATEFUL


PKG = r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+"
NUM = r"\+?[0-9]{3,15}"


def build_registry(cfg: Config, runner: Runner, *, recall=None, fs: Optional[FsSandbox] = None,
                   frida_scripts: Optional[FridaScripts] = None, audio=None) -> Registry:
    fs = fs or FsSandbox(cfg.read_roots, cfg.scratch_root, cfg.trash_root, cfg.delete_roots)
    audio = audio or LinuxAudio(runner)
    scripts = frida_scripts or FridaScripts(cfg.frida_dir, cfg.frida_manifest)
    adbs, roots, fridas = {}, {}, {}
    for n in cfg.adb_devices:
        d = cfg.devices[n]
        kw = {}
        if d.shizuku:
            kw = dict(shell_prefix=list(d.shizuku_prefix), env={"RISH_APPLICATION_ID": d.shizuku_app_id})
        adbs[n] = Adb(runner, d.serial, cfg.adb_bin, **kw)
        if d.root: roots[n] = RootShell(Adb(runner, d.serial, cfg.adb_bin), d.su_style)
        fridas[n] = Frida(runner, scripts, d.serial, cfg.frida_bin, cfg.frida_mode)
    A, L = cfg.adb_devices, cfg.local_devices
    ALL = A + L
    R = Registry()
    adb = lambda a: adbs[a.device]

    # ---------------- T0 observe
    def recall_q(a):
        if recall is None: raise RuntimeError("recall layer not configured")
        return recall.query(a.args["q"], a.args.get("k", 5))
    R.register("recall.query", Tier.T0_OBSERVE, recall_q, summary="Query the personal recall layer",
               schema={"q": Arg(str, max_len=512), "k": Arg(int, False, min=1, max=20)}, output_untrusted=True)
    R.register("screen.read", Tier.T0_OBSERVE, lambda a: adb(a).screen_text(), devices=A,
               summary="Read on-screen text via uiautomator", output_untrusted=True)
    R.register("screen.elements", Tier.T0_OBSERVE, lambda a: adb(a).screen_elements(), devices=A,
               summary="List visible interactive UI elements with coordinates", output_untrusted=True)
    R.register("device.foreground", Tier.T0_OBSERVE, lambda a: adb(a).foreground(), devices=A,
               summary="Foreground app/window", output_untrusted=True)
    R.register("device.battery", Tier.T0_OBSERVE, lambda a: adb(a).battery_status(), devices=A,
               summary="Get battery level and charging status")
    R.register("app.list", Tier.T0_OBSERVE, lambda a: adb(a).list_user_apps(), devices=A,
               summary="List installed third-party apps")
    R.register("fs.list", Tier.T0_OBSERVE, lambda a: fs.list(a.args["path"]), devices=L,
               summary="List a directory inside allowed roots", schema={"path": Arg(str, max_len=1024)},
               output_untrusted=True)
    R.register("fs.read", Tier.T0_OBSERVE, lambda a: fs.read(a.args["path"], a.args.get("max_bytes", 65536)),
               devices=L, summary="Read a file inside allowed roots (<=64KB)",
               schema={"path": Arg(str, max_len=1024), "max_bytes": Arg(int, False, min=1, max=65536)},
               output_untrusted=True)
    R.register("frida.scripts", Tier.T0_OBSERVE, lambda a: scripts.names(), devices=A,
               summary="List vetted Frida scripts")

    # ---------------- T1 reversible
    def vol(a):
        prev = audio.get(); audio.set(a.args["level"]); return {"from": prev, "to": a.args["level"]}
    R.register("audio.volume", Tier.T1_REVERSIBLE, vol, devices=L, summary="Set host volume 0-100",
               schema={"level": Arg(int, min=0, max=100)},
               undo=lambda a, out: (lambda: audio.set(out["from"])))
    R.register("app.open", Tier.T1_REVERSIBLE, lambda a: adb(a).open_app(a.args["package"]), devices=A,
               summary="Launch an app by package", schema={"package": Arg(str, max_len=256, pattern=PKG)})
    R.register("url.open", Tier.T1_REVERSIBLE, lambda a: adb(a).open_url(a.args["url"]), devices=A,
               summary="Open an http(s) URL on the phone", schema={"url": Arg(str, max_len=2048, pattern=r"https?://\S+")})
    R.register("youtube.play", Tier.T1_REVERSIBLE, lambda a: adb(a).youtube_play(a.args["query"]), devices=A,
               summary="Search and play a video on YouTube", schema={"query": Arg(str, max_len=256)})
    R.register("spotify.play", Tier.T1_REVERSIBLE, lambda a: adb(a).spotify_play(a.args["query"]), devices=A,
               summary="Search and play a track on Spotify", schema={"query": Arg(str, max_len=256)})
    R.register("device.home", Tier.T1_REVERSIBLE, lambda a: adb(a).device_home(), devices=A,
               summary="Press the home button / navigate to home screen")
    R.register("device.back", Tier.T1_REVERSIBLE, lambda a: adb(a).device_back(), devices=A,
               summary="Press the back button")
    R.register("device.wake", Tier.T1_REVERSIBLE, lambda a: adb(a).wake_screen(), devices=A,
               summary="Wake up the device screen")
    R.register("msg.compose", Tier.T1_REVERSIBLE,
               lambda a: adb(a).compose_sms(a.args["number"], a.args["body"]), devices=A,
               summary="Open SMS compose with a draft (does NOT send)",
               schema={"number": Arg(str, pattern=NUM), "body": Arg(str, max_len=1000)})
    R.register("call.dial", Tier.T1_REVERSIBLE, lambda a: adb(a).dial(a.args["number"]), devices=A,
               summary="Open the dialer with a number (does NOT place the call)",
               schema={"number": Arg(str, pattern=NUM)})

    # undo callables are produced at execution time and keyed by action id
    _undo_box = {}
    def w_exec(a):
        _undo_box[a.id] = fs.write_scratch(a.args["name"], a.args["content"]); return a.args["name"]
    R.register("fs.write_scratch", Tier.T1_REVERSIBLE, w_exec, devices=L,
               summary="Write a file in the scratch dir (undoable)",
               schema={"name": Arg(str, max_len=128), "content": Arg(str, max_len=65536)},
               undo=lambda a, out: _undo_box.pop(a.id, None))

    # ---------------- T2 stateful
    R.register("instagram.dm", Tier.T2_STATEFUL,
               lambda a: adb(a).instagram_dm(a.args["username"], a.args["message"]), devices=A,
               summary="Send an Instagram direct message to a user",
               schema={"username": Arg(str, max_len=64), "message": Arg(str, max_len=1000)})
    R.register("whatsapp.message", Tier.T2_STATEFUL,
               lambda a: adb(a).whatsapp_message(a.args["recipient"], a.args["message"]), devices=A,
               summary="Send a WhatsApp message to a contact or phone number",
               schema={"recipient": Arg(str, max_len=64), "message": Arg(str, max_len=1000)})
    R.register("screen.click", Tier.T2_STATEFUL,
               lambda a: adb(a).click_label(a.args["label"]), devices=A,
               summary="Click an on-screen element by visible text or content description",
               schema={"label": Arg(str, max_len=128)})
    R.register("app.close", Tier.T2_STATEFUL,
               lambda a: adb(a).close_app(a.args["package"]), devices=A,
               summary="Force-stop an app by package name",
               schema={"package": Arg(str, max_len=256, pattern=PKG)})
    R.register("ui.tap", Tier.T2_STATEFUL, lambda a: adb(a).tap(a.args["x"], a.args["y"]), devices=A,
               summary="Tap screen coordinates", schema={"x": Arg(int, min=0, max=10000), "y": Arg(int, min=0, max=10000)})
    R.register("ui.swipe", Tier.T2_STATEFUL,
               lambda a: adb(a).swipe(a.args["x1"], a.args["y1"], a.args["x2"], a.args["y2"], a.args.get("ms", 300)),
               devices=A, summary="Swipe between coordinates",
               schema={**{k: Arg(int, min=0, max=10000) for k in ("x1", "y1", "x2", "y2")},
                       "ms": Arg(int, False, min=50, max=3000)})
    R.register("ui.text", Tier.T2_STATEFUL, lambda a: adb(a).text(a.args["text"]), devices=A,
               summary="Type ASCII text into the focused field", schema={"text": Arg(str, max_len=500)})
    R.register("ui.key", Tier.T2_STATEFUL, lambda a: adb(a).keyevent(a.args["key"]), devices=A,
               summary="Send a keyevent", schema={"key": Arg(str, max_len=40, pattern=r"[0-9]+|KEYCODE_[A-Z0-9_]+")})

    def install(a):
        p = fs.resolve(a.args["apk"])
        if not p.endswith(".apk"): raise ValueError("not an apk")
        return adb(a).install(p)
    R.register("app.install", Tier.T2_STATEFUL, install, devices=A, summary="Install an APK from allowed roots",
               schema={"apk": Arg(str, max_len=1024)})

    def shell(a):
        cmd, root = a.args["cmd"], a.args.get("root", False)
        if a.device in L:
            import shlex
            if shell_tier(a) == Tier.T2_STATEFUL and not root:
                res = runner.run(shlex.split(cmd), timeout=30)
            else:
                res = runner.run(["/bin/sh", "-c", cmd], timeout=30)
            return (res.stdout + res.stderr).strip()
        if root:
            if a.device not in roots: raise RuntimeError("device not rooted")
            return roots[a.device].run(cmd)
        res = adb(a).shell_raw(cmd)
        return (res.stdout + res.stderr).strip()
    R.register("shell.exec", Tier.T2_STATEFUL, shell, devices=ALL, classify=shell_tier, output_untrusted=True,
               summary="Run a shell command (T2 plain, T3 if destructive/chained/root)",
               schema={"cmd": Arg(str, max_len=2048), "root": Arg(bool, False)})

    def fs_del_tier(a):
        return Tier.T2_STATEFUL if fs.in_scratch(a.args.get("path", "")) else Tier.T3_DANGEROUS
    _dbox = {}
    def fs_del(a):
        _dbox[a.id] = fs.delete(a.args["path"]); return "moved to trash"
    R.register("fs.delete", Tier.T3_DANGEROUS, fs_del, devices=L, classify=fs_del_tier,
               summary="Delete (move to trash) inside delete roots; scratch=T2, elsewhere T3",
               schema={"path": Arg(str, max_len=1024)}, undo=lambda a, out: _dbox.pop(a.id, None))

    # ---------------- T3 dangerous
    R.register("app.uninstall", Tier.T3_DANGEROUS, lambda a: adb(a).uninstall(a.args["package"]), devices=A,
               summary="Uninstall an app", schema={"package": Arg(str, max_len=256, pattern=PKG)})
    R.register("frida.run", Tier.T3_DANGEROUS,
               lambda a: fridas[a.device].run(a.args["target"], a.args["script"], a.args.get("duration_s", 10)),
               devices=A, requires_root=True, output_untrusted=True,
               summary="Run a vetted Frida script against an app (rooted devices only)",
               schema={"target": Arg(str, max_len=256, pattern=PKG), "script": Arg(str, max_len=64, pattern=r"[A-Za-z0-9_-]+"),
                       "duration_s": Arg(int, False, min=1, max=120)})
    return R


def build_dispatcher(cfg: Config, confirm, *, runner: Optional[Runner] = None, recall=None, dry_run=False,
                     on_alert=None, extra_sinks=None, ledger: Optional[TaintLedger] = None, **kw) -> Dispatcher:
    runner = runner or Runner()
    reg = build_registry(cfg, runner, recall=recall, **kw)
    sinks = list(extra_sinks or [])
    if cfg.trace_path: sinks.append(JsonlSink(cfg.trace_path))
    audit = Audit(cfg.audit_path, sinks)
    return Dispatcher(reg, audit=audit, confirm=confirm, dry_run=dry_run, ledger=ledger or TaintLedger(),
                      on_alert=on_alert, root_devices=cfg.root_devices)
