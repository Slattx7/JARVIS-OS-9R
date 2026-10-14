"""EXPERIMENTAL Frida sensor bridge (runs on a host such as the Nitro, against frida-server on the rooted phone).

Real-time alternative to the dumpsys poller: catches notifications the instant they are enqueued, including ones
dismissed before the next poll. Use ONE capture path -- do not run this alongside `recall.device.capture run`.

    python -m recall.device.frida_bridge --config device.toml            # enumerate hook candidates only (safe)
    python -m recall.device.frida_bridge --config device.toml --hook     # install hooks (can soft-reboot the phone)

Requires `pip install frida` matching the phone's frida-server version. Host-side events go through the same
privacy policy, outbox and uploader as the on-device daemon.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Any

from .. import privacy, schema
from . import capture
from .parsers import FLAG_FOREGROUND_SERVICE, FLAG_GROUP_SUMMARY, FLAG_ONGOING

log = logging.getLogger("recall.frida")
SCRIPT = Path(__file__).parent / "frida" / "notif_hook.js"


class Normalizer:
    """Turns hook messages into policy-conformant events; dedupes re-posts with unchanged content."""

    def __init__(self, cfg: dict[str, Any]):
        self.device = cfg["device_id"]
        self.deny = privacy.Denylist.from_config(cfg["privacy"])
        self.redact = bool(cfg["privacy"]["redact"])
        self.cap = cfg["capture"]
        self.seen: dict[str, str] = {}

    def __call__(self, p: dict[str, Any], ts: int | None = None) -> dict[str, Any] | None:
        pkg = str(p.get("pkg") or "")
        if not pkg:
            return None
        flags = int(p.get("flags") or 0)
        if (self.cap["skip_ongoing"] and flags & (FLAG_ONGOING | FLAG_FOREGROUND_SERVICE)) or \
                (self.cap["skip_group_summary"] and flags & FLAG_GROUP_SUMMARY):
            return None
        key = f"{pkg}|{p.get('id')}|{p.get('tag') or ''}"
        ch = schema.content_hash(p.get("title"), p.get("text"), p.get("big_text"), p.get("sub_text"), p.get("conversation"))
        if self.seen.get(key) == ch:
            return None
        self.seen[key] = ch
        if len(self.seen) > 5000:
            self.seen.pop(next(iter(self.seen)))
        data = {k: v for k, v in {
            "id": p.get("id"), "channel": p.get("channel"), "category": p.get("category"), "title": p.get("title"),
            "text": p.get("text"), "big_text": p.get("big_text"), "sub_text": p.get("sub_text"),
            "conversation": p.get("conversation"), "template": str(p.get("template") or "").rsplit(".", 1)[-1],
            "is_group": bool(p.get("is_group"))}.items() if v not in ("", None, False)}
        ts = ts or schema.now_ms()
        ev = schema.Event(schema.make_event_id(self.device, "notification", key, ts, ch), self.device,
                          "notification", ts, pkg, data).to_dict()
        return privacy.apply_policy(ev, self.deny, redact=self.redact)


def run(cfg: dict[str, Any], hook: bool, remote: str | None) -> int:
    try:
        import frida  # type: ignore[import-not-found]
    except ImportError:
        print("frida is not installed (pip install frida; version must match the phone's frida-server)", file=sys.stderr)
        return 2
    outbox = capture.Outbox(cfg["db_path"])
    up, norm, stop = capture.Uploader(outbox, cfg), Normalizer(cfg), threading.Event()
    threading.Thread(target=up.run, args=(stop,), daemon=True, name="uploader").start()

    def on_message(msg: dict[str, Any], _data: Any) -> None:
        if msg.get("type") != "send":
            log.warning("script error: %s", msg.get("description") or msg)
            return
        p = msg["payload"]
        kind = p.get("kind")
        if kind == "notification":
            ev = norm(p)
            if ev and outbox.put(ev):
                up.wake.set()
        elif kind == "enum":
            print("hook candidates in system_server:")
            for c in p["candidates"]:
                print("  ", c)
        elif kind == "hooked":
            print(f"hooked {p['count']} overload(s); capturing. Ctrl-C to stop.")
        elif kind == "error":
            log.warning("hook: %s", p.get("msg"))

    dev = frida.get_device_manager().add_remote_device(remote) if remote else frida.get_usb_device(timeout=10)
    session = dev.attach("system_server")
    script = session.create_script(SCRIPT.read_text().replace("__ENUM_ONLY__", "false" if hook else "true"))
    script.on("message", on_message)
    script.load()
    try:
        threading.Event().wait() if hook else stop.wait(3)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        session.detach()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recall-frida")
    ap.add_argument("--config", required=True)
    ap.add_argument("--hook", action="store_true", help="install hooks (default: enumerate candidates only)")
    ap.add_argument("--remote", help="host:port of a network frida-server instead of USB")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(capture.load_config(a.config), a.hook, a.remote)


if __name__ == "__main__":
    sys.exit(main())
