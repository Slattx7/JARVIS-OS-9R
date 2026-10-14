"""ADB backend (host -> phone). Used for the 14 Pro (wireless debugging / Shizuku `rish`) and the 9R.

Everything the device shell sees is shlex-quoted. Screen text is untrusted by construction.
Shizuku: pass shell_prefix=["sh","/data/local/tmp/rish"] + env={"RISH_APPLICATION_ID": "<pkg>"}.
That launcher path is an ASSUMPTION -- verify on the 14 Pro; plain adb shell is equivalent for the
input/monkey/uiautomator calls below.
"""
from __future__ import annotations
import re, shlex, xml.etree.ElementTree as ET
from typing import Optional, Sequence

from ..runner import Runner, RunResult

PKG_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+")
KEYNAME_RE = re.compile(r"KEYCODE_[A-Z0-9_]+")
NUM_RE = re.compile(r"\+?[0-9]{3,15}")
MAX_XML = 1_000_000


class AdbError(RuntimeError):
    pass


class Adb:
    def __init__(self, runner: Runner, serial: Optional[str] = None, adb_bin: str = "adb",
                 shell_prefix: Optional[Sequence[str]] = None, env: Optional[dict] = None):
        self.r, self.serial, self.bin = runner, serial, adb_bin
        self.prefix, self.env = list(shell_prefix or []), dict(env or {})

    def _base(self):
        return [self.bin] + (["-s", self.serial] if self.serial else [])

    def _wrap(self, argv: Sequence[str]) -> str:
        cmd = " ".join(shlex.quote(x) for x in argv)
        if self.prefix:
            envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
            cmd = f"{envs} {' '.join(shlex.quote(p) for p in self.prefix)} -c {shlex.quote(cmd)}".strip()
        return cmd

    def shell(self, argv: Sequence[str], timeout=30.0) -> RunResult:
        return self.r.run(self._base() + ["shell", self._wrap(argv)], timeout=timeout)

    def shell_raw(self, cmd: str, timeout=30.0) -> RunResult:
        """cmd is interpreted by the device shell (pipes etc.). Only for fixed, vetted strings or T3 paths."""
        return self.r.run(self._base() + ["shell", cmd], timeout=timeout)

    def _check(self, res: RunResult, what: str) -> str:
        if not res.ok:
            raise AdbError(f"{what} failed rc={res.returncode}: {res.stderr.strip()[:200]}")
        return res.stdout

    # ---- input
    def tap(self, x: int, y: int):
        return self._check(self.shell(["input", "tap", str(int(x)), str(int(y))]), "tap")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300):
        return self._check(self.shell(["input", "swipe", *map(lambda v: str(int(v)), (x1, y1, x2, y2, ms))]), "swipe")

    def text(self, s: str):
        if not s.isascii(): raise ValueError("input text supports ASCII only")
        return self._check(self.shell(["input", "text", s.replace(" ", "%s")]), "text")

    def keyevent(self, key: str):
        if not (key.isdigit() or KEYNAME_RE.fullmatch(key)): raise ValueError("bad keycode")
        return self._check(self.shell(["input", "keyevent", key]), "keyevent")

    # ---- apps / intents
    def open_app(self, package: str):
        if not PKG_RE.fullmatch(package): raise ValueError("bad package")
        return self._check(self.shell(["monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"]), "open_app")

    def open_url(self, url: str):
        if not re.fullmatch(r"https?://[^\s]+", url): raise ValueError("only http(s) urls")
        return self._check(self.shell(["am", "start", "-a", "android.intent.action.VIEW", "-d", url]), "open_url")

    def compose_sms(self, number: str, body: str):
        if not NUM_RE.fullmatch(number): raise ValueError("bad number")
        return self._check(self.shell(["am", "start", "-a", "android.intent.action.SENDTO", "-d", f"sms:{number}",
                                       "--es", "sms_body", body]), "compose_sms")

    def dial(self, number: str):
        if not NUM_RE.fullmatch(number): raise ValueError("bad number")
        return self._check(self.shell(["am", "start", "-a", "android.intent.action.DIAL", "-d", f"tel:{number}"]), "dial")

    def install(self, apk_path: str):
        return self._check(self.r.run(self._base() + ["install", "-r", apk_path], timeout=180), "install")

    def uninstall(self, package: str):
        if not PKG_RE.fullmatch(package): raise ValueError("bad package")
        return self._check(self.r.run(self._base() + ["uninstall", package], timeout=60), "uninstall")

    def foreground(self) -> str:
        res = self.shell_raw("dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'")
        return res.stdout.strip()

    # ---- screen
    def dump_tree(self) -> ET.Element:
        res = self.r.run(self._base() + ["exec-out", "uiautomator", "dump", "/dev/tty"], timeout=30)
        xml = self._check(res, "uiautomator")
        i, j = xml.find("<?xml"), xml.rfind("</hierarchy>")
        if i < 0 or j < 0: raise AdbError("no hierarchy in uiautomator output")
        blob = xml[i:j + len("</hierarchy>")]
        if len(blob) > MAX_XML: raise AdbError("hierarchy too large")
        return ET.fromstring(blob)

    @staticmethod
    def find_node(tree: ET.Element, text_or_desc: Optional[str] = None, class_name: Optional[str] = None,
                  resource_id: Optional[str] = None) -> Optional[ET.Element]:
        if tree is None: return None
        for n in tree.iter("node"):
            t = (n.get("text") or "").strip().lower()
            d = (n.get("content-desc") or "").strip().lower()
            c = n.get("class") or ""
            r = n.get("resource-id") or ""
            if resource_id and resource_id.lower() in r.lower():
                return n
            if text_or_desc and (text_or_desc.lower() == t or text_or_desc.lower() == d or text_or_desc.lower() in d):
                return n
            if class_name and class_name in c:
                return n
        return None

    @staticmethod
    def node_center(node: ET.Element) -> tuple[int, int]:
        m = re.findall(r"\[(\d+),(\d+)\]", node.get("bounds") or "")
        if len(m) == 2:
            (x1, y1), (x2, y2) = map(int, m[0]), map(int, m[1])
            return (x1 + x2) // 2, (y1 + y2) // 2
        return 0, 0

    def screen_nodes(self) -> list:
        root = self.dump_tree()
        out = []
        for n in root.iter("node"):
            for attr in ("text", "content-desc"):
                v = (n.get(attr) or "").strip()
                if v: out.append(v)
        return out

    def screen_text(self) -> str:
        return "\n".join(self.screen_nodes())

    def screenshot(self, dest: str):
        res = self.r.run(self._base() + ["exec-out", "screencap", "-p"], timeout=30)
        if not res.ok or not res.out.startswith(b"\x89PNG"): raise AdbError("screencap failed")
        with open(dest, "wb") as f: f.write(res.out)
        return dest

    # ---- composite high-level app actions
    def instagram_dm(self, username: str, message: str) -> dict:
        import time
        user = username.strip().lstrip("@")
        if not re.fullmatch(r"[A-Za-z0-9_.]+", user):
            raise ValueError(f"invalid username: {user}")

        # Check if already inside the direct message thread
        tree = self.dump_tree()
        inp = self.find_node(tree, class_name="android.widget.EditText")

        # If not already in chat, deep link to profile and tap Message
        if inp is None:
            self.shell(["am", "start", "-a", "android.intent.action.VIEW", "-d", f"instagram://user?username={user}"])
            btn = None
            t0 = time.time()
            while time.time() - t0 < 8.0:
                time.sleep(1.2)
                tree = self.dump_tree()
                inp = self.find_node(tree, class_name="android.widget.EditText")
                if inp is not None:
                    break
                btn = self.find_node(tree, text_or_desc="Message")
                if btn is not None:
                    bx, by = self.node_center(btn)
                    self.tap(bx, by)
                    break

            if inp is None and btn is not None:
                t0 = time.time()
                while time.time() - t0 < 8.0:
                    time.sleep(1.0)
                    tree = self.dump_tree()
                    inp = self.find_node(tree, class_name="android.widget.EditText")
                    if inp is not None:
                        break

        if inp is None:
            raise AdbError(f"could not find message input field in chat with {user}")

        ix, iy = self.node_center(inp)
        self.tap(ix, iy)
        time.sleep(0.4)
        self.text(message)
        time.sleep(0.8)

        tree3 = self.dump_tree()
        send_btn = self.find_node(tree3, text_or_desc="Send") or \
                   self.find_node(tree3, resource_id="row_thread_composer_send_button")
        if send_btn is not None:
            sx, sy = self.node_center(send_btn)
            self.tap(sx, sy)
        else:
            self.tap(964, 1394)
            self.keyevent("KEYCODE_ENTER")

        return {"status": "sent", "platform": "instagram", "recipient": user, "message": message}

    def whatsapp_message(self, recipient: str, message: str) -> dict:
        import time, urllib.parse
        encoded = urllib.parse.quote(message)
        clean = re.sub(r"[^\d+]", "", recipient)
        if clean and len(clean) >= 10:
            url = f"https://api.whatsapp.com/send?phone={clean}&text={encoded}"
            self.shell(["am", "start", "-a", "android.intent.action.VIEW", "-d", url])
        else:
            url = f"whatsapp://send?text={encoded}"
            self.shell(["am", "start", "-a", "android.intent.action.VIEW", "-d", url])
        time.sleep(2.0)
        tree = self.dump_tree()
        send_btn = self.find_node(tree, text_or_desc="Send")
        if send_btn:
            sx, sy = self.node_center(send_btn)
            self.tap(sx, sy)
        return {"status": "sent", "platform": "whatsapp", "recipient": recipient, "message": message}

    def youtube_play(self, query: str) -> dict:
        self._check(self.shell(["am", "start", "-a", "android.media.action.MEDIA_PLAY_FROM_SEARCH", "-e", "query", query, "com.google.android.youtube"]), "youtube_play")
        return {"status": "playing", "platform": "youtube", "query": query}

    def spotify_play(self, query: str) -> dict:
        self._check(self.shell(["am", "start", "-a", "android.media.action.MEDIA_PLAY_FROM_SEARCH", "-e", "query", query, "com.spotify.music"]), "spotify_play")
        return {"status": "playing", "platform": "spotify", "query": query}

    def click_label(self, label: str) -> dict:
        tree = self.dump_tree()
        node = self.find_node(tree, text_or_desc=label)
        if node is None and tree is not None:
            lbl = label.strip().lower()
            for n in tree.iter("node"):
                t = (n.get("text") or "").strip().lower()
                d = (n.get("content-desc") or "").strip().lower()
                if lbl in t or lbl in d:
                    node = n
                    break
        if node is None:
            raise AdbError(f"could not find element matching '{label}' on screen")
        x, y = self.node_center(node)
        self.tap(x, y)
        return {"status": "clicked", "label": label, "x": x, "y": y}

    def screen_elements(self) -> list:
        tree = self.dump_tree()
        elements = []
        for n in (tree.iter("node") if tree is not None else []):
            t = (n.get("text") or "").strip()
            d = (n.get("content-desc") or "").strip()
            c = n.get("class") or ""
            if t or d or "edit" in c.lower() or "button" in c.lower():
                cx, cy = self.node_center(n)
                elements.append({"label": t or d, "type": c.split(".")[-1], "pos": [cx, cy]})
        return elements

    def battery_status(self) -> dict:
        out = self._check(self.shell_raw("dumpsys battery"), "dumpsys battery")
        res = {}
        for line in out.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k in ("level", "temperature", "voltage", "status", "health", "present"):
                    res[k] = int(v) if v.isdigit() else v
        if "temperature" in res and isinstance(res["temperature"], (int, float)):
            res["temp_c"] = round(res["temperature"] / 10.0, 1)
        if not res:
            raise AdbError("failed to parse battery status from dumpsys battery")
        return res

    def close_app(self, package: str) -> dict:
        if not PKG_RE.fullmatch(package): raise ValueError("bad package")
        self._check(self.shell(["am", "force-stop", package]), "close_app")
        return {"status": "stopped", "package": package}

    def list_user_apps(self) -> list:
        out = self._check(self.shell_raw("pm list packages -3"), "pm list packages")
        pkgs = [l.replace("package:", "").strip() for l in out.splitlines() if l.strip()]
        return sorted(pkgs)

    def wake_screen(self) -> dict:
        self._check(self.shell(["input", "keyevent", "KEYCODE_WAKEUP"]), "wake_screen")
        return {"status": "awake"}

    def device_home(self) -> dict:
        self._check(self.shell(["input", "keyevent", "KEYCODE_HOME"]), "device_home")
        return {"status": "home"}

    def device_back(self) -> dict:
        self._check(self.shell(["input", "keyevent", "KEYCODE_BACK"]), "device_back")
        return {"status": "back"}
