from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeviceCfg:
    kind: str = "adb"                 # adb | local
    serial: Optional[str] = None
    root: bool = False                # only meaningful for adb devices you actually rooted
    shizuku: bool = False
    shizuku_prefix: tuple = ("sh", "/data/local/tmp/rish")
    shizuku_app_id: str = "com.android.shell"
    su_style: str = "-c"


@dataclass
class Config:
    devices: dict = field(default_factory=lambda: {
        "nitro": DeviceCfg(kind="local"),
        "9r": DeviceCfg(kind="adb", root=True),
        "14pro": DeviceCfg(kind="adb", shizuku=True)})
    read_roots: list = field(default_factory=lambda: [os.path.expanduser("~/jarvis")])
    scratch_root: str = os.path.expanduser("~/jarvis/scratch")
    trash_root: Optional[str] = None
    delete_roots: Optional[list] = None
    frida_dir: str = os.path.expanduser("~/jarvis/frida_scripts")
    frida_manifest: Optional[str] = None
    frida_mode: str = "spawn"
    adb_bin: str = "adb"
    frida_bin: str = "frida"
    audit_path: str = os.path.expanduser("~/jarvis/audit.jsonl")
    trace_path: Optional[str] = None
    recall_url: Optional[str] = None
    voice_timeout_s: float = 8.0
    strong_min_delay_s: float = 2.0

    @property
    def root_devices(self): return {n for n, d in self.devices.items() if d.root}
    @property
    def adb_devices(self): return tuple(n for n, d in self.devices.items() if d.kind == "adb")
    @property
    def local_devices(self): return tuple(n for n, d in self.devices.items() if d.kind == "local")


def _devices(raw: dict) -> dict:
    out = {}
    for n, v in raw.items():
        v = dict(v)
        if "shizuku_prefix" in v: v["shizuku_prefix"] = tuple(v["shizuku_prefix"])
        out[n] = DeviceCfg(**v)
    return out


def load_config(path: Optional[str]) -> Config:
    if not path: return Config()
    with open(path, "rb") as f:
        if path.endswith(".json"): raw = json.load(f)
        else:
            import tomllib; raw = tomllib.load(f)
    devs = _devices(raw.pop("devices", {})) or None
    cfg = Config(**raw) if devs is None else Config(devices=devs, **raw)
    for k in ("read_roots", "delete_roots"):
        v = getattr(cfg, k)
        if v: setattr(cfg, k, [os.path.expanduser(x) for x in v])
    for k in ("scratch_root", "trash_root", "frida_dir", "audit_path", "trace_path"):
        v = getattr(cfg, k)
        if v: setattr(cfg, k, os.path.expanduser(v))
    return cfg
