import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jarvis_actions import Action, Source, Tier
from jarvis_actions.config import Config, DeviceCfg
from jarvis_actions.confirm import ScriptedConfirmer
from jarvis_actions.registry import build_dispatcher, build_registry
from jarvis_actions.runner import FakeRunner, RunResult
from jarvis_actions.taint import TaintLedger


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t
    def advance(self, s): self.t += s


class FakeAudio:
    def __init__(self, v=40): self.v = v
    def get(self): return self.v
    def set(self, x): self.v = x


def make_cfg(tmp):
    return Config(devices={"nitro": DeviceCfg(kind="local"), "9r": DeviceCfg(kind="adb", serial="9R", root=True),
                           "14pro": DeviceCfg(kind="adb", serial="14P", shizuku=True)},
                  read_roots=[os.path.join(tmp, "data")], scratch_root=os.path.join(tmp, "scratch"),
                  frida_dir=os.path.join(tmp, "frida"), audit_path=os.path.join(tmp, "audit.jsonl"))


def make_env(answers=None, default=False, handler=None, clock=None, recall=None):
    tmp = tempfile.mkdtemp(prefix="jaT")
    os.makedirs(os.path.join(tmp, "data"), exist_ok=True); os.makedirs(os.path.join(tmp, "frida"), exist_ok=True)
    cfg = make_cfg(tmp)
    runner = FakeRunner(handler)
    conf = ScriptedConfirmer(answers, default)
    clk = clock or Clock()
    audio = FakeAudio()
    d = build_dispatcher(cfg, conf, runner=runner, recall=recall, audio=audio, ledger=TaintLedger(clock=clk))
    d.clock = clk; d.policy.clock = clk; d.grants.clock = clk
    return type("Env", (), dict(tmp=tmp, cfg=cfg, runner=runner, conf=conf, d=d, audio=audio, clk=clk))()
