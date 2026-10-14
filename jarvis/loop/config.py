from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .checker import ToolPolicy
from .gate import GateConfig
from .monitor import MonitorConfig


@dataclass
class LoopConfig:
    root: str = "jarvis_loop_data"            # registry/, runs/, state.json, replay_cache.json live here
    traces: str = "traces"                    # .jsonl file or directory of them (append-only jarvis_trace output)
    base_model: str = ""                      # HF id or local path (training base; also v0's artifact)
    model_config: str = ""                    # path to the base model's HF config.json (VRAM planning)
    tiers: dict = field(default_factory=dict)         # tool -> permission tier (0 = read-only ... N)
    default_grant: int = 1                    # tier assumed when a trace doesn't record granted_tier
    deny_tools: list = field(default_factory=list)
    arg_match: str = "exact"                  # exact | subset | tool_only
    holdout_frac: float = 0.15
    max_holdout: int = 400                    # replays per candidate: bound the wall-clock on a laptop GPU
    max_per_task: int = 3
    max_noop_frac: float = 0.6                # schema guard: refuse to train if >60% of demos have no tool calls
    min_train: int = 100                      # no training below this many curated demos
    min_new_examples: int = 50                # ...or below this many new ones since the last trained cycle
    max_cycles_per_day: int = 2               # crash-loop / compute guard
    vram_gb: float = 6.0
    epochs: int = 2
    lr: float = 2e-4
    effective_batch: int = 16
    train_timeout_s: int = 6 * 3600
    trainer: dict = field(default_factory=lambda: {"kind": "lora"})      # lora | dry | command
    replay: dict = field(default_factory=lambda: {
        "kind": "openai", "base_url": "http://127.0.0.1:8080/v1", "model_template": "{version}",
        "timeout_s": 60, "max_tokens": 384})                               # openai | command | sim
    replay_workers: int = 1
    use_replay_cache: bool = True
    activate_cmd: list = field(default_factory=list)   # argv template ({version},{artifact}) run before replaying a version
    promote_hook: list = field(default_factory=list)   # argv template run AFTER promote; non-zero exit => auto-rollback
    gate: dict = field(default_factory=dict)           # GateConfig overrides
    golden_file: str = ""                     # optional JSON list of trace ids that must never regress
    monitor: dict = field(default_factory=dict)        # MonitorConfig overrides
    auto_rollback: bool = True

    # ---- paths
    @property
    def registry_dir(self) -> str:
        return str(Path(self.root) / "registry")

    @property
    def runs_dir(self) -> str:
        return str(Path(self.root) / "runs")

    @property
    def state_path(self) -> str:
        return str(Path(self.root) / "state.json")

    @property
    def cache_path(self) -> str:
        return str(Path(self.root) / "replay_cache.json")

    # ---- factories
    def policy(self) -> ToolPolicy:
        return ToolPolicy(tiers=dict(self.tiers), default_grant=self.default_grant, deny=frozenset(self.deny_tools))

    def gate_config(self, golden=()) -> GateConfig:
        return GateConfig(**{**self.gate, "golden": frozenset(golden)})

    def monitor_config(self) -> MonitorConfig:
        return MonitorConfig(**self.monitor)

    # ---- io
    @classmethod
    def load(cls, path) -> "LoopConfig":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        bad = sorted(set(data) - known)
        if bad:
            raise ValueError(f"unknown config keys: {bad}")
        cfg = cls(**data)
        for k in ("root", "traces", "model_config", "golden_file"):
            v = getattr(cfg, k)
            if v and not Path(v).is_absolute():
                setattr(cfg, k, str((p.parent / v).resolve()))
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
