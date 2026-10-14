from __future__ import annotations
from .checker import Checker
from .config import LoopConfig
from .replay import CommandActivator, CommandPropose, OpenAICompat, ProposalReplayer
from .sim import SimReplayer
from .trainer import CommandTrainer, DryRunTrainer, LoraTrainer, ModelSpec


def build_trainer(cfg: LoopConfig):
    t = dict(cfg.trainer)
    kind = t.pop("kind", "lora")
    if kind == "dry":
        return DryRunTrainer()
    if kind == "command":
        return CommandTrainer(t["cmd"], timeout=cfg.train_timeout_s)
    if kind == "lora":
        if not cfg.model_config:
            raise ValueError("trainer.kind=lora needs `model_config` (path to the base model's HF config.json)")
        return LoraTrainer(cfg.base_model, ModelSpec.from_hf_config(cfg.model_config), vram_gb=cfg.vram_gb,
                           epochs=cfg.epochs, lr=cfg.lr, effective_batch=cfg.effective_batch,
                           timeout_s=cfg.train_timeout_s, **t)
    raise ValueError(f"unknown trainer kind {kind!r}")


def build_replayer(cfg: LoopConfig):
    r = dict(cfg.replay)
    kind = r.pop("kind", "openai")
    if kind == "sim":
        return SimReplayer(**r.get("params", {}))
    checker = Checker(cfg.policy(), cfg.arg_match)
    activate = CommandActivator(cfg.activate_cmd) if cfg.activate_cmd else None
    if kind == "openai":
        propose = OpenAICompat(r.get("base_url", "http://127.0.0.1:8080/v1"), r.get("model_template", "{version}"),
                               r.get("timeout_s", 60), r.get("max_tokens", 384), r.get("temperature", 0.0),
                               r.get("api_key"))
    elif kind == "command":
        propose = CommandPropose(r["cmd"], r.get("timeout_s", 120))
    else:
        raise ValueError(f"unknown replay kind {kind!r}")
    return ProposalReplayer(propose, checker, activate)


def build_loop(cfg: LoopConfig, clock=None):
    from .loop import Loop
    import time
    return Loop(cfg, build_trainer(cfg), build_replayer(cfg), clock or time.time)
