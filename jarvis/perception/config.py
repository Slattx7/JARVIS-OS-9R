"""Typed config tree, loaded from TOML. Unknown keys are errors (typos fail loudly)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass
class AudioCfg:
    sample_rate: int = 16000
    frame_samples: int = 512          # 32 ms @ 16 kHz (Silero's required frame size)
    input_device: str | int | None = None
    output_device: str | int | None = None
    preroll_ms: int = 300
    # After a reply finishes, ignore the mic for this long so the room's reverb tail (the assistant's own
    # voice) is never mistaken for the user, e.g. an echoed "yes" confirming its own prompt.
    post_speech_guard_ms: int = 350


@dataclass
class VadCfg:
    backend: str = "silero"           # silero (auto: onnx, else torch) | silero-onnx | silero-torch | energy
    model_path: str | None = None     # override for the Silero ONNX file (default: ~/.cache/jarvis/models, see `jarvis fetch-models`)
    threshold: float = 0.5
    start_frames: int = 3
    endpoint_ms: int = 450            # trailing silence that ends an utterance
    min_utterance_ms: int = 250
    max_utterance_s: float = 20.0
    no_speech_timeout_s: float = 6.0
    # Speculative STT: after `spec_ms` of trailing silence, start transcribing; if the user stays quiet until the
    # endpoint, the result is reused (hides STT latency inside the endpoint wait). 0 disables.
    spec_ms: int = 220
    # Opt-in: end the utterance as soon as the speculative transcript ends like a finished sentence (. ? !).
    # Cuts perceived latency by (endpoint_ms - spec_ms - stt time) but can cut off users who pause mid-thought.
    fast_commit: bool = False


@dataclass
class WakeCfg:
    mode: str = "openwakeword"        # openwakeword | always | fake
    model: str = "hey_jarvis"
    threshold: float = 0.5
    cooldown_ms: int = 1500
    followup_window_s: float = 6.0    # after a reply, keep listening without the wake word (0 = off)


@dataclass
class SttCfg:
    backend: str = "faster-whisper"
    model: str = "distil-small.en"
    device: str = "auto"
    compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = 1


@dataclass
class LlmCfg:
    backend: str = "llama-server"
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local"
    max_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.9
    timeout_s: float = 30.0
    history_turns: int = 6
    history_char_budget: int = 6000
    max_tool_rounds: int = 2
    system_prompt: str | None = None
    # merged into the request body; e.g. {"chat_template_kwargs": {"enable_thinking": false}} for Qwen3
    extra_body: dict = field(default_factory=dict)


@dataclass
class TtsCfg:
    backend: str = "kokoro"
    voice: str = "af_heart"
    speed: float = 1.05
    lang_code: str = "a"
    device: str = "cpu"
    sample_rate: int = 24000
    first_min_chars: int = 12
    min_chars: int = 40
    first_clause_min_chars: int = 28
    clause_min_chars: int = 70


@dataclass
class BargeCfg:
    enabled: bool = True
    min_speech_ms: int = 220
    min_rms: float = 0.02             # normalized 0..1; rejects speaker bleed


@dataclass
class DeviceCfg:
    serial: str | None = None
    root: bool = False


@dataclass
class OobCfg:
    enabled: bool = False
    bind: str = "127.0.0.1"           # bind to your tailscale0 address to approve from the phone
    port: int = 8765


@dataclass
class BrokerCfg:
    dry_run: bool = True              # state-changing actions are logged, not executed
    audit_path: str = "~/.local/share/jarvis/audit.jsonl"
    approval_scheme: str = "ed25519"  # ed25519 | hmac
    approver_pub_path: str = "~/.config/jarvis/approver.pub"   # ed25519: public key only; private key stays on the phone
    secret_path: str = "~/.config/jarvis/approval.key"          # hmac scheme only
    confirm_ttl_s: float = 30.0
    oob_ttl_s: float = 120.0
    max_intents_per_turn: int = 3
    notes_path: str = "~/.local/share/jarvis/notes.md"


@dataclass
class FastpathCfg:
    enabled: bool = True              # deterministic local commands skip the LLM (time, battery, volume, media, timers, notes)
    default_device: str = "pro14"


@dataclass
class N8nCfg:
    """Allowlisted n8n webhook workflows. Each becomes its own action `n8n.<name>` with its own tier.
    The model supplies at most a short `text`; URL, path and headers always come from here."""
    enabled: bool = False
    base_url: str = "http://127.0.0.1:5678"
    timeout_s: float = 10.0
    auth_header: str | None = None          # e.g. "X-Jarvis-Token": the header NAME
    auth_env: str | None = None             # env var holding its VALUE (never put secrets in this file)
    # name -> {path = "/webhook/<id>", tier = 2, description = "..."}   (tier 0..3; default 2 = spoken confirmation)
    workflows: dict = field(default_factory=dict)


@dataclass
class TraceCfg:
    path: str = "~/.local/share/jarvis/trace.jsonl"
    log_text: bool = False            # transcripts in traces are off by default


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    wake: WakeCfg = field(default_factory=WakeCfg)
    stt: SttCfg = field(default_factory=SttCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    tts: TtsCfg = field(default_factory=TtsCfg)
    barge: BargeCfg = field(default_factory=BargeCfg)
    broker: BrokerCfg = field(default_factory=BrokerCfg)
    oob: OobCfg = field(default_factory=OobCfg)
    trace: TraceCfg = field(default_factory=TraceCfg)
    fastpath: FastpathCfg = field(default_factory=FastpathCfg)
    n8n: N8nCfg = field(default_factory=N8nCfg)
    pro14: DeviceCfg = field(default_factory=DeviceCfg)
    r9: DeviceCfg = field(default_factory=lambda: DeviceCfg(root=True))
    adb_path: str = "adb"
    frida_ps_path: str = "frida-ps"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


def _merge(obj: Any, data: dict, path: str = "") -> None:
    for key, val in data.items():
        if not hasattr(obj, key):
            raise ConfigError(f"unknown config key: {path}{key}")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            if not isinstance(val, dict):
                raise ConfigError(f"{path}{key} must be a table")
            _merge(cur, val, f"{path}{key}.")
        else:
            setattr(obj, key, val)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    p = expand(path)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    _merge(cfg, data)
    return cfg
