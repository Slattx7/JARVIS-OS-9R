"""Wiring: real stack from config."""
from __future__ import annotations

from .actions import ExecContext, build_registry
from .audio.io import SoundDeviceMic, SoundDeviceSpeaker
from .audio.vad import make_vad
from .audio.wake import make_wake
from .audit import AuditLog
from .approval import load_verifier
from .broker import Broker
from .config import Config, expand
from .fastpath import Fastpath
from .llm import Conversation, build_system_prompt, make_llm
from .oob import OOBServer
from .pipeline import Pipeline
from .stt import make_stt
from .trace import Tracer
from .tts import make_tts


def build_real(cfg: Config, log=print, mic=None, speaker=None):
    registry = build_registry(cfg)
    audit = AuditLog(expand(cfg.broker.audit_path))
    verifier = load_verifier(cfg, log)
    ctx = ExecContext(cfg)
    broker = Broker(cfg, registry, audit, verifier, ctx)

    log("loading VAD / wake / STT / TTS ...")
    vad, wake = make_vad(cfg), make_wake(cfg)
    stt, tts, llm = make_stt(cfg), make_tts(cfg), make_llm(cfg)
    cfg.tts.sample_rate = tts.sample_rate
    speaker = speaker or SoundDeviceSpeaker(tts.sample_rate, cfg.audio.output_device)
    mic = mic or SoundDeviceMic(cfg.audio.sample_rate, cfg.audio.frame_samples, cfg.audio.input_device)
    conv = Conversation(build_system_prompt(registry.catalog(), cfg.llm.system_prompt),
                        cfg.llm.history_turns, cfg.llm.history_char_budget)
    pipe = Pipeline(cfg, speaker=speaker, vad=vad, wake=wake, stt=stt, llm=llm, tts=tts, broker=broker,
                    tracer=Tracer(expand(cfg.trace.path), cfg.trace.log_text), conversation=conv,
                    fastpath=Fastpath(cfg.fastpath.default_device) if cfg.fastpath.enabled else None)
    ctx.say = pipe.say
    oob = OOBServer(broker, verifier, cfg.oob.bind, cfg.oob.port, on_event=pipe.emit) if cfg.oob.enabled else None
    return pipe, mic, speaker, oob, conv, stt, tts, llm
