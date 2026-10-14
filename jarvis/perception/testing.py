"""Synthetic audio + fake-stack factory. Used by the test-suite and `jarvis demo` (no models, no devices)."""
from __future__ import annotations

import numpy as np

from .actions import CmdResult, ExecContext, FakeTransport, build_registry
from . import crypto
from .approval import Ed25519Signer, Ed25519Verifier
from .audit import AuditLog
from .audio.io import NullSpeaker, ScriptedMic
from .audio.vad import EnergyVAD
from .audio.wake import FakeWake
from .broker import Broker
from .config import Config
from .llm import Conversation, FakeLLM, build_system_prompt
from .pipeline import Pipeline
from .stt import FakeSTT
from .trace import Tracer
from .tts import FakeTTS

FRAME = 512
SR = 16000
TEST_SEED = bytes(range(32))
TEST_SIGNER = Ed25519Signer(TEST_SEED)


def silence(n_frames: int, level: float = 0.0008, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(FRAME) * level * 32768).astype(np.int16) for _ in range(n_frames)]


def speech(n_frames: int, amp: float = 0.25, f0: float = 140.0) -> list[np.ndarray]:
    out, t0 = [], 0
    for _ in range(n_frames):
        t = (np.arange(FRAME) + t0) / SR
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        out.append((amp * sig / 2.2 * 32768).astype(np.int16))
        t0 += FRAME
    return out


def frames_for_ms(ms: float) -> int:
    return int(np.ceil(ms / (1000 * FRAME / SR)))


def utterance(speech_ms: int = 700, lead_ms: int = 200, tail_ms: int = 900) -> list[np.ndarray]:
    return silence(frames_for_ms(lead_ms)) + speech(frames_for_ms(speech_ms)) + silence(frames_for_ms(tail_ms))


def fast_config() -> Config:
    cfg = Config()
    cfg.vad.backend = "energy"
    cfg.wake.mode = "fake"
    cfg.wake.followup_window_s = 2.0
    cfg.vad.no_speech_timeout_s = 2.0
    cfg.vad.spec_ms = 0  # deterministic STT scripts; spec tests opt in
    cfg.audio.post_speech_guard_ms = 0  # guard tests opt in
    cfg.broker.dry_run = False
    cfg.trace.path = ""
    return cfg


def battery_transport() -> FakeTransport:
    return FakeTransport({"dumpsys battery": CmdResult(0, "  level: 83\n  status: 3\n  temperature: 301\n")})


def make_stack(frames, stt_script, llm_script, *, cfg: Config | None = None, wake_at=(5,), dry_run=False,
               llm_delay=0.0, tts_latency=0.0, realtime_speaker=False, transport=None, audit_path=None,
               tts_ms_per_char=20.0, pace_s=0.0, verifier=None, fastpath=None, vad=None, wake=None):
    cfg = cfg or fast_config()
    cfg.broker.dry_run = dry_run
    transport = transport or battery_transport()
    ctx = ExecContext(cfg, transports={("pro14", False): transport, ("r9", False): transport, ("r9", True): transport})
    registry = build_registry(cfg)
    audit = AuditLog(audit_path)
    broker = Broker(cfg, registry, audit, verifier or Ed25519Verifier(crypto.public_key(TEST_SEED)), ctx)
    conv = Conversation(build_system_prompt(registry.catalog()), cfg.llm.history_turns, cfg.llm.history_char_budget)
    speaker = NullSpeaker(24000, realtime=realtime_speaker)
    llm = FakeLLM(llm_script, token_delay=llm_delay)
    tts = FakeTTS(ms_per_char=tts_ms_per_char, latency=tts_latency)
    stt = FakeSTT(stt_script)
    pipe = Pipeline(cfg, speaker=speaker, vad=vad or EnergyVAD(), wake=wake or FakeWake(wake_at), stt=stt, llm=llm, tts=tts,
                    broker=broker, tracer=Tracer(None), conversation=conv, fastpath=fastpath)
    ctx.say = pipe.say
    mic = ScriptedMic(frames, pace_s=pace_s)
    return pipe, mic, dict(cfg=cfg, broker=broker, audit=audit, speaker=speaker, llm=llm, tts=tts, stt=stt,
                           transport=transport, conv=conv, ctx=ctx)


class GatedMic:
    """Yields blocks of frames; after each block waits until the pipeline has finished its turn.
    Models 'the human waits for the reply' without real-time pacing."""

    def __init__(self, blocks: list[list[np.ndarray]], pipe_getter, settle_s: float = 0.0):
        self.blocks, self._pipe, self.settle_s = blocks, pipe_getter, settle_s

    async def frames(self):
        import asyncio

        from .pipeline import State
        for block in self.blocks:
            for f in block:
                yield f
                await asyncio.sleep(0)
            pipe = self._pipe()
            await asyncio.sleep(0.01)
            while pipe.state is State.BUSY:
                await asyncio.sleep(0.005)
            await asyncio.sleep(self.settle_s)
