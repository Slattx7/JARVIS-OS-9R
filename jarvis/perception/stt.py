from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .audio.dsp import to_float32


class STT:
    def transcribe(self, pcm: np.ndarray) -> str:  # blocking; pipeline runs it in a thread
        raise NotImplementedError

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.int16))


class FasterWhisperSTT(STT):
    def __init__(self, cfg):
        from faster_whisper import WhisperModel

        self._m = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        self._lang, self._beam = cfg.language, cfg.beam_size

    def transcribe(self, pcm: np.ndarray) -> str:
        segs, _ = self._m.transcribe(
            to_float32(pcm), language=self._lang, beam_size=self._beam, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segs).strip()


class FakeSTT(STT):
    """script: list of transcripts returned in order, or callable(pcm)->str."""

    def __init__(self, script: list[str] | Callable[[np.ndarray], str], latency: float = 0.0):
        self.script, self.latency = script, latency
        self.calls: list[int] = []

    def transcribe(self, pcm):
        self.calls.append(len(pcm))
        if self.latency:
            time.sleep(self.latency)
        if callable(self.script):
            return self.script(pcm)
        return self.script.pop(0) if self.script else ""

    def warmup(self) -> None: ...


def make_stt(cfg) -> STT:
    if cfg.stt.backend == "faster-whisper":
        return FasterWhisperSTT(cfg.stt)
    raise ValueError(f"unknown stt backend {cfg.stt.backend!r}")
