from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .audio.dsp import to_int16


class TTS:
    sample_rate = 24000

    def synth(self, text: str) -> Iterator[np.ndarray]:  # blocking generator of int16 chunks
        raise NotImplementedError

    def warmup(self) -> None:
        for _ in self.synth("Ready."):
            pass


class KokoroTTS(TTS):
    def __init__(self, cfg):
        from kokoro import KPipeline

        self.sample_rate = cfg.sample_rate
        self._voice, self._speed = cfg.voice, cfg.speed
        self._pipe = KPipeline(lang_code=cfg.lang_code, device=cfg.device)

    def synth(self, text):
        # Verified against kokoro 0.9.4 source: __call__ yields Result objects that unpack as
        # (graphemes, phonemes, audio); `audio` is a torch.FloatTensor at 24 kHz and may be None for empty segments.
        for _gs, _ps, audio in self._pipe(text, voice=self._voice, speed=self._speed):
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            pcm = to_int16(np.asarray(audio, dtype=np.float32).reshape(-1))
            if pcm.size:
                yield pcm


class FakeTTS(TTS):
    def __init__(self, sample_rate: int = 24000, ms_per_char: float = 20.0, latency: float = 0.0, chunk_ms: int = 100):
        self.sample_rate, self.ms_per_char, self.latency, self.chunk_ms = sample_rate, ms_per_char, latency, chunk_ms
        self.synthesized: list[str] = []

    def synth(self, text):
        self.synthesized.append(text)
        if self.latency:
            time.sleep(self.latency)
        total = int(self.sample_rate * len(text) * self.ms_per_char / 1000)
        step = int(self.sample_rate * self.chunk_ms / 1000)
        t = np.arange(total)
        wave = to_int16(0.2 * np.sin(2 * np.pi * 220 * t / self.sample_rate))
        for i in range(0, total, step):
            yield wave[i:i + step]

    def warmup(self) -> None: ...


def make_tts(cfg) -> TTS:
    if cfg.tts.backend == "kokoro":
        return KokoroTTS(cfg.tts)
    raise ValueError(f"unknown tts backend {cfg.tts.backend!r}")
