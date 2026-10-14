"""Voice activity detection + endpointing.

`VAD.prob` scores one frame; `Endpointer` turns a probability stream into
speech_start / speech_end events using frame counts only (never wall time), so behaviour is
identical at 1x realtime and in accelerated tests.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from .dsp import rms


class VAD:
    def prob(self, frame: np.ndarray) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class EnergyVAD(VAD):
    """Adaptive-noise-floor energy gate. Dependency-free fallback and test VAD."""

    def __init__(self, abs_min: float = 0.008, ratio: float = 3.0, alpha: float = 0.02):
        self.abs_min = abs_min
        self.ratio = ratio
        self.alpha = alpha
        self.floor = abs_min / ratio

    def prob(self, frame: np.ndarray) -> float:
        r = rms(frame)
        thr = max(self.abs_min, self.floor * self.ratio)
        if r < thr:  # only learn the floor from non-speech
            self.floor = (1 - self.alpha) * self.floor + self.alpha * r
        return min(1.0, 0.5 * r / thr)


class SileroVAD(VAD):
    def __init__(self, sample_rate: int = 16000):
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self._torch = torch
        self._model = load_silero_vad()
        self._sr = sample_rate

    def prob(self, frame: np.ndarray) -> float:
        x = self._torch.from_numpy(frame.astype(np.float32) / 32768.0)
        with self._torch.no_grad():
            return float(self._model(x, self._sr).item())

    def reset(self) -> None:
        self._model.reset_states()


def make_vad(cfg) -> VAD:
    """silero (auto: torch-free ONNX if onnxruntime + model are available, else torch) | silero-onnx | silero-torch | energy"""
    backend = cfg.vad.backend
    if backend == "energy":
        return EnergyVAD()
    if backend in ("silero", "silero-onnx"):
        try:
            from .silero_onnx import SileroOnnxVAD

            return SileroOnnxVAD(cfg.audio.sample_rate, cfg.vad.model_path)
        except (ImportError, FileNotFoundError):
            if backend == "silero-onnx":
                raise
    if backend in ("silero", "silero-torch"):
        return SileroVAD(cfg.audio.sample_rate)
    raise ValueError(f"unknown vad backend {backend!r}")


class Ev(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    DISCARD = "discard"        # speech shorter than min_speech_frames (click/cough)
    TIMEOUT = "timeout"        # no speech began within no_speech_frames
    MAX_LEN = "max_len"        # utterance hit the hard cap while still speaking


class Endpointer:
    def __init__(
        self,
        threshold: float,
        start_frames: int,
        endpoint_frames: int,
        min_speech_frames: int,
        max_frames: int,
        no_speech_frames: int,
        hysteresis: float = 0.15,
    ):
        self.threshold = threshold
        self.start_frames = start_frames
        self.endpoint_frames = endpoint_frames
        self.min_speech_frames = min_speech_frames
        self.max_frames = max_frames
        self.default_no_speech_frames = no_speech_frames
        self.hysteresis = hysteresis
        self.reset()

    def reset(self, no_speech_frames: int | None = None) -> None:
        self.no_speech_frames = no_speech_frames or self.default_no_speech_frames
        self.n = 0
        self.in_speech = False
        self._run = 0
        self._silence = 0
        self._speech_frames = 0
        self._prev = False

    @property
    def silence_frames(self) -> int:
        """Consecutive non-speech frames since the last speech frame (0 while not in speech)."""
        return self._silence if self.in_speech else 0

    def update(self, p: float) -> Ev | None:
        self.n += 1
        low = max(0.0, self.threshold - self.hysteresis)
        speechy = p >= self.threshold or (self._prev and p >= low)
        self._prev = speechy
        if not self.in_speech:
            self._run = self._run + 1 if speechy else 0
            if self._run >= self.start_frames:
                self.in_speech = True
                self._speech_frames = self._run
                self._silence = 0
                return Ev.SPEECH_START
            if self.n >= self.no_speech_frames:
                return Ev.TIMEOUT
            return None
        if speechy:
            self._speech_frames += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self.endpoint_frames:
            self.in_speech = False
            return Ev.SPEECH_END if self._speech_frames >= self.min_speech_frames else Ev.DISCARD
        if self.n >= self.max_frames:
            return Ev.MAX_LEN
        return None
