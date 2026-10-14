from __future__ import annotations

import numpy as np


class WakeDetector:
    def process(self, frame: np.ndarray) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class AlwaysWake(WakeDetector):
    """Always-listening dev mode: every frame is a wake. Use only for bench/debug."""

    def process(self, frame):
        return True


class FakeWake(WakeDetector):
    """Fires on specific frame indices (0-based count of process() calls). For tests/demo."""

    def __init__(self, at):
        self.at = set(at)
        self.n = -1

    def process(self, frame):
        self.n += 1
        return self.n in self.at


class OpenWakeWord(WakeDetector):
    CHUNK = 1280  # 80 ms @ 16 kHz, openWakeWord's native hop

    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5):
        from openwakeword.model import Model

        try:
            self._m = Model(wakeword_models=[model], inference_framework="onnx")
        except Exception:
            import openwakeword

            openwakeword.utils.download_models()
            self._m = Model(wakeword_models=[model], inference_framework="onnx")
        self._thr = threshold
        self._buf = np.zeros(0, dtype=np.int16)

    def process(self, frame):
        self._buf = np.concatenate([self._buf, frame])
        hit = False
        while len(self._buf) >= self.CHUNK:
            chunk, self._buf = self._buf[: self.CHUNK], self._buf[self.CHUNK :]
            scores = self._m.predict(chunk)
            if scores and max(scores.values()) >= self._thr:
                hit = True
        if hit:
            self.reset()
        return hit

    def reset(self):
        self._buf = np.zeros(0, dtype=np.int16)
        try:
            self._m.reset()
        except Exception:
            pass


def make_wake(cfg) -> WakeDetector:
    mode = cfg.wake.mode
    if mode == "always":
        return AlwaysWake()
    if mode == "openwakeword":
        return OpenWakeWord(cfg.wake.model, cfg.wake.threshold)
    raise ValueError(f"unknown wake mode {mode!r} (use 'fake' only via code)")
