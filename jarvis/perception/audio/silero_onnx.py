"""Torch-free Silero VAD (v5 ONNX). Mirrors the official OnnxWrapper: a 64-sample context is prepended to every
512-sample frame (16 kHz) and a (2, 1, 128) recurrent state is carried between frames.

I/O verified against the shipped model: inputs input[b, n] f32, state[2, b, 128] f32, sr[] i64;
outputs output[b, 1], stateN.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import urllib.request
from pathlib import Path

import numpy as np

from .vad import VAD

SILERO_VERSION = "v5.1.2"
SILERO_URL = f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_VERSION}/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
CACHE_DIR = Path(os.path.expanduser("~/.cache/jarvis/models"))
CACHE_PATH = CACHE_DIR / f"silero_vad_{SILERO_VERSION}.onnx"
CONTEXT = {16000: 64, 8000: 32}
FRAME = {16000: 512, 8000: 256}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fetch_model(dest: Path = CACHE_PATH, url: str = SILERO_URL, sha256: str = SILERO_SHA256) -> Path:
    """Download the pinned model and verify its hash (refuses to keep a mismatching file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        f.write(r.read())
    got = sha256_file(tmp)
    if got != sha256:
        tmp.unlink()
        raise RuntimeError(f"silero model hash mismatch: expected {sha256[:12]}..., got {got[:12]}...")
    tmp.replace(dest)
    return dest


def find_model(explicit: str | None = None) -> Path:
    cands: list[Path] = []
    if explicit:
        cands.append(Path(os.path.expanduser(explicit)))
    cands.append(CACHE_PATH)
    spec = importlib.util.find_spec("silero_vad")
    if spec and spec.submodule_search_locations:
        cands += [Path(loc) / "data" / "silero_vad.onnx" for loc in spec.submodule_search_locations]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("Silero ONNX model not found; run `jarvis fetch-models` (or set vad.model_path)")


class SileroOnnxVAD(VAD):
    def __init__(self, sample_rate: int = 16000, model_path: str | None = None):
        import onnxruntime as ort

        if sample_rate not in CONTEXT:
            raise ValueError("Silero VAD supports 8000/16000 Hz")
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(str(find_model(model_path)), sess_options=so,
                                          providers=["CPUExecutionProvider"])
        self._sr = sample_rate
        self._ctx_n = CONTEXT[sample_rate]
        self._frame_n = FRAME[sample_rate]
        self._sr_arr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._ctx_n), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        if frame.shape[-1] != self._frame_n:
            raise ValueError(f"expected {self._frame_n} samples, got {frame.shape[-1]}")
        x = np.concatenate([self._context, (frame.astype(np.float32) / 32768.0)[None, :]], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr_arr})
        self._context = x[:, -self._ctx_n:]
        return float(out[0, 0])
