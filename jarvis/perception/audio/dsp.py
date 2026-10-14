from __future__ import annotations

import numpy as np


def rms(frame: np.ndarray) -> float:
    """Normalized RMS (0..1) of an int16 frame."""
    if frame.size == 0:
        return 0.0
    x = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def resample(pcm: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """int16 mono resample. scipy's polyphase filter when available (anti-aliased), linear interpolation otherwise."""
    if sr_in == sr_out or pcm.size == 0:
        return pcm
    x = pcm.astype(np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        y = resample_poly(x, sr_out // g, sr_in // g)
    except ImportError:
        n = round(len(x) * sr_out / sr_in)
        y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(y, -32768, 32767).astype(np.int16)


def read_wav(path, sample_rate: int = 16000) -> np.ndarray:
    """Any 8/16/24/32-bit PCM wav -> mono int16 at `sample_rate`."""
    import wave

    with wave.open(str(path)) as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 256.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 65536.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample(np.clip(x, -32768, 32767).astype(np.int16), sr, sample_rate)


def write_wav(path, pcm: np.ndarray, sample_rate: int) -> None:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
