"""Barge-in / VAD calibration. No echo cancellation yet, so the barge-in RMS floor must sit above the speaker's
bleed into the mic but below the user's speech. Measure all three and say so plainly when they don't separate."""
from __future__ import annotations

import numpy as np

from .audio.dsp import rms


def pct(xs, q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if len(xs) else 0.0


def suggest(ambient: list[float], bleed: list[float] | None = None, speech: list[float] | None = None) -> dict:
    """Inputs: per-frame normalized RMS from three recordings. Returns suggested config + warnings."""
    n95 = pct(ambient, 95)
    b95 = pct(bleed, 95) if bleed else 0.0
    s10 = pct(speech, 10) if speech else 0.0
    floor = max(0.015, 3.0 * n95, 1.5 * b95)
    warnings: list[str] = []
    barge = True
    if speech and floor > 0.6 * s10:
        barge = False
        warnings.append(
            f"speaker bleed (p95 {b95:.3f}) is too close to your speech level (p10 {s10:.3f}) for reliable barge-in. "
            "Use headphones, lower the speaker volume, or enable PipeWire echo-cancel (docs/CALIBRATION.md); "
            "barge-in is suggested OFF until then.")
    if n95 > 0.02:
        warnings.append(f"ambient noise is high (p95 {n95:.3f}): expect VAD false starts; raise vad.threshold to 0.6")
    if speech and s10 < 0.02:
        warnings.append(f"your speech is quiet at the mic (p10 {s10:.3f}); raise input gain or move closer")
    return {"barge_min_rms": round(floor, 3), "barge_enabled": barge, "ambient_p95": round(n95, 4),
            "bleed_p95": round(b95, 4), "speech_p10": round(s10, 4),
            "vad_threshold": 0.6 if n95 > 0.02 else 0.5, "warnings": warnings}


def toml_snippet(s: dict) -> str:
    return (f"[barge]\nenabled = {str(s['barge_enabled']).lower()}\nmin_rms = {s['barge_min_rms']}\n\n"
            f"[vad]\nthreshold = {s['vad_threshold']}\n")


def frame_levels(pcm: np.ndarray, frame: int = 512) -> list[float]:
    return [rms(pcm[i:i + frame]) for i in range(0, len(pcm) - frame + 1, frame)]


def test_signal(sample_rate: int = 24000, seconds: float = 4.0) -> np.ndarray:
    """Speech-like calibration sound: harmonic voice with syllabic amplitude modulation + band-limited noise."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.2 * t) ** 2
    rng = np.random.default_rng(0)
    noise = np.convolve(rng.standard_normal(len(t)), np.ones(4) / 4, mode="same") * 0.15
    x = (voice / 3.2 * env + noise) * 0.5
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def meter_line(level: float, prob: float, width: int = 30) -> str:
    db = 20 * np.log10(max(level, 1e-5))
    filled = int(np.clip((db + 60) / 60, 0, 1) * width)
    return f"{db:6.1f} dBFS |{'#' * filled}{'.' * (width - filled)}| vad {prob:4.2f} {'SPEECH' if prob >= 0.5 else '      '}"
