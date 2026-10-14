"""Backend benchmarks + a first-run self-test. Everything takes backend *objects*, so it runs on fakes in tests
and on the real stack from the CLI."""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .audio.dsp import resample
from .audio.vad import VAD

SAMPLE_TEXT = "The weather is nice today, and the battery on your phone is at eighty three percent."


def bench_vad(vad: VAD, frames: int = 400, frame_samples: int = 512) -> dict:
    rng = np.random.default_rng(0)
    f = (rng.standard_normal(frame_samples) * 800).astype(np.int16)
    vad.reset()
    t = time.perf_counter()
    for _ in range(frames):
        vad.prob(f)
    per = (time.perf_counter() - t) / frames
    return {"us_per_frame": round(per * 1e6, 1), "rtf": round(per / (frame_samples / 16000), 4)}


def bench_stt(transcribe: Callable[[np.ndarray], str], pcm: np.ndarray, runs: int = 3, sr: int = 16000) -> dict:
    times, text = [], ""
    for _ in range(runs):
        t = time.perf_counter()
        text = transcribe(pcm)
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    return {"ms_median": round(med * 1e3), "ms_first": round(times[0] * 1e3), "audio_s": round(len(pcm) / sr, 2),
            "rtf": round(med / (len(pcm) / sr), 3), "text": text}


def bench_tts(tts, text: str = SAMPLE_TEXT, runs: int = 2) -> dict:
    best: dict = {}
    for _ in range(runs):
        t = time.perf_counter()
        first, n = None, 0
        for chunk in tts.synth(text):
            if first is None:
                first = time.perf_counter() - t
            n += len(chunk)
        total = time.perf_counter() - t
        audio_s = n / tts.sample_rate
        r = {"ttfa_ms": round((first or total) * 1e3), "total_ms": round(total * 1e3), "audio_s": round(audio_s, 2),
             "rtf": round(total / audio_s, 3) if audio_s else float("inf")}
        if not best or r["ttfa_ms"] < best["ttfa_ms"]:
            best = r   # keep the warm run
    return best


def bench_llm(llm, system_prompt: str, prompt: str = "Say hello in five words.", runs: int = 2) -> dict:
    async def once():
        t = time.perf_counter()
        first, n = None, 0
        async for _ in llm.stream([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]):
            if first is None:
                first = time.perf_counter() - t
            n += 1
        total = time.perf_counter() - t
        return {"ttft_ms": round((first or total) * 1e3), "chunks": n,
                "chunks_per_s": round(n / max(total - (first or 0), 1e-6), 1)}
    results = [asyncio.run(once()) for _ in range(runs)]
    return min(results, key=lambda r: r["ttft_ms"])


@dataclass
class Check:
    name: str
    status: str            # PASS | FAIL | SKIP
    detail: str = ""
    data: dict = field(default_factory=dict)


def run_selftest(*, vad=None, wake=None, tts=None, stt=None, llm=None, system_prompt: str = "You are a test.",
                 expect_word: str = "weather", log: Callable[[Check], None] | None = None) -> list[Check]:
    """TTS -> STT round trip proves both models load, run on the intended device, and agree with each other."""
    out: list[Check] = []

    def add(c: Check) -> None:
        out.append(c)
        if log:
            log(c)

    def guarded(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            add(Check(name, "FAIL", f"{type(e).__name__}: {e}"))
            return None

    audio16 = None
    if tts is not None:
        def do_tts():
            r = bench_tts(tts, "What is the weather like today?", runs=2)
            add(Check("tts", "PASS", f"first audio {r['ttfa_ms']} ms, rtf {r['rtf']}", r))
            pcm = np.concatenate(list(tts.synth("What is the weather like today?")))
            return resample(pcm, tts.sample_rate, 16000)
        audio16 = guarded("tts", do_tts)
    else:
        add(Check("tts", "SKIP"))

    if vad is not None and audio16 is not None:
        def do_vad():
            r = bench_vad(vad)
            vad.reset()
            probs = [vad.prob(audio16[i:i + 512]) for i in range(0, len(audio16) - 511, 512)]
            speech = float(np.mean(np.array(probs) > 0.5)) if probs else 0.0
            ok = speech > 0.3 and r["rtf"] < 0.5
            add(Check("vad", "PASS" if ok else "FAIL", f"{r['us_per_frame']} us/frame, {speech:.0%} of synthesized speech detected", r))
        guarded("vad", do_vad)
    else:
        add(Check("vad", "SKIP"))

    if wake is not None:
        def do_wake():
            fired = sum(bool(wake.process(f)) for f in [np.zeros(512, np.int16)] * 80)
            add(Check("wake", "PASS" if fired == 0 else "FAIL", "silence does not trigger" if fired == 0 else f"fired {fired}x on silence"))
        guarded("wake", do_wake)
    else:
        add(Check("wake", "SKIP"))

    if stt is not None and audio16 is not None:
        def do_stt():
            r = bench_stt(stt.transcribe, audio16, runs=2)
            ok = expect_word in r["text"].lower()
            add(Check("stt", "PASS" if ok else "FAIL", f"{r['ms_median']} ms (rtf {r['rtf']}) heard: {r['text']!r}", r))
        guarded("stt", do_stt)
    else:
        add(Check("stt", "SKIP"))

    if llm is not None:
        def do_llm():
            r = bench_llm(llm, system_prompt)
            add(Check("llm", "PASS" if r["chunks"] > 0 else "FAIL", f"first token {r['ttft_ms']} ms, {r['chunks_per_s']} chunks/s", r))
        guarded("llm", do_llm)
    else:
        add(Check("llm", "SKIP"))
    return out
