"""Mic/speaker backends. Real: sounddevice. Test: ScriptedMic / NullSpeaker.

Speaker contract (shared by both):
  play(pcm) async   - enqueue int16 mono @ sample_rate, applies backpressure
  drain()   async   - wait until everything enqueued has been played
  flush()           - drop queued audio instantly (barge-in); aligns counters
  total_enqueued / total_played - monotonic sample counters (played==enqueued after drain/flush)
  on_first_audio(fn) - one-shot fn(t_monotonic) when the next audio actually starts playing
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import AsyncIterator, Callable

import numpy as np


class ScriptedMic:
    def __init__(self, frames: list[np.ndarray], pace_s: float = 0.0, tail_idle_s: float = 0.0):
        self._frames = frames
        self._pace = pace_s
        self._tail = tail_idle_s

    async def frames(self) -> AsyncIterator[np.ndarray]:
        for f in self._frames:
            yield f
            await asyncio.sleep(self._pace)
        if self._tail:
            await asyncio.sleep(self._tail)


class SoundDeviceMic:
    def __init__(self, sample_rate: int, frame_samples: int, device=None):
        self.sr, self.n, self.device = sample_rate, frame_samples, device
        self.dropped = 0

    async def frames(self) -> AsyncIterator[np.ndarray]:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)

        def put(frame):
            if q.full():
                q.get_nowait()
                self.dropped += 1
            q.put_nowait(frame)

        def cb(indata, n, t, status):
            try:
                loop.call_soon_threadsafe(put, indata[:, 0].copy())
            except RuntimeError:
                pass  # loop closed during shutdown

        with sd.InputStream(samplerate=self.sr, blocksize=self.n, channels=1, dtype="int16",
                            device=self.device, callback=cb):
            while True:
                yield await q.get()


class SoundDeviceSpeaker:
    def __init__(self, sample_rate: int, device=None, blocksize: int = 480, max_buffer_s: float = 1.5):
        self.sr, self.device, self.blocksize, self.max_buffer_s = sample_rate, device, blocksize, max_buffer_s
        self._chunks: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._buffered = 0
        self.total_enqueued = 0
        self.total_played = 0
        self._first_cb: Callable[[float], None] | None = None
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(samplerate=self.sr, blocksize=self.blocksize, channels=1,
                                       dtype="int16", device=self.device, callback=self._cb)
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def on_first_audio(self, fn: Callable[[float], None]) -> None:
        with self._lock:
            self._first_cb = fn

    def _cb(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        pos, need = 0, frames
        fire = None
        with self._lock:
            while need and self._chunks:
                c = self._chunks[0]
                take = min(len(c), need)
                out[pos:pos + take] = c[:take]
                pos += take
                need -= take
                if take == len(c):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = c[take:]
            self._buffered -= pos
            self.total_played += pos
            if pos and self._first_cb:
                fire, self._first_cb = self._first_cb, None
        if need:
            out[pos:] = 0
        if fire:
            fire(time.monotonic())

    async def play(self, pcm: np.ndarray) -> None:
        while self._buffered > self.max_buffer_s * self.sr:
            await asyncio.sleep(0.01)
        with self._lock:
            self._chunks.append(pcm)
            self._buffered += len(pcm)
            self.total_enqueued += len(pcm)

    async def drain(self) -> None:
        while self._buffered > 0:
            await asyncio.sleep(0.02)
        await asyncio.sleep(self.blocksize / self.sr)

    def flush(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._buffered = 0
            self.total_played = self.total_enqueued
            self._first_cb = None


class NullSpeaker:
    """Simulated speaker. realtime=True models playback duration with a virtual clock so tests can
    exercise drain/barge-in timing; realtime=False plays instantly."""

    def __init__(self, sample_rate: int = 24000, realtime: bool = False, max_buffer_s: float = 1.5):
        self.sr, self.realtime, self.max_buffer_s = sample_rate, realtime, max_buffer_s
        self.total_enqueued = 0
        self._busy_until = 0.0
        self._first_cb: Callable[[float], None] | None = None
        self.log: list[tuple[float, int]] = []
        self.flush_count = 0
        self.flushed_samples = 0

    def start(self) -> None: ...
    def close(self) -> None: ...

    @property
    def total_played(self) -> int:
        if not self.realtime:
            return self.total_enqueued
        remaining = max(0.0, self._busy_until - time.monotonic()) * self.sr
        return self.total_enqueued - int(remaining)

    def on_first_audio(self, fn) -> None:
        self._first_cb = fn

    async def play(self, pcm: np.ndarray) -> None:
        now = time.monotonic()
        if self.realtime:
            while self._busy_until - time.monotonic() > self.max_buffer_s:
                await asyncio.sleep(0.005)
            now = time.monotonic()
        start = max(now, self._busy_until)
        if self.realtime:
            self._busy_until = start + len(pcm) / self.sr
        self.total_enqueued += len(pcm)
        self.log.append((start, len(pcm)))
        if self._first_cb:
            fn, self._first_cb = self._first_cb, None
            fn(start)

    async def drain(self) -> None:
        if self.realtime:
            delay = self._busy_until - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    def flush(self) -> None:
        self.flush_count += 1
        if self.realtime:
            self.flushed_samples += self.total_enqueued - self.total_played
            self._busy_until = 0.0
        self._first_cb = None


class WavMic:
    """Feeds a wav file through the pipeline as if it were a live mic (then `tail_s` of silence so the last
    utterance can endpoint). Pair with WavSpeaker to run the full real stack with no audio hardware."""

    def __init__(self, path, sample_rate: int = 16000, frame_samples: int = 512, tail_s: float = 2.5, lead_s: float = 0.5):
        from .dsp import read_wav

        x = read_wav(path, sample_rate)
        pad = lambda sec: np.zeros(int(sample_rate * sec), dtype=np.int16)  # noqa: E731
        self._sig = np.concatenate([pad(lead_s), x, pad(tail_s)])
        self._n = frame_samples

    @property
    def seconds(self) -> float:
        return len(self._sig) / 16000

    async def frames(self) -> AsyncIterator[np.ndarray]:
        for i in range(0, len(self._sig) - self._n + 1, self._n):
            yield self._sig[i:i + self._n]
            await asyncio.sleep(0)


class WavSpeaker(NullSpeaker):
    """NullSpeaker that keeps the audio so it can be written out as a wav."""

    def __init__(self, sample_rate: int = 24000):
        super().__init__(sample_rate, realtime=False)
        self.chunks: list[np.ndarray] = []

    async def play(self, pcm: np.ndarray) -> None:
        self.chunks.append(np.asarray(pcm, dtype=np.int16).copy())
        await super().play(pcm)

    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.int16)
