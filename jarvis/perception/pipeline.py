"""Voice pipeline: wake -> listen/endpoint -> STT -> LLM stream -> sentence chunks -> TTS -> speaker.

One asyncio loop consumes mic frames continuously; a turn (STT..playback) runs as a cancellable task
alongside it, so barge-in is just "speech detected while a turn is live -> cancel + flush + re-listen".
All timing logic is frame-count based (deterministic); wall clock is used only for latency traces.
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures as cf
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable

import numpy as np

from .audio.dsp import rms
from .audio.vad import Endpointer, Ev
from .broker import Broker, Outcome, parse_yes_no
from .fastpath import FastMatch
from .intents import IntentError, StreamSplitter, parse_intent
from .llm import Conversation
from .sentence import SentenceChunker
from .utterance import looks_complete
from .trace import Tracer, TurnTrace

_DONE = object()


def _leaf(e: BaseException) -> BaseException:
    """TaskGroup wraps failures in ExceptionGroup; surface the real cause in logs and events."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return e


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    BUSY = "busy"


@dataclass
class TurnCtx:
    tt: TurnTrace
    user_text: str = ""
    raw_out: list[str] = field(default_factory=list)
    sentences: list[tuple[str, int]] = field(default_factory=list)  # (text, end_sample_rel)
    base: int = 0
    stage: str = "stt"                 # stt | llm | other: which stage an exception came from


class Pipeline:
    def __init__(self, cfg, *, speaker, vad, wake, stt, llm, tts, broker: Broker, tracer: Tracer,
                 conversation: Conversation, on_event: Callable[[str, dict], None] | None = None, fastpath=None):
        self.cfg, self.speaker, self.vad, self.wake = cfg, speaker, vad, wake
        self.fastpath = fastpath
        self.stt, self.llm, self.tts, self.broker = stt, llm, tts, broker
        self.tracer, self.conv, self.on_event = tracer, conversation, on_event
        self.events: list[tuple[str, dict]] = []

        self.frame_ms = 1000.0 * cfg.audio.frame_samples / cfg.audio.sample_rate
        f = self._frames
        v = cfg.vad
        self.gate = Endpointer(v.threshold, v.start_frames, f(v.endpoint_ms), f(v.min_utterance_ms),
                               f(v.max_utterance_s * 1000), f(v.no_speech_timeout_s * 1000))
        self._preroll_frames = f(cfg.audio.preroll_ms)
        self._followup_frames = f(cfg.wake.followup_window_s * 1000) if cfg.wake.followup_window_s > 0 else 0
        self._confirm_frames = f(cfg.broker.confirm_ttl_s * 1000)
        self._cooldown_frames = f(cfg.wake.cooldown_ms)
        self._guard_frames = f(cfg.audio.post_speech_guard_ms) if cfg.audio.post_speech_guard_ms > 0 else 0
        self._guard_until = 0
        self._barge_needed = f(cfg.barge.min_speech_ms)
        spec = max(5, f(v.spec_ms)) if v.spec_ms > 0 else 0   # >= 5 so the trimmed audio matches the final audio
        self._spec_frames = spec if spec < self.gate.endpoint_frames else 0
        self._spec: tuple[asyncio.Task, int] | None = None
        self._spec_tasks: set[asyncio.Task] = set()   # every speculative STT task ever started, so shutdown can reap them
        self._stt_busy = False

        self.state = State.IDLE
        self._tick = 0
        self._wake_block_until = 0
        self._utt: list[np.ndarray] = []
        self._tt: TurnTrace | None = None
        self._wake_t: float | None = None
        self._turn_task: asyncio.Task | None = None
        self._ctx: TurnCtx | None = None
        self._barge_buf: collections.deque = collections.deque(maxlen=self._barge_needed + self._preroll_frames + 8)
        self._barge_run = 0
        self._proactive: collections.deque[str] = collections.deque()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stt_pool = cf.ThreadPoolExecutor(1, thread_name_prefix="stt")
        self._tts_pool = cf.ThreadPoolExecutor(1, thread_name_prefix="tts")
        self._act_pool = cf.ThreadPoolExecutor(2, thread_name_prefix="act")

    # ---------- helpers ----------
    def _frames(self, ms: float) -> int:
        return max(1, math.ceil(ms / self.frame_ms))

    def emit(self, name: str, **data) -> None:
        self.events.append((name, data))
        if self.on_event:
            self.on_event(name, data)

    def names(self) -> list[str]:
        return [n for n, _ in self.events]

    def say(self, text: str) -> None:
        """Thread-safe proactive speech (timers etc). Spoken when the pipeline is next idle."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._proactive.append, text)

    async def _thread(self, pool, fn, *a):
        return await asyncio.get_running_loop().run_in_executor(pool, fn, *a)

    # ---------- main loop ----------
    async def run(self, frames: AsyncIterator[np.ndarray]) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            async for frame in frames:
                await self._on_frame(frame)
            if self._turn_task is not None:  # source ended: let the in-flight reply finish
                await asyncio.gather(self._turn_task, return_exceptions=True)
        finally:
            self._spec = None
            pending = list(self._spec_tasks)
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self._cancel_turn()
            for p in (self._stt_pool, self._tts_pool, self._act_pool):
                p.shutdown(wait=False, cancel_futures=True)

    async def _on_frame(self, frame: np.ndarray) -> None:
        self._tick += 1
        p = self.vad.prob(frame)
        if self.state is State.BUSY:
            await self._busy_step(frame, p)
        elif self.state is State.IDLE:
            if self._proactive:
                self._start_task(self._run_say_turn(self._proactive.popleft(), self.tracer.start_turn("proactive")))
            elif self._tick >= self._wake_block_until and self.wake.process(frame):
                self._wake_t = self.tracer.clock()
                self.emit("wake")
                self._enter_listening(self.gate.default_no_speech_frames)
        elif self._tick >= self._guard_until:
            await self._listening_step(frame, p)

    def _enter_listening(self, no_speech_frames: int | None) -> None:
        self.state = State.LISTENING
        self._utt = []
        self._spec = None
        self.gate.reset(no_speech_frames)
        self.vad.reset()

    def _go_idle(self, why: str) -> None:
        self.state = State.IDLE
        self._utt = []
        self._spec = None
        self._wake_t = None
        self.wake.reset()
        self.emit("idle", why=why)

    # ---------- listening ----------
    async def _listening_step(self, frame: np.ndarray, p: float) -> None:
        self._utt.append(frame)
        ev = self.gate.update(p)
        if ev is None:
            self._speculate()
            return
        if ev is Ev.SPEECH_START:
            keep = self.cfg.vad.start_frames + self._preroll_frames
            self._utt = self._utt[-keep:]
            self._tt = self.tracer.start_turn()
            if self._wake_t is not None:
                self._tt.mark("wake", self._wake_t)
            self._tt.mark("speech_start")
            self.emit("speech_start")
        elif ev is Ev.SPEECH_END:
            self._finish_utterance("speech_end", max(0, self.gate.endpoint_frames - 4), self.gate.endpoint_frames)
        elif ev is Ev.MAX_LEN:
            self._finish_utterance("max_len", 0, 0)
        elif ev is Ev.DISCARD:
            self.emit("discard")
            self._tt = None
            self._enter_listening(self.gate.no_speech_frames)
        elif ev is Ev.TIMEOUT:
            self._go_idle("no_speech")

    def _speculate(self) -> None:
        """Speculative STT during trailing silence, plus optional fast-commit."""
        sf = self._spec_frames
        if not sf or not self.gate.in_speech:
            return
        sil = self.gate.silence_frames
        if sil == 0:
            self._spec = None  # speech resumed: the speculative transcript is stale
            return
        if sil >= sf and self._spec is None and not self._stt_busy:
            trim = max(0, sil - 4)
            audio = np.concatenate(self._utt[: len(self._utt) - trim] if trim else self._utt)
            self._stt_busy = True

            async def work():
                try:
                    return await self._thread(self._stt_pool, self.stt.transcribe, audio)
                finally:
                    self._stt_busy = False

            task = asyncio.create_task(work())
            self._spec_tasks.add(task)
            task.add_done_callback(self._spec_tasks.discard)
            task.add_done_callback(lambda t: t.cancelled() or t.exception())  # mark retrieved
            self._spec = (task, len(audio))
        elif self._spec is not None and self.cfg.vad.fast_commit and self._spec[0].done():
            t = self._spec[0]
            if not t.cancelled() and t.exception() is None and looks_complete(t.result()):
                self._finish_utterance("fast_commit", max(0, sil - 4), sil)

    def _finish_utterance(self, reason: str, trim: int, silence_frames: int) -> None:
        audio = np.concatenate(self._utt[: len(self._utt) - trim] if trim else self._utt)
        pre = self._spec[0] if (self._spec is not None and self._spec[1] == len(audio)) else None
        self._spec = None
        tt = self._tt or self.tracer.start_turn()
        tt.mark("speech_end")
        tt.meta["endpoint_ms"] = round(silence_frames * self.frame_ms)   # the silence the user actually sat through
        tt.meta["audio_ms"] = round(len(audio) * 1000 / self.cfg.audio.sample_rate)
        self.emit("speech_end", reason=reason, audio_ms=tt.meta["audio_ms"])
        self._utt, self._tt = [], None
        self._start_task(self._run_turn(audio, tt, pre))

    def _start_task(self, coro) -> None:
        self.state = State.BUSY
        self._barge_buf.clear()
        self._barge_run = 0
        self._turn_task = asyncio.create_task(coro)

    # ---------- barge-in ----------
    async def _busy_step(self, frame: np.ndarray, p: float) -> None:
        if not self.cfg.barge.enabled:
            return
        self._barge_buf.append((frame, p))
        loud = rms(frame) >= self.cfg.barge.min_rms
        self._barge_run = self._barge_run + 1 if (p >= self.cfg.vad.threshold and loud) else 0
        if self._barge_run >= self._barge_needed:
            seed = list(self._barge_buf)
            await self._cancel_turn()
            self.emit("barge_in")
            self._enter_listening(self.gate.default_no_speech_frames)
            for f, pp in seed:
                await self._listening_step(f, pp)

    async def _cancel_turn(self) -> None:
        t, self._turn_task = self._turn_task, None
        if t is None:
            return
        if not t.done():
            t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        self.speaker.flush()

    # ---------- turns ----------
    def _on_turn_done(self) -> None:
        self._turn_task = None
        awaiting = self.broker.has_pending("confirm")
        window = self._confirm_frames if awaiting else self._followup_frames
        self._wake_block_until = self._tick + self._cooldown_frames
        self._guard_until = self._tick + self._guard_frames
        if window:
            self._enter_listening(window)
        else:
            self._go_idle("turn_done")

    async def _run_turn(self, audio: np.ndarray, tt: TurnTrace, pre: asyncio.Task | None = None) -> None:
        ctx = TurnCtx(tt)
        self._ctx = ctx
        ctx.base = self.speaker.total_enqueued
        loop = asyncio.get_running_loop()
        self.speaker.on_first_audio(lambda t: loop.call_soon_threadsafe(tt.mark, "audio_first_out", t))
        try:
            await self._turn_body(audio, ctx, pre)
        except asyncio.CancelledError:
            self._commit(ctx, interrupted=True)
            tt.finish(interrupted=True)
            raise
        except Exception as e:  # noqa: BLE001
            e = _leaf(e)
            self.emit("error", error=f"{type(e).__name__}: {e}", stage=ctx.stage)
            await self._apologize(ctx)
            tt.finish(error=f"{type(e).__name__}: {e}")
            self._on_turn_done()
            return
        self._commit(ctx, interrupted=False)
        tt.finish()
        self.emit("turn_done")
        self._on_turn_done()

    ERROR_LINES = {"stt": "Sorry, I didn't catch that.", "llm": "Sorry, I can't reach my language model right now."}

    async def _apologize(self, ctx: TurnCtx) -> None:
        line = self.ERROR_LINES.get(ctx.stage)
        if not line:
            return
        try:  # never let the apology itself take the loop down
            await self._say_text(line, ctx)
            await self.speaker.drain()
        except Exception:  # noqa: BLE001
            pass

    async def _run_say_turn(self, text: str, tt: TurnTrace) -> None:
        ctx = TurnCtx(tt)
        ctx.base = self.speaker.total_enqueued
        loop = asyncio.get_running_loop()
        self.speaker.on_first_audio(lambda t: loop.call_soon_threadsafe(tt.mark, "audio_first_out", t))
        try:
            self.emit("proactive", text=text)
            await self._say_text(text, ctx)
            await self.speaker.drain()
        except asyncio.CancelledError:
            tt.finish(interrupted=True)
            raise
        except Exception as e:  # noqa: BLE001
            self.emit("error", error=f"{type(e).__name__}: {e}")
        else:
            tt.finish()
            self._on_turn_done()

    def _commit(self, ctx: TurnCtx, interrupted: bool) -> None:
        if not ctx.user_text:
            return
        if interrupted:
            played = self.speaker.total_played - ctx.base
            spoken = " ".join(s for s, end in ctx.sentences if end <= played)
            self.emit("interrupted", spoken=spoken)
            self.conv.commit(ctx.user_text, spoken, interrupted=True)
        else:
            self.conv.commit(ctx.user_text, "\n".join(ctx.raw_out))

    async def _turn_body(self, audio: np.ndarray, ctx: TurnCtx, pre: asyncio.Task | None = None) -> None:
        tt = ctx.tt
        ctx.stage = "stt"
        text = None
        if pre is not None:
            try:
                text = await pre
                tt.meta["spec"] = True
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - speculative attempt failed; fall back to a normal transcription
                text = None
        if text is None:
            text = await self._thread(self._stt_pool, self.stt.transcribe, audio)
        tt.mark("stt_done")
        ctx.stage = "other"
        tt.meta["text"] = text
        self.emit("transcript", text=text)
        if not text.strip():
            return
        ctx.user_text = text
        if self.broker.has_pending("confirm"):
            ans = parse_yes_no(text)
            if ans is True and self.cfg.barge.enabled and rms(audio) < self.cfg.barge.min_rms:
                # Quieter than speaker bleed: this is probably the assistant's own "say yes to confirm" echoing
                # back. Only an affirmative the mic hears *louder than the speaker* may execute anything.
                self.emit("confirm_ignored", reason="below_bleed_level", rms=round(rms(audio), 4))
                ctx.user_text = ""
                return
            if ans is not None:  # the broker, not the LLM, adjudicates confirmation
                out = await self._thread(self._act_pool, self.broker.resolve_confirm, ans)
                self.emit("confirm", answer=ans, status=out.status)
                await self._say_text(out.say or ("Done." if out.status == "executed" else ""), ctx)
                await self.speaker.drain()
                ctx.user_text = ""  # confirmations don't enter LLM history
                return
            await self._thread(self._act_pool, self.broker.cancel_all, "superseded")
        if self.fastpath is not None:
            fm = self.fastpath.match(text)
            if fm is not None:
                await self._run_fast(fm, ctx)
                await self.speaker.drain()
                return
        ctx.stage = "llm"
        await self._converse(text, ctx)
        await self.speaker.drain()

    async def _run_fast(self, fm: FastMatch, ctx: TurnCtx) -> None:
        tt = ctx.tt
        tt.meta["fast"] = True
        self.emit("fastpath", action=fm.intent.action if fm.intent else None)
        if fm.intent is None:
            spoken = fm.say
        else:
            out = await self._thread(self._act_pool, self.broker.propose, fm.intent)
            self.emit("intent", action=fm.intent.action, status=out.status, tainted=False)
            if out.status == "executed":
                spoken = fm.render(out.result) if fm.render else "Okay."
            else:
                spoken = out.say
        if spoken:
            tt.mark("first_sentence")
            await self._say_text(spoken, ctx)
            ctx.raw_out.append(spoken)

    # ---------- LLM round(s) ----------
    async def _converse(self, text: str, ctx: TurnCtx) -> None:
        msgs = self.conv.messages_for(text)
        tainted = False
        for rnd in range(self.cfg.llm.max_tool_rounds + 1):
            raw, intents = await self._stream_round(msgs, ctx)
            ctx.raw_out.append(raw)
            if not intents:
                return
            outcomes = await self._run_intents(intents, tainted)
            for o in outcomes:
                if o.say:
                    await self._say_text(o.say, ctx)
            obs = [o for o in outcomes if o.status == "executed" and o.result and o.result.output]
            if not obs or rnd >= self.cfg.llm.max_tool_rounds:
                return
            block = "\n".join(f"[{o.decision.spec.name}] {o.result.output[:800]}" for o in obs)[:1600]
            msgs = msgs + [{"role": "assistant", "content": raw},
                           {"role": "user", "content": f'<action_results untrusted="true">\n{block}\n</action_results>'}]
            tainted = True  # context now contains tool output: any intents it yields are escalated

    async def _stream_round(self, msgs: list[dict], ctx: TurnCtx) -> tuple[str, list[str]]:
        tt = ctx.tt
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        splitter = StreamSplitter()
        t = self.cfg.tts
        chunker = SentenceChunker(t.first_min_chars, t.min_chars, t.first_clause_min_chars, t.clause_min_chars)
        raw_parts: list[str] = []
        intents: list[str] = []

        async def producer():
            first = True
            async for tok in self.llm.stream(msgs):
                if first:
                    tt.mark("llm_first_token")
                    first = False
                raw_parts.append(tok)
                speech, found = splitter.feed(tok)
                intents.extend(found)
                for s in chunker.feed(speech):
                    tt.mark("first_sentence")
                    await q.put(s)
            speech, found = splitter.flush()
            intents.extend(found)
            for s in chunker.feed(speech) + chunker.flush():
                tt.mark("first_sentence")
                await q.put(s)
            await q.put(_DONE)

        async def consumer():
            while (s := await q.get()) is not _DONE:
                await self._speak_sentence(s, ctx)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(producer())
            tg.create_task(consumer())
        return "".join(raw_parts), intents

    # ---------- speech out ----------
    async def _say_text(self, text: str, ctx: TurnCtx) -> None:
        t = self.cfg.tts
        ch = SentenceChunker(t.first_min_chars, t.min_chars, t.first_clause_min_chars, t.clause_min_chars)
        for s in ch.feed(text) + ch.flush():
            await self._speak_sentence(s, ctx)

    async def _speak_sentence(self, sentence: str, ctx: TurnCtx) -> None:
        it = iter(self.tts.synth(sentence))
        first = True
        while True:
            chunk = await self._thread(self._tts_pool, next, it, None)
            if chunk is None:
                break
            if first:
                ctx.tt.mark("tts_first_pcm")
                first = False
            await self.speaker.play(chunk)
        ctx.sentences.append((sentence, self.speaker.total_enqueued - ctx.base))

    # ---------- intents ----------
    async def _run_intents(self, raws: list[str], tainted: bool) -> list[Outcome]:
        prov = ("tool_result",) if tainted else ("user_voice",)
        outs: list[Outcome] = []
        for raw in raws[: self.cfg.broker.max_intents_per_turn]:
            try:
                intent = parse_intent(raw, prov)
            except IntentError as e:
                self.emit("intent_error", error=str(e))
                outs.append(Outcome("denied", "I produced a malformed action, so I ignored it."))
                continue
            out = await self._thread(self._act_pool, self.broker.propose, intent)
            self.emit("intent", action=intent.action, status=out.status, tainted=intent.tainted)
            outs.append(out)
        return outs
