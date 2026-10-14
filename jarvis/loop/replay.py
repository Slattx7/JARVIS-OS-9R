from __future__ import annotations
import json, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Protocol

from .checker import Checker
from .gate import Outcome
from .prompt import build_messages, extract_calls
from .refs import ModelRef
from .traces import Trace
from .util import read_json, write_json


class Replayer(Protocol):
    def prepare(self, ref: ModelRef) -> None: ...
    def replay(self, ref: ModelRef, trace: Trace) -> Outcome: ...


Propose = Callable[[ModelRef, list], str]      # (model, chat messages) -> raw model text


class ProposalReplayer:
    """Dry-run replay: the model PROPOSES tool calls for a recorded task; nothing is ever executed.
    Scored against the recorded gold calls + the tier policy."""

    def __init__(self, propose: Propose, checker: Checker, activate: Optional[Callable[[ModelRef], None]] = None):
        self.propose, self.checker, self.activate = propose, checker, activate

    def prepare(self, ref: ModelRef) -> None:
        if self.activate:
            self.activate(ref)

    def replay(self, ref: ModelRef, trace: Trace) -> Outcome:
        msgs = build_messages(trace.task, trace.context, trace.granted_tier)
        t0 = time.perf_counter()
        try:
            text = self.propose(ref, msgs)
        except Exception as e:                  # timeouts, server down, bad exit code
            ms = (time.perf_counter() - t0) * 1000
            return Outcome(False if trace.success else None, 0, ms, error=f"{type(e).__name__}: {e}"[:300])
        ms = (time.perf_counter() - t0) * 1000
        ok, viol = self.checker.check(trace, extract_calls(text))
        return Outcome(ok, viol, ms)


class OpenAICompat:
    """Any OpenAI-compatible local server (llama.cpp server, Ollama /v1, vLLM, LM Studio)."""

    def __init__(self, base_url="http://127.0.0.1:8080/v1", model_template="{version}", timeout=60.0,
                 max_tokens=384, temperature=0.0, api_key: Optional[str] = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.tpl, self.timeout, self.max_tokens, self.temperature, self.api_key = (
            model_template, timeout, max_tokens, temperature, api_key)

    def __call__(self, ref: ModelRef, messages: list) -> str:
        body = {"model": self.tpl.format(version=ref.version, artifact=ref.artifact), "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]


class CommandPropose:
    """argv template with {version}/{artifact}; chat messages JSON on stdin, raw model text on stdout."""

    def __init__(self, argv: list, timeout: float = 120.0):
        self.argv, self.timeout = argv, timeout

    def __call__(self, ref: ModelRef, messages: list) -> str:
        argv = [a.format(version=ref.version, artifact=ref.artifact) for a in self.argv]
        r = subprocess.run(argv, input=json.dumps(messages), capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}: {r.stderr[-200:]}")
        return r.stdout


class CommandActivator:
    """Swap the served model/adapter before replaying a version (restart llama-server with --lora, call an API, ...)."""

    def __init__(self, argv: list, timeout: float = 300.0):
        self.argv, self.timeout = argv, timeout

    def __call__(self, ref: ModelRef) -> None:
        argv = [a.format(version=ref.version, artifact=ref.artifact) for a in self.argv]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"activate {ref.version} failed: exit {r.returncode}: {r.stderr[-200:]}")


class OutcomeCache:
    """(version, trace id, trace key) -> Outcome. Baseline replays are paid for once, not every cycle. Errors never cached."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.d: dict = (read_json(self.path, {}) if self.path else {}) or {}

    @staticmethod
    def key(ref: ModelRef, t: Trace) -> str:
        return f"{ref.version}|{t.id}|{t.key}"

    def get(self, ref: ModelRef, t: Trace) -> Optional[Outcome]:
        v = self.d.get(self.key(ref, t))
        return Outcome(**v) if v else None

    def put(self, ref: ModelRef, t: Trace, o: Outcome) -> None:
        self.d[self.key(ref, t)] = asdict(o)

    def save(self) -> None:
        if self.path:
            write_json(self.path, self.d)


def replay_all(replayer: Replayer, ref: ModelRef, traces: list, cache: Optional[OutcomeCache] = None,
               workers: int = 1) -> dict:
    out: dict = {}
    todo = []
    for t in traces:
        hit = cache.get(ref, t) if cache else None
        if hit is not None:
            out[t.id] = hit
        else:
            todo.append(t)
    if todo:
        replayer.prepare(ref)
        if workers > 1:
            with ThreadPoolExecutor(workers) as ex:
                results = list(ex.map(lambda t: replayer.replay(ref, t), todo))
        else:
            results = [replayer.replay(ref, t) for t in todo]
        for t, o in zip(todo, results):
            out[t.id] = o
            if cache and not o.error:
                cache.put(ref, t, o)
        if cache:
            cache.save()
    return out
