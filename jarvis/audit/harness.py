"""Replay harness.

    report = run_suite(store, agent, select={"label": "golden"}, live={"llm", "permission"})
    sys.exit(report.exit_code())

`agent(rt, inputs)` is the orchestration entrypoint (sync or async). It must talk to components through
`rt.call(...)` / @traced so that frozen kinds can be served from the recording.

Baseline traces are never mutated: candidate traces go to a separate (default in-memory) store that
shares the baseline's key/redaction config, so input hashes are comparable.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .runtime import DEFAULT_LIVE, TraceRuntime
from .schema import Trace, canonical, kind_value, percentile
from .scorers import REGRESSION, SAFETY, ScoreContext, ScoreResult, Scorer, default_scorers
from .store import Store


@dataclass
class CaseResult:
    trace_id: str
    name: str
    verdict: str  # pass | fail | safety_fail | error
    scores: List[ScoreResult] = field(default_factory=list)
    divergences: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    candidate_trace_id: Optional[str] = None
    baseline_pre_tts_ms: float = 0.0
    candidate_pre_tts_ms: float = 0.0
    baseline_actions: List[str] = field(default_factory=list)
    candidate_actions: List[str] = field(default_factory=list)


def _action_lines(tr: Trace) -> List[str]:
    out = []
    for s in tr.spans_of("action"):
        a = canonical(s.inputs)
        t = s.meta.get("tier")
        out.append(f"{s.name} {a if len(a) <= 80 else a[:79] + '…'} [{s.status}{'' if t is None else f', T{t}'}]")
    return out


class GateFailure(AssertionError):
    pass


@dataclass
class Report:
    label: str
    created_ns: int
    min_pass: float
    live: List[str]
    on_miss: str
    cases: List[CaseResult] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    # -- aggregates --------------------------------------------------------- #
    def count(self, verdict: str) -> int:
        return sum(1 for c in self.cases if c.verdict == verdict)

    @property
    def pass_rate(self) -> float:
        return self.count("pass") / len(self.cases) if self.cases else 1.0

    @property
    def verdict(self) -> str:
        if self.count("safety_fail"):
            return "safety_fail"
        if self.pass_rate < self.min_pass:
            return "fail"
        return "pass"

    def exit_code(self) -> int:
        return {"pass": 0, "fail": 1, "safety_fail": 2}[self.verdict]

    def raise_on_fail(self) -> "Report":
        """pytest/unittest hook: `run_suite(...).raise_on_fail()` raises GateFailure with the full summary."""
        if self.exit_code():
            raise GateFailure(self.summary_text())
        return self

    def failures(self, severity: str) -> List[str]:
        out = []
        for c in self.cases:
            for s in c.scores:
                if s.severity == severity and not s.passed:
                    out.append(f"{c.trace_id} ({c.name}) [{s.scorer}] {s.detail}".rstrip())
            if severity == REGRESSION and c.error:
                out.append(f"{c.trace_id} ({c.name}) [agent_error] {c.error}")
        return out

    def scorer_table(self) -> List[Dict[str, Any]]:
        agg: Dict[str, Dict[str, Any]] = {}
        for c in self.cases:
            for s in c.scores:
                a = agg.setdefault(s.scorer, {"scorer": s.scorer, "severity": s.severity, "n": 0, "passed": 0, "sum": 0.0})
                a["n"] += 1
                a["passed"] += int(s.passed)
                a["sum"] += s.score
        return [{"scorer": a["scorer"], "severity": a["severity"], "passed": a["passed"], "n": a["n"],
                 "mean_score": a["sum"] / a["n"] if a["n"] else 1.0} for a in agg.values()]

    def latency(self) -> Dict[str, float]:
        b = [c.baseline_pre_tts_ms for c in self.cases]
        d = [c.candidate_pre_tts_ms for c in self.cases]
        return {"baseline_p50": percentile(b, 50), "baseline_p95": percentile(b, 95),
                "candidate_p50": percentile(d, 50), "candidate_p95": percentile(d, 95)}

    # -- rendering ---------------------------------------------------------- #
    def summary_line(self) -> str:
        return (f"{self.verdict.upper().replace('_', ' ')}: {len(self.cases)} cases | {self.count('pass')} pass | "
                f"{self.count('fail')} fail | {self.count('error')} error | {self.count('safety_fail')} safety-fail | "
                f"pass rate {self.pass_rate:.1%} (min {self.min_pass:.0%})")

    def summary_text(self) -> str:
        L = [f"replay: {self.label}", self.summary_line()]
        for sev, title in ((SAFETY, "SAFETY"), (REGRESSION, "REGRESSION")):
            f = self.failures(sev)
            if f:
                L.append(f"-- {title} ({len(f)})")
                L += [f"  {x}" for x in f[:15]]
                if len(f) > 15:
                    L.append(f"  … {len(f) - 15} more")
        if self.skipped:
            L.append(f"-- skipped {len(self.skipped)} baseline traces (status != ok)")
        lat = self.latency()
        L.append(f"pre-TTS est. p50/p95: baseline {lat['baseline_p50']:.0f}/{lat['baseline_p95']:.0f} ms -> "
                 f"candidate {lat['candidate_p50']:.0f}/{lat['candidate_p95']:.0f} ms")
        return "\n".join(L)

    def to_markdown(self) -> str:
        L = [f"# Replay report: {self.label}", "", f"**{self.summary_line()}**", ""]
        L += ["| scorer | severity | passed | mean score |", "|---|---|---|---|"]
        for r in self.scorer_table():
            L.append(f"| {r['scorer']} | {r['severity']} | {r['passed']}/{r['n']} | {r['mean_score']:.3f} |")
        for sev, title in ((SAFETY, "Safety failures"), (REGRESSION, "Regressions")):
            f = self.failures(sev)
            L += ["", f"## {title} ({len(f)})"] + ([f"- {x}" for x in f] if f else ["none"])
        div: Dict[str, int] = {}
        for c in self.cases:
            for d in c.divergences:
                div[d["reason"]] = div.get(d["reason"], 0) + 1
        L += ["", "## Divergences", ", ".join(f"{k}={v}" for k, v in sorted(div.items())) or "none"]
        lat = self.latency()
        L += ["", "## Latency (estimated wake to first audio, ms)", "",
              "| | p50 | p95 |", "|---|---|---|",
              f"| baseline | {lat['baseline_p50']:.0f} | {lat['baseline_p95']:.0f} |",
              f"| candidate | {lat['candidate_p50']:.0f} | {lat['candidate_p95']:.0f} |",
              "", f"live kinds: {', '.join(self.live)} · on_miss: {self.on_miss}"]
        return "\n".join(L) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "created_ns": self.created_ns, "verdict": self.verdict,
            "exit_code": self.exit_code(), "pass_rate": self.pass_rate, "min_pass": self.min_pass,
            "live": self.live, "on_miss": self.on_miss, "scorers": self.scorer_table(), "latency": self.latency(),
            "skipped": self.skipped,
            "cases": [{"trace_id": c.trace_id, "name": c.name, "verdict": c.verdict, "error": c.error,
                       "candidate_trace_id": c.candidate_trace_id, "divergences": c.divergences,
                       "baseline_pre_tts_ms": c.baseline_pre_tts_ms, "candidate_pre_tts_ms": c.candidate_pre_tts_ms,
                       "baseline_actions": c.baseline_actions, "candidate_actions": c.candidate_actions,
                       "scores": [{"scorer": s.scorer, "passed": s.passed, "score": s.score,
                                   "severity": s.severity, "detail": s.detail, "data": s.data} for s in c.scores]}
                      for c in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def _verdict(scores: List[ScoreResult], err: Optional[str]) -> str:
    if any(s.severity == SAFETY and not s.passed for s in scores):
        return "safety_fail"
    if err:
        return "error"
    if any(s.severity == REGRESSION and not s.passed for s in scores):
        return "fail"
    return "pass"


def run_case(base_store: Store, cand_store: Store, base: Trace, agent: Callable[..., Any], *,
             live: Iterable[str], on_miss: str, label: str, scorers: List[Scorer]) -> CaseResult:
    rt = TraceRuntime.for_replay(base, base_store, cand_store, live=live, on_miss=on_miss, label=label)
    err: Optional[str] = None
    try:
        if inspect.iscoroutinefunction(agent):
            async def _go() -> None:
                async with rt:
                    rt.set_output(await agent(rt, rt.inputs))
            asyncio.run(_go())
        else:
            with rt:
                rt.set_output(agent(rt, rt.inputs))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    cand = cand_store.load_trace(rt.trace_id)
    ctx = ScoreContext(base, cand, list(rt.divergences), err)
    scores: List[ScoreResult] = []
    for sc in scorers:
        try:
            scores.append(sc.score(ctx))
        except Exception as e:  # noqa: BLE001
            scores.append(ScoreResult(sc.name, False, 0.0, REGRESSION, f"scorer crashed: {type(e).__name__}: {e}"))
    return CaseResult(trace_id=base.trace_id, name=base.name, verdict=_verdict(scores, err), scores=scores,
                      divergences=[d.to_dict() for d in ctx.divergences], error=err,
                      candidate_trace_id=rt.trace_id, baseline_pre_tts_ms=base.first_audio_ms(),
                      candidate_pre_tts_ms=cand.first_audio_ms(), baseline_actions=_action_lines(base),
                      candidate_actions=_action_lines(cand))


def run_suite(store: Store, agent: Callable[..., Any], *, trace_ids: Optional[Iterable[str]] = None,
              select: Optional[Dict[str, Any]] = None, live: Iterable[Any] = DEFAULT_LIVE,
              on_miss: str = "positional", scorers: Optional[List[Scorer]] = None,
              candidate_store: Optional[Store] = None, label: str = "candidate", min_pass: float = 1.0,
              include_failed: bool = False, progress: Optional[Callable[[CaseResult], None]] = None) -> Report:
    live_set = sorted({kind_value(k) for k in live} - {"action"})
    scorers = default_scorers() if scorers is None else scorers
    if trace_ids is not None:
        bases = [store.load_trace(t) for t in trace_ids]
    else:
        sel = dict(select or {})
        bases = [store.load_trace(h.trace_id) for h in store.list_traces(**sel)]
    cand_store = candidate_store or store.clone_empty()
    if candidate_store is not None and candidate_store.fallback is None:
        candidate_store.fallback = store
    report = Report(label=label, created_ns=time.time_ns(), min_pass=min_pass, live=live_set, on_miss=on_miss)
    for base in bases:
        if base.status != "ok" and not include_failed:
            report.skipped.append({"trace_id": base.trace_id, "status": base.status})
            continue
        res = run_case(store, cand_store, base, agent, live=live_set, on_miss=on_miss, label=label, scorers=scorers)
        report.cases.append(res)
        if progress:
            progress(res)
    return report
