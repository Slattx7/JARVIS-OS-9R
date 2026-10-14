"""Recall-layer integration.

Record every retrieval as a `recall` span so a turn can be replayed without the graph (frozen), or with a *new*
retriever live (RecallOverlapScorer compares what it fetched to what production fetched).

    ctx = recall_call(rt, graph.query, text, k=8, graph_version=graph.version)

Contract: a recall span's inputs are the retriever's kwargs (`q`, `k`, ...), outputs a list of items; an item is a
string or a dict with an `id` (also `uid`/`key`/`node_id`). Put the graph/index version in `meta` so a regression can
be tied to a data change rather than a code change.

recall_eval() is the offline version: re-run a candidate retriever over every recorded recall query (no agent, no
LLM) and measure overlap with what was retrieved at the time. Recorded query text is stored redacted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .scorers import RecallOverlapScorer, item_id, overlap_metrics  # noqa: F401  (re-exported)
from .store import Store


def recall_call(rt: Any, fn: Callable[..., Any], q: str, k: int = 8, *, name: str = "graph.query",
                graph_version: Optional[str] = None, **extra: Any) -> Any:
    meta = {"graph_version": graph_version} if graph_version else None
    return rt.call("recall", name, {"q": q, "k": k, **extra}, fn, meta=meta)


@dataclass
class RecallEvalCase:
    trace_id: str
    span_id: str
    name: str
    jaccard: float = 0.0
    recall: float = 0.0
    top1_hit: float = 0.0
    n_base: int = 0
    n_cand: int = 0
    error: Optional[str] = None


@dataclass
class RecallEvalReport:
    cases: List[RecallEvalCase] = field(default_factory=list)
    min_jaccard: float = 0.0

    def _mean(self, attr: str) -> float:
        ok = [getattr(c, attr) for c in self.cases if c.error is None]
        return sum(ok) / len(ok) if ok else 0.0

    @property
    def errors(self) -> int:
        return sum(1 for c in self.cases if c.error)

    @property
    def mean_jaccard(self) -> float:
        return self._mean("jaccard")

    @property
    def passed(self) -> bool:
        return bool(self.cases) and self.errors == 0 and self.mean_jaccard >= self.min_jaccard

    def worst(self, n: int = 5) -> List[RecallEvalCase]:
        return sorted((c for c in self.cases if c.error is None), key=lambda c: c.jaccard)[:n]

    def summary(self) -> str:
        return (f"recall-eval: {len(self.cases)} queries | mean jaccard {self.mean_jaccard:.3f} | "
                f"mean recall {self._mean('recall'):.3f} | top1 hit {self._mean('top1_hit'):.1%} | "
                f"{self.errors} errors | {'PASS' if self.passed else 'FAIL'} (min jaccard {self.min_jaccard})")


def recall_eval(store: Store, retriever: Callable[..., Any], *, since_ns: Optional[int] = None,
                limit: Optional[int] = None, k: Optional[int] = None, name: Optional[str] = None,
                min_jaccard: float = 0.0) -> RecallEvalReport:
    rep = RecallEvalReport(min_jaccard=min_jaccard)
    for h in store.list_traces(since_ns=since_ns, limit=limit, newest_first=False):
        for s in store.load_trace(h.trace_id).spans_of("recall"):
            if s.status != "ok" or not isinstance(s.outputs, list) or (name and s.name != name):
                continue
            case = RecallEvalCase(h.trace_id, s.span_id, s.name, n_base=len(s.outputs))
            try:
                got = retriever(**store.rehydrate(s.inputs if isinstance(s.inputs, dict) else {}))
                got = list(got)
                m = overlap_metrics(s.outputs, got, k)
                case.jaccard, case.recall, case.top1_hit, case.n_cand = m["jaccard"], m["recall"], m["top1_hit"], len(got)
            except Exception as e:  # noqa: BLE001
                case.error = f"{type(e).__name__}: {e}"
            rep.cases.append(case)
    return rep
