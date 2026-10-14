"""Wires config -> store, pipeline, tools. Shared by the HTTP API, the CLI and in-process agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import extract
from .embed import make_embedder
from .pipeline import EnrichWorker, Pipeline
from .privacy import Denylist
from .query import Clock
from .store import Store
from .tools import RecallTools


@dataclass
class Runtime:
    cfg: dict[str, Any]
    store: Store
    pipeline: Pipeline
    tools: RecallTools
    clock: Clock
    labels: dict[str, str]
    worker: EnrichWorker | None = field(default=None)

    def close(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.join(2)
        self.store.close()


def build_runtime(cfg: dict[str, Any], *, start_worker: bool = False, store: Store | None = None) -> Runtime:
    db = Path(cfg["server"]["db_path"])
    db.parent.mkdir(parents=True, exist_ok=True)
    st = store or Store(str(db))
    labels = extract.make_labels(cfg["apps"]["labels"])
    llm = cfg["llm"]
    extractor = None
    if llm.get("enabled"):
        extractor = extract.LLMExtractor(
            llm["base_url"], llm["model"], mode=llm.get("mode", "json_schema"), timeout=float(llm.get("timeout_s", 60)),
            min_chars=int(llm.get("min_chars", 24)), categories=tuple(llm.get("categories", ("msg", "email"))),
            min_interval_s=float(llm.get("min_interval_s", 2.0)), labels=labels)
    pipe = Pipeline(st, Denylist.from_config(cfg["privacy"]), labels, redact=bool(cfg["privacy"].get("redact", True)),
                    embedder=make_embedder(cfg), extractor=extractor)
    clock = Clock(cfg["server"].get("timezone", "local"))
    worker = EnrichWorker(pipe) if start_worker else None
    return Runtime(cfg, st, pipe, RecallTools(pipe, clock, labels, worker), clock, labels, worker)
