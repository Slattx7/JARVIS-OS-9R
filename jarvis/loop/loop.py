from __future__ import annotations
import subprocess, time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import LoopConfig
from .curate import curate, write_dataset
from .gate import gate
from .monitor import MonitorResult, evaluate
from .refs import ModelRef
from .registry import Registry
from .replay import OutcomeCache, replay_all
from .traces import load_traces
from .util import file_lock, read_json, write_json


@dataclass
class CycleReport:
    status: str            # skipped | promoted | held | rejected | rolled_back | error
    version: str = ""
    reason: str = ""
    verdict: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Loop:
    """One cycle = curate -> train from base -> register -> replay baseline+candidate -> gate -> promote|archive.
    Invariants: nothing is deleted; only a gate 'promote' verdict can move CURRENT; PAUSE blocks promotion;
    a failing promote hook auto-rolls back; every cycle is rate-limited and ledgered."""

    def __init__(self, cfg: LoopConfig, trainer, replayer, clock=time.time):
        self.cfg, self.trainer, self.replayer, self.clock = cfg, trainer, replayer, clock
        self.root = Path(cfg.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.reg = Registry(cfg.registry_dir, clock=clock)
        self.runs = Path(cfg.runs_dir)
        self.runs.mkdir(parents=True, exist_ok=True)

    # ---- setup / introspection
    def init(self, version: str = "v0", artifact=None) -> dict:
        return self.reg.bootstrap(version, self.cfg.base_model if artifact is None else artifact,
                                  {"note": "bootstrap baseline"})

    def status(self) -> dict:
        h = self.reg.history()
        return {"current": self.reg.current(), "stack": self.reg.stack(), "paused": self.reg.paused(),
                "versions": self.reg.versions(), "state": read_json(self.cfg.state_path, {}),
                "recent": [e for e in h if e["event"] in ("promote", "rollback", "cycle", "monitor")][-8:]}

    # ---- helpers
    def _next_version(self) -> str:
        existing = set(self.reg.versions())
        n = len(existing)
        while f"v{n:04d}" in existing:
            n += 1
        return f"v{n:04d}"

    def _cycles_since(self, t: float) -> int:
        return sum(1 for e in self.reg.history() if e["event"] == "cycle" and e["ts"] >= t)

    def _golden_file(self) -> list:
        return read_json(self.cfg.golden_file, []) if self.cfg.golden_file else []

    def _run_hook(self, vid: str):
        if not self.cfg.promote_hook:
            return True, ""
        ref = self.reg.ref(vid)
        argv = [a.format(version=vid, artifact=ref.artifact) for a in self.cfg.promote_hook]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        return p.returncode == 0, (p.stderr or p.stdout or "")[-200:]

    def _finish(self, rep: CycleReport, now: float, run: Path = None, train_count: int = None,
                event: str = "cycle", report: str = "report.json") -> CycleReport:
        if rep.version:
            self.reg.log(event=event, version=rep.version, decision=rep.status, reason=rep.reason[:300], ts=now)
            if run is not None:
                write_json(run / report, rep.to_dict())
        if train_count is not None:
            st = read_json(self.cfg.state_path, {})
            st.update(last_train_count=train_count, last_cycle_ts=now, cycles=st.get("cycles", 0) + 1)
            write_json(self.cfg.state_path, st)
        return rep

    def _schema_problem(self, c, ls: dict):
        """Silent trace-schema drift is the way a loop like this poisons itself (e.g. `steps` renamed -> the model
        learns to answer [] to everything). Refuse to train on data that looks mis-mapped."""
        seen = ls.get("loaded", 0) + ls.get("bad_lines", 0)
        if seen and ls.get("bad_lines", 0) / seen > 0.2:
            return f"schema check: {ls['bad_lines']}/{seen} trace lines unparseable"
        loaded = max(1, ls.get("loaded", 0))
        if c.dropped.get("empty", 0) / loaded > 0.5:
            return f"schema check: {c.dropped['empty']}/{loaded} traces have no task text (fix ALIASES in traces.py)"
        if c.train:
            noop = sum(1 for t in c.train if not t.calls) / len(c.train)
            if noop > self.cfg.max_noop_frac:
                return (f"schema check: {noop:.0%} of successful traces have no tool calls - is `steps` mapped? "
                        f"(raise max_noop_frac if that is real)")
        return None

    # ---- the cycle
    def cycle(self, force: bool = False) -> CycleReport:
        why = self.reg.paused()
        if why is not None:
            return CycleReport("skipped", reason=f"paused: {why}")
        cur = self.reg.current()
        if cur is None:
            return CycleReport("error", reason="not initialised: run `init` first")
        with ExitStack() as st:
            try:
                st.enter_context(file_lock(self.root / "cycle.lock", timeout=0.0, stale=self.cfg.train_timeout_s + 3600))
            except TimeoutError:
                return CycleReport("skipped", reason="another cycle is running")
            return self._cycle(cur, force)

    def _cycle(self, cur: str, force: bool) -> CycleReport:
        cfg, now = self.cfg, self.clock()
        if not force and self._cycles_since(now - 86400) >= cfg.max_cycles_per_day:
            return CycleReport("skipped", reason=f"rate limit: {cfg.max_cycles_per_day} cycles / 24h")
        if not Path(cfg.traces).exists():
            return CycleReport("error", reason=f"traces not found: {cfg.traces}")
        ls: dict = {}
        c = curate(load_traces(cfg.traces, stats=ls), cfg.holdout_frac, max_per_task=cfg.max_per_task,
                   max_holdout=cfg.max_holdout)
        golden = set(self._golden_file()) | set(c.golden)
        n_new = len(c.train) - read_json(cfg.state_path, {}).get("last_train_count", 0)
        info = {"train": len(c.train), "holdout": len(c.holdout), "new": n_new, **ls}
        problem = self._schema_problem(c, ls)
        if problem:
            return CycleReport("error", reason=problem, data=info)
        if not force:
            if len(c.train) < cfg.min_train:
                return CycleReport("skipped", reason=f"{len(c.train)} train examples < min_train {cfg.min_train}", data=info)
            if n_new < cfg.min_new_examples:
                return CycleReport("skipped", reason=f"{n_new} new examples < min_new_examples {cfg.min_new_examples}", data=info)
        if not c.train or not c.holdout:
            return CycleReport("skipped", reason="empty train or holdout split", data=info)

        vid = self._next_version()
        run = self.runs / vid
        if run.exists():                           # leftover of a failed attempt for the same (unregistered) version id
            aside = self.runs / "_stale" / f"{vid}-{int(now)}"
            aside.parent.mkdir(exist_ok=True)
            run.rename(aside)
        run.mkdir(parents=True)
        info["dataset"] = write_dataset(c, run / "data")
        try:
            res = self.trainer.train(run / "data", run / "train")
        except Exception as e:                     # not counted as trained: retry next cycle (rate-limited)
            return self._finish(CycleReport("error", vid, f"training failed: {e}", data=info), now, run)
        self.reg.register(vid, res.artifact, {"parent": cur, "metrics": res.metrics, "dataset": info["dataset"],
                                              "run": str(run)})
        return self._gate_and_promote(vid, cur, c.holdout, golden, info, now, run, len(c.train))

    def _gate_and_promote(self, vid, cur, holdout, golden, info, now, run, train_count=None,
                          event="cycle", report="report.json") -> CycleReport:
        cfg = self.cfg
        cache = OutcomeCache(cfg.cache_path) if cfg.use_replay_cache else None
        fin = lambda rep, tc=None: self._finish(rep, now, run, tc, event, report)
        try:
            base = replay_all(self.replayer, self.reg.ref(cur), holdout, cache, cfg.replay_workers)
            cand = replay_all(self.replayer, self.reg.ref(vid), holdout, cache, cfg.replay_workers)
        except Exception as e:
            # candidate stays registered (never promoted); don't consume the data trigger so the next cycle retries
            return fin(CycleReport("error", vid, f"replay failed: {e}", data=info))
        slices = {t.id: (t.calls[0]["tool"] if t.calls else "(none)") for t in holdout}
        verdict = gate(base, cand, cfg.gate_config(golden), slices)
        status = {"promote": "promoted", "hold": "held", "reject": "rejected"}[verdict.decision]
        reason = "; ".join(verdict.reasons)
        if verdict.decision == "promote":
            try:
                self.reg.promote(vid, verdict)
            except Exception as e:
                status, reason = "error", f"promote refused: {e}"
            else:
                ok, msg = self._run_hook(vid)
                if not ok:
                    self.reg.rollback(reason=f"promote hook failed: {msg}")
                    status, reason = "rolled_back", f"promote hook failed: {msg}"
        rep = CycleReport(status, vid, reason, {"decision": verdict.decision, "reasons": verdict.reasons,
                                                 "stats": verdict.stats}, info)
        return fin(rep, train_count)

    def regate(self, vid: str) -> CycleReport:
        """Re-run the gate for an already-trained version against the CURRENT model, on that version's stored
        holdout (no retraining). For candidates held back by a pause, a replay-server outage, or a config change."""
        why = self.reg.paused()
        if why is not None:
            return CycleReport("skipped", vid, f"paused: {why}")
        cur = self.reg.current()
        if cur is None:
            return CycleReport("error", vid, "not initialised: run `init` first")
        if self.reg.meta(vid) is None:
            return CycleReport("error", vid, f"unknown version {vid}")
        if vid == cur:
            return CycleReport("skipped", vid, "already current")
        run = self.runs / vid
        hp = run / "data" / "holdout.jsonl"
        if not hp.exists():
            return CycleReport("error", vid, "no stored holdout for this version")
        with ExitStack() as st:
            try:
                st.enter_context(file_lock(self.root / "cycle.lock", timeout=0.0, stale=self.cfg.train_timeout_s + 3600))
            except TimeoutError:
                return CycleReport("skipped", vid, "another cycle is running")
            holdout = list(load_traces(hp))
            golden = set(read_json(run / "data" / "golden.json", [])) | set(self._golden_file())
            now = self.clock()
            return self._gate_and_promote(vid, cur, holdout, golden, {"regate": True, "holdout": len(holdout)}, now,
                                          run, None, "regate", f"report.regate-{int(now)}.json")

    # ---- post-deploy guard
    def monitor_once(self) -> MonitorResult:
        st = self.reg.stack()
        if len(st) < 2:
            return MonitorResult("insufficient", ["no previous version to compare against"])
        new, prev = st[-1], st[-2]
        promoted_ts = max((e["ts"] for e in self.reg.history() if e["event"] == "promote" and e["version"] == new),
                          default=0.0)
        traces = list(load_traces(self.cfg.traces)) if Path(self.cfg.traces).exists() else []
        live_new = [t for t in traces if t.model == new and (t.ts == 0 or t.ts >= promoted_ts)]
        live_prev = [t for t in traces if t.model == prev and (t.ts == 0 or t.ts < promoted_ts)]
        res = evaluate(live_new, live_prev, self.cfg.monitor_config())
        self.reg.log(event="monitor", version=new, action=res.action, reasons=res.reasons, stats=res.stats)
        if res.action == "rollback" and self.cfg.auto_rollback:
            self.reg.rollback(reason="monitor: " + "; ".join(res.reasons))
        return res
