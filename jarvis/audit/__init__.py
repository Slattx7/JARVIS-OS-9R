"""jarvis_trace: trace schema, append-only audit store, deterministic replay + eval harness. Stdlib only."""
from .schema import (SCHEMA_VERSION, STRICTNESS, Decision, Divergence, Kind, Span, Tier, Trace, canonical, new_id)
from .redact import Redactor
from .store import Store, VerifyReport
from .runtime import (DEFAULT_LIVE, AuditUnavailable, Recorder, ReplayedError, ReplayMiss, ReplayPlan,
                      TraceRuntime, annotate, current, traced)
from .scorers import (ActionSequenceScorer, DivergenceScorer, GatingScorer, InvariantScorer, LatencyScorer,
                      NewRiskyProposalScorer, NoErrorScorer, PermissionLooseningScorer, RecallOverlapScorer,
                      ScoreContext, ScoreResult, Scorer, TextSimilarityScorer, TierIntegrityScorer, default_scorers)
from .harness import CaseResult, GateFailure, Report, run_case, run_suite
from .invariants import (ConfirmFreshnessInvariant, GatingInvariant, Invariant, OrphanInvariant, SnapshotInvariant,
                         TierConsistencyInvariant, Violation, check_trace, live_invariants)
from .policy import DEFAULT_POLICY, PolicyEngine, PolicyError, diff_policies
from .actions import ActionRegistry, ActionSpec, Executor, ExecResult, Param, UndoRunner
from .confirm import ApproveAll, CallbackConfirmer, CLIConfirmer, Confirmer, DenyAll, ScriptedConfirmer
from .risk import RiskAdapter, fail_closed
from .recall import recall_call, recall_eval
from .gate import Gate, GateError, load_gate
from . import audit, killswitch

__version__ = "0.2.0"
