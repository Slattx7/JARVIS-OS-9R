"""jarvis_loop: trace-driven, replay-gated self-improvement for a local agent. Core is stdlib-only."""
__version__ = "1.0.0"

from .traces import Trace, load_traces
from .curate import Curated, curate
from .gate import GateConfig, Outcome, Verdict, gate
from .registry import Registry
from .refs import ModelRef
from .config import LoopConfig
from .loop import CycleReport, Loop

__all__ = ["Trace", "load_traces", "Curated", "curate", "GateConfig", "Outcome", "Verdict", "gate",
           "Registry", "ModelRef", "LoopConfig", "Loop", "CycleReport", "__version__"]
