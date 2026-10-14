"""Unified JARVIS-9R Super-Project (JARVIS-OS)
Autonomous Edge Intelligence, Cryptographic Flight Recording & Zero-Trust Policy Engine
Designed for the OnePlus 9R + Acer Nitro 5 Workstation Architecture.
"""

__version__ = "1.0.0"

# Expose core namespaces
from . import perception
from . import policy
from . import context
from . import audit
from . import loop
from . import telemetry

__all__ = [
    "perception",
    "policy",
    "context",
    "audit",
    "loop",
    "telemetry",
    "__version__",
]
