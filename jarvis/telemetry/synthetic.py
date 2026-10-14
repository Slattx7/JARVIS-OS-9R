"""Synthetic trace-log generation.

Exists to exercise the training/scoring pipeline end-to-end before you
have real 9R data. This is NOT a substitute for evaluation against real
collected traffic or a labeled malware corpus - the "malicious" sessions
here are a caricature (systematically more hosts/AppOps/location calls and more root/Frida probing),
good enough to prove the plumbing works, not to validate real detection
performance.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .schema import EventType, TraceEvent


def generate_synthetic_session(
    package_name: str,
    malicious: bool,
    rng: random.Random,
) -> list[TraceEvent]:
    """One synthetic session for `package_name`."""
    session_id = str(uuid4())
    t0 = datetime.now(timezone.utc)
    events: list[TraceEvent] = []

    def add(event_type: EventType, details: dict, offset_s: float) -> None:
        events.append(TraceEvent(
            session_id=session_id,
            package_name=package_name,
            event_type=event_type,
            timestamp=t0 + timedelta(seconds=offset_s),
            details=details,
        ))

    n_hosts = rng.randint(4, 10) if malicious else rng.randint(1, 3)
    n_app_ops = rng.randint(8, 20) if malicious else rng.randint(1, 6)
    n_locations = rng.randint(2, 5) if malicious else rng.randint(0, 1)

    t = 0.0
    for _ in range(n_hosts):
        t += rng.uniform(0.1, 2.0)
        add(EventType.NETWORK_CONNECT, {"host": f"host{rng.randint(1, 50)}.example.net", "port": 443}, t)
    for _ in range(n_app_ops):
        t += rng.uniform(0.1, 1.0)
        add(EventType.APP_OP, {"method": "noteOp", "args": [f"op{rng.randint(1, 10)}"]}, t)
    for _ in range(n_locations):
        t += rng.uniform(0.5, 3.0)
        add(EventType.LOCATION_REQUEST, {"provider": "gps", "arg_count": 3}, t)
    if rng.random() < 0.5:
        t += rng.uniform(0.1, 1.0)
        add(EventType.SENSOR_REGISTER, {"sensor_type": rng.randint(1, 5)}, t)

    # Environment probing: malicious sessions usually check whether they're
    # being watched (root/Frida/debugger); benign ones almost never do.
    probe_pool = [
        ("file_exists", "/system/xbin/su", "root_binary"),
        ("file_exists", "/data/local/tmp/frida-server", "frida"),
        ("exec", "which su", "root_binary"),
        ("debugger_check", "isDebuggerConnected", "debugger"),
    ]
    if malicious:
        n_probes = rng.randint(1, 4) if rng.random() < 0.7 else 0
    else:
        n_probes = 1 if rng.random() < 0.05 else 0
    for _ in range(n_probes):
        kind, target, category = rng.choice(probe_pool)
        t += rng.uniform(0.1, 1.0)
        add(EventType.ENV_PROBE, {"kind": kind, "target": target, "category": category}, t)
    if malicious and rng.random() < 0.3:
        t += rng.uniform(0.1, 1.0)
        add(EventType.NETWORK_CONNECT, {"host": "127.0.0.1", "port": 27042}, t)

    add(EventType.AGENT_READY, {}, 0.0)
    return sorted(events, key=lambda e: e.timestamp)


def generate_synthetic_dataset(
    n_benign: int = 20,
    n_malicious: int = 20,
    seed: int = 0,
) -> tuple[list[TraceEvent], dict[str, int]]:
    """Returns (all_events, labels): one synthetic package per session,
    split between benign and malicious by the given counts."""
    rng = random.Random(seed)
    all_events: list[TraceEvent] = []
    labels: dict[str, int] = {}

    for i in range(n_benign):
        pkg = f"com.example.benign{i}"
        all_events.extend(generate_synthetic_session(pkg, malicious=False, rng=rng))
        labels[pkg] = 0
    for i in range(n_malicious):
        pkg = f"com.example.malicious{i}"
        all_events.extend(generate_synthetic_session(pkg, malicious=True, rng=rng))
        labels[pkg] = 1

    return all_events, labels
