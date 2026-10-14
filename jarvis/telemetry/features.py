"""Feature engineering: turn raw per-event traces into one row per session,
suitable for a tabular classifier.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from .schema import EventType, TraceEvent
from .storage import TraceStore

FEATURE_COLUMNS = [
    "duration_s", "event_count", "events_per_second",
    "app_op_count", "distinct_app_ops",
    "network_connect_count", "distinct_hosts",
    "sensor_register_count", "distinct_sensor_types",
    "location_request_count", "distinct_location_providers",
    "hook_error_count",
    "env_probe_count", "distinct_env_probes", "frida_port_probe_count",
]


class SessionFeatures(BaseModel):
    session_id: str
    package_name: str
    duration_s: float
    event_count: int
    events_per_second: float
    app_op_count: int
    distinct_app_ops: int
    network_connect_count: int
    distinct_hosts: int
    sensor_register_count: int
    distinct_sensor_types: int
    location_request_count: int
    distinct_location_providers: int
    hook_error_count: int
    env_probe_count: int
    distinct_env_probes: int
    frida_port_probe_count: int

    def to_row(self) -> dict:
        """Numeric feature dict for model training (drops identifier columns)."""
        return {col: getattr(self, col) for col in FEATURE_COLUMNS}


FRIDA_DEFAULT_PORT = 27042
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def op_identifier(event: TraceEvent) -> str | None:
    """The op an APP_OP event checked: args[0] of the noteOp/noteOpNoThrow
    call, or None if the event carries no usable identifier."""
    args = event.details.get("args")
    if not args or args[0] is None:
        return None
    return str(args[0])


def extract_session_features(events: list[TraceEvent]) -> SessionFeatures:
    """Compute one SessionFeatures row from all events of a single session.

    Assumes every event in `events` shares the same session_id - group
    first if loading from a log with multiple sessions in it (see
    `extract_all_sessions`).
    """
    if not events:
        raise ValueError("cannot extract features from an empty event list")

    session_id = events[0].session_id
    package_name = events[0].package_name
    timestamps = [e.timestamp for e in events]
    duration = (max(timestamps) - min(timestamps)).total_seconds()
    duration = max(duration, 1e-6)  # guard divide-by-zero for near-instantaneous sessions

    app_ops = [e for e in events if e.event_type == EventType.APP_OP]
    net = [e for e in events if e.event_type == EventType.NETWORK_CONNECT]
    sensors = [e for e in events if e.event_type == EventType.SENSOR_REGISTER]
    locations = [e for e in events if e.event_type == EventType.LOCATION_REQUEST]
    hook_errors = [e for e in events if e.event_type == EventType.HOOK_ERROR]
    env_probes = [e for e in events if e.event_type == EventType.ENV_PROBE]

    # args[0] is the op identifier passed to noteOp/noteOpNoThrow - an int
    # code on older Android internals, a String constant on newer public
    # overloads. Either way it identifies *which op*, unlike `method`
    # (which is only ever "noteOp"/"noteOpNoThrow" - a real bug fixed
    # here, since keying distinctness off `method` capped distinct_app_ops
    # at 2 regardless of how many different ops the app actually touched).

    distinct_app_ops = {op for op in map(op_identifier, app_ops) if op is not None}
    distinct_hosts = {e.details.get("host") for e in net if e.details.get("host")}
    distinct_sensor_types = {
        e.details.get("sensor_type") for e in sensors if e.details.get("sensor_type") is not None
    }
    distinct_providers = {e.details.get("provider") for e in locations if e.details.get("provider")}
    distinct_env_probes = {
        (e.details.get("kind"), e.details.get("target")) for e in env_probes
    }
    # A connection to loopback on Frida's default port is an app looking for
    # frida-server. Derived from NETWORK_CONNECT, so it needs no extra hook.
    frida_port_probes = [
        e for e in net
        if e.details.get("port") == FRIDA_DEFAULT_PORT and e.details.get("host") in LOOPBACK_HOSTS
    ]

    return SessionFeatures(
        session_id=session_id,
        package_name=package_name,
        duration_s=duration,
        event_count=len(events),
        events_per_second=len(events) / duration,
        app_op_count=len(app_ops),
        distinct_app_ops=len(distinct_app_ops),
        network_connect_count=len(net),
        distinct_hosts=len(distinct_hosts),
        sensor_register_count=len(sensors),
        distinct_sensor_types=len(distinct_sensor_types),
        location_request_count=len(locations),
        distinct_location_providers=len(distinct_providers),
        hook_error_count=len(hook_errors),
        env_probe_count=len(env_probes),
        distinct_env_probes=len(distinct_env_probes),
        frida_port_probe_count=len(frida_port_probes),
    )


def extract_all_sessions(paths: list[str | Path]) -> list[SessionFeatures]:
    """Load one or more TraceStore JSONL logs and compute a SessionFeatures
    row for every distinct session_id found across all of them."""
    grouped: dict[str, list[TraceEvent]] = defaultdict(list)
    for path in paths:
        for event in TraceStore(path).iter_events():
            grouped[event.session_id].append(event)

    return [extract_session_features(events) for events in grouped.values()]


def features_to_dataframe(features: list[SessionFeatures]) -> pd.DataFrame:
    """One row per session, including session_id/package_name for joining
    with labels - drop those two before fitting (see FEATURE_COLUMNS)."""
    return pd.DataFrame([f.model_dump() for f in features])
