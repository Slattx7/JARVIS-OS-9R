"""Structured schema for behavioral trace events emitted by the Frida agent.

Every event the agent.js hooks observe gets normalized into a TraceEvent
before it touches disk. This is the contract between the Frida side (JS,
loose and defensive) and everything downstream (feature engineering,
model training) that wants a stable, typed shape to consume.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    APP_OP = "APP_OP"
    NETWORK_CONNECT = "NETWORK_CONNECT"
    SENSOR_REGISTER = "SENSOR_REGISTER"
    LOCATION_REQUEST = "LOCATION_REQUEST"
    ENV_PROBE = "ENV_PROBE"
    AGENT_READY = "AGENT_READY"
    HOOK_ERROR = "HOOK_ERROR"


class TraceEvent(BaseModel):
    """One structured event captured by the Frida agent during a session."""

    session_id: str
    package_name: str
    event_type: EventType
    timestamp: datetime
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_agent_timestamp(cls, value: Any) -> Any:
        # agent.js emits `new Date().toISOString()`, which ends in "Z".
        # Python's datetime.fromisoformat wants an explicit offset instead.
        if isinstance(value, str) and value.endswith("Z"):
            return value[:-1] + "+00:00"
        return value


class Session(BaseModel):
    """Metadata for one collection run against one target app."""

    session_id: str
    package_name: str
    device_serial: str
    mode: str  # "spawn" or "attach"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
