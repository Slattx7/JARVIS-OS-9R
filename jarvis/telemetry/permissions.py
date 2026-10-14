"""Permission usage analysis: what an app *declared* vs. what its behavior,
per the collected trace, actually exercised.

A gap in either direction is a signal independent of whether any single
behavior looks malicious:
  - declared but never exercised: broad permissions requested, narrow use
    observed (over-ask). Only reflects the collected session(s) - a short
    run can miss a permission the app legitimately uses later.
  - exercised but never declared: the app touched an op gated by a
    permission it did not request.

Matching is per-permission, against the public AppOpsManager OPSTR_*
string constants ("android:camera", ...), normalised for case and the
"android:" prefix. The hooked noteOp/noteOpNoThrow can also be called with
numeric op codes, and those cannot be safely mapped without confirming the
code table for your exact Android build - they are counted separately as
`numeric_op_events` rather than guessed at. If that count is high, the
`unexercised` list is overstated. Cross-check `args[0]` in a real trace
from the 9R and extend `extra_mapping` accordingly.

Declared permissions come from you (no APK/manifest inspection here), e.g.
`adb shell dumpsys package <pkg>` or `aapt dump permissions <apk>`; any
text containing `android.permission.*` names works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .features import op_identifier
from .schema import EventType, TraceEvent

# Dangerous permission -> the public AppOps op string(s) gating its use.
PERMISSION_TO_OPSTR: dict[str, set[str]] = {
    "android.permission.CAMERA": {"android:camera"},
    "android.permission.RECORD_AUDIO": {"android:record_audio"},
    "android.permission.ACCESS_FINE_LOCATION": {"android:fine_location"},
    "android.permission.ACCESS_COARSE_LOCATION": {"android:coarse_location"},
    "android.permission.READ_CONTACTS": {"android:read_contacts"},
    "android.permission.WRITE_CONTACTS": {"android:write_contacts"},
    "android.permission.READ_CALL_LOG": {"android:read_call_log"},
    "android.permission.READ_SMS": {"android:read_sms"},
    "android.permission.SEND_SMS": {"android:send_sms"},
    "android.permission.READ_PHONE_STATE": {"android:read_phone_state"},
    "android.permission.BODY_SENSORS": {"android:body_sensors"},
    "android.permission.ACTIVITY_RECOGNITION": {"android:activity_recognition"},
    "android.permission.READ_EXTERNAL_STORAGE": {"android:read_external_storage"},
    "android.permission.WRITE_EXTERNAL_STORAGE": {"android:write_external_storage"},
}

# Holding the key permission also legitimately covers access gated by these.
IMPLIES: dict[str, set[str]] = {
    "android.permission.ACCESS_FINE_LOCATION": {"android.permission.ACCESS_COARSE_LOCATION"},
    "android.permission.WRITE_EXTERNAL_STORAGE": {"android.permission.READ_EXTERNAL_STORAGE"},
}

_PERMISSION_RE = re.compile(r"android\.permission\.[A-Z0-9_]+")


@dataclass
class PermissionUsage:
    package_name: str
    declared_dangerous: list[str]
    exercised: list[str]
    unexercised: list[str]
    exercised_but_undeclared: list[str]
    total_op_events: int
    numeric_op_events: int  # identifiers we cannot decode without the build's op-code table
    unmapped_op_events: int  # string ops that map to no tracked dangerous permission

    @property
    def unexercised_ratio(self) -> float:
        if not self.declared_dangerous:
            return 0.0
        return len(self.unexercised) / len(self.declared_dangerous)


def normalize_op(op: str) -> str | None:
    """Canonical op string, or None for a numeric code (undecodable)."""
    op = op.strip().lower()
    if not op or op.isdigit():
        return None
    return op if op.startswith("android:") else f"android:{op}"


def load_declared_permissions(path: str | Path) -> list[str]:
    """Extract every android.permission.* name from a file, in order,
    deduplicated. Works on JSON lists, dumpsys output, aapt output, or
    one-per-line text alike."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    seen: dict[str, None] = {}
    for match in _PERMISSION_RE.findall(text):
        seen.setdefault(match, None)
    return list(seen)


def assess_permissions(
    package_name: str,
    declared_permissions: list[str],
    events: list[TraceEvent],
    extra_mapping: dict[str, set[str]] | None = None,
) -> PermissionUsage:
    mapping = {perm: {normalize_op(op) or op for op in ops} for perm, ops in PERMISSION_TO_OPSTR.items()}
    for perm, ops in (extra_mapping or {}).items():
        mapping.setdefault(perm, set()).update(normalize_op(op) or op for op in ops)

    op_to_perms: dict[str, set[str]] = {}
    for perm, ops in mapping.items():
        for op in ops:
            op_to_perms.setdefault(op, set()).add(perm)

    observed: set[str] = set()
    total = numeric = unmapped = 0
    for event in events:
        if event.event_type != EventType.APP_OP:
            continue
        total += 1
        raw = op_identifier(event)
        norm = normalize_op(raw) if raw is not None else None
        if norm is None:
            numeric += 1
            continue
        perms = op_to_perms.get(norm)
        if perms:
            observed |= perms
        else:
            unmapped += 1

    declared_dangerous = [p for p in declared_permissions if p in mapping]

    def covered(perm: str) -> bool:
        return perm in observed or any(implied in observed for implied in IMPLIES.get(perm, ()))

    exercised = [p for p in declared_dangerous if covered(p)]
    unexercised = [p for p in declared_dangerous if not covered(p)]

    effectively_declared = set(declared_dangerous)
    for perm in declared_dangerous:
        effectively_declared |= IMPLIES.get(perm, set())
    exercised_but_undeclared = sorted(observed - effectively_declared)

    return PermissionUsage(
        package_name=package_name,
        declared_dangerous=declared_dangerous,
        exercised=exercised,
        unexercised=unexercised,
        exercised_but_undeclared=exercised_but_undeclared,
        total_op_events=total,
        numeric_op_events=numeric,
        unmapped_op_events=unmapped,
    )
