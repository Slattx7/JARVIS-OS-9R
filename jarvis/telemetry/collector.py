"""Frida-based orchestration: attach to (or spawn) a target app, load the
behavioral hooking agent, and stream captured events into a TraceStore.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Literal

import frida

from .schema import EventType, TraceEvent
from .storage import TraceStore

logger = logging.getLogger("warden.collector")

AGENT_SOURCE_PATH = Path(__file__).parent / "agent.js"


class DeviceNotFoundError(RuntimeError):
    """No USB-connected Android device is reachable."""


class CollectionError(RuntimeError):
    """Something went wrong attaching to or instrumenting the target app."""


class Collector:
    """Runs one behavioral-collection session against one target package."""

    def __init__(self, store: TraceStore) -> None:
        self._store = store

    def _get_device(self):
        try:
            return frida.get_usb_device(timeout=5)
        except frida.TimedOutError as exc:
            raise DeviceNotFoundError(
                "No USB device found within 5s. Confirm the 9R is connected, "
                "USB debugging is on, and frida-server is running as root on-device."
            ) from exc

    def run(
        self,
        package_name: str,
        duration_s: float | None = None,
        mode: Literal["spawn", "attach"] = "spawn",
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> str:
        """Collect trace events for `package_name`.

        Runs for `duration_s` seconds, or until interrupted if None.
        Returns the session_id used for this run.

        `on_event`, if given, is called with every parsed TraceEvent right
        after it's been written to storage (live scoring hooks in here). An
        exception raised by `on_event` is logged and swallowed - a broken
        downstream consumer must never interrupt collection.
        """
        session_id = str(uuid.uuid4())
        device = self._get_device()
        agent_source = AGENT_SOURCE_PATH.read_text(encoding="utf-8")

        def on_message(message: dict, data) -> None:
            msg_type = message.get("type")
            if msg_type == "error":
                logger.error("agent error: %s", message.get("stack", message))
                return
            if msg_type != "send":
                return

            payload = message.get("payload", {})
            event_type_raw = payload.get("event_type")
            try:
                event_type = EventType(event_type_raw)
            except ValueError:
                logger.warning("unrecognized event_type from agent: %r", event_type_raw)
                return

            event = TraceEvent(
                session_id=session_id,
                package_name=package_name,
                event_type=event_type,
                timestamp=payload.get("timestamp"),
                details=payload.get("details", {}),
            )
            self._store.append(event)

            if on_event is not None:
                try:
                    on_event(event)
                except Exception:  # noqa: BLE001 - deliberately broad, see docstring
                    logger.exception("on_event consumer raised; continuing collection")

        pid = None
        try:
            if mode == "spawn":
                pid = device.spawn(package_name)
                session = device.attach(pid)
            else:
                session = device.attach(package_name)
        except frida.ProcessNotFoundError as exc:
            raise CollectionError(
                f"Could not attach to '{package_name}' (mode={mode}): "
                f"not installed, or not running for attach mode."
            ) from exc

        script = session.create_script(agent_source)
        script.on("message", on_message)
        script.load()

        if mode == "spawn" and pid is not None:
            device.resume(pid)

        logger.info("collecting: package=%s session_id=%s mode=%s", package_name, session_id, mode)

        try:
            if duration_s is None:
                while True:
                    time.sleep(1)
            else:
                time.sleep(duration_s)
        except KeyboardInterrupt:
            logger.info("collection interrupted by user")
        finally:
            try:
                session.detach()
            except frida.InvalidOperationError:
                pass

        return session_id
