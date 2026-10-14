"""Continuous (streaming) risk scoring.

Feeds events in one at a time, keeps a bounded buffer of recent events for
one collection session, re-scores that buffer with the trained model, and
raises a drift alert when the score moves far from an established
baseline. This is what turns one-shot `score` into something that can
catch sleeper behavior: an app that looks clean for the first N minutes
and then starts something new.

Calibration caveat, on purpose up front: the model is trained on
whole-session features, but a rolling window only ever contains a slice of
a session (smaller counts, shorter duration). Absolute window scores are
therefore not calibrated to the training distribution. Treat the *change*
against the app's own baseline as the signal, not the raw number. Pass
`window_s=None` for a cumulative (session-so-far) view, which sits closer
to the training distribution but dilutes recent behavior.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from .features import extract_session_features
from .model import score_session
from .schema import TraceEvent


@dataclass
class DriftAlert:
    timestamp: datetime
    baseline_score: float
    current_score: float
    delta: float


class StreamingScorer:
    """Score a live event stream and flag drift from the app's own baseline.

    The baseline is the first score computed once `baseline_events` events
    have been seen. Alerts are edge-triggered: one alert when the score
    crosses `drift_threshold` away from baseline, then silence until it
    returns inside the threshold and crosses again - not one alert per
    event while it stays elevated.
    """

    def __init__(
        self,
        model,
        window_s: float | None = 60.0,
        min_events_to_score: int = 3,
        baseline_events: int = 10,
        drift_threshold: float = 0.3,
        score_every: int = 1,
        max_events: int = 5000,
    ) -> None:
        if window_s is not None and window_s <= 0:
            raise ValueError("window_s must be positive, or None for a cumulative window")
        if drift_threshold <= 0:
            raise ValueError("drift_threshold must be positive")
        if score_every < 1:
            raise ValueError("score_every must be >= 1")

        self._model = model
        self._window_s = window_s
        self._min_events = min_events_to_score
        self._baseline_events = baseline_events
        self._drift_threshold = drift_threshold
        self._score_every = score_every
        self._buffer: deque[TraceEvent] = deque(maxlen=max_events)
        self._baseline_score: float | None = None
        self._last_score: float | None = None
        self._alerting = False
        self._events_seen = 0
        self._lock = threading.Lock()

    @property
    def baseline_score(self) -> float | None:
        return self._baseline_score

    @property
    def last_score(self) -> float | None:
        return self._last_score

    def _evict_old(self, now: datetime) -> None:
        if self._window_s is None:
            return
        cutoff = now - timedelta(seconds=self._window_s)
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def ingest(self, event: TraceEvent) -> DriftAlert | None:
        """Feed one event. Returns a DriftAlert on the event that first
        pushes the score past the drift threshold, otherwise None."""
        with self._lock:
            self._buffer.append(event)
            self._events_seen += 1
            self._evict_old(event.timestamp)

            if len(self._buffer) < self._min_events:
                return None
            if self._events_seen % self._score_every != 0:
                return None

            features = extract_session_features(list(self._buffer))
            score = score_session(self._model, features.to_row())
            self._last_score = score

            if self._baseline_score is None:
                if self._events_seen >= self._baseline_events:
                    self._baseline_score = score
                return None

            delta = score - self._baseline_score
            if abs(delta) < self._drift_threshold:
                self._alerting = False
                return None
            if self._alerting:
                return None

            self._alerting = True
            return DriftAlert(
                timestamp=event.timestamp,
                baseline_score=self._baseline_score,
                current_score=score,
                delta=delta,
            )
