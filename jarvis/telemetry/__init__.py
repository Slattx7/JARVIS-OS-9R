from .schema import EventType, Session, TraceEvent
from .storage import TraceStore
from .collector import Collector, CollectionError, DeviceNotFoundError
from .features import (
    FEATURE_COLUMNS,
    SessionFeatures,
    extract_all_sessions,
    extract_session_features,
    op_identifier,
)
from .labels import LabelError, build_training_set, load_labels
from .model import TrainingError, load_model, save_model, score_session, train_classifier
from .explain import FeatureContribution, explain_in_plain_english, explain_session
from .streaming import DriftAlert, StreamingScorer
from .permissions import PermissionUsage, assess_permissions, load_declared_permissions
from .trackers import (
    SharedDomain,
    SimilarPair,
    base_domain,
    find_known_trackers,
    hosts_by_package,
    match_known_tracker,
    shared_domains,
    similar_apps,
)

__all__ = [
    "EventType", "Session", "TraceEvent", "TraceStore",
    "Collector", "CollectionError", "DeviceNotFoundError",
    "FEATURE_COLUMNS", "SessionFeatures", "extract_all_sessions", "extract_session_features", "op_identifier",
    "LabelError", "build_training_set", "load_labels",
    "TrainingError", "load_model", "save_model", "score_session", "train_classifier",
    "FeatureContribution", "explain_in_plain_english", "explain_session",
    "DriftAlert", "StreamingScorer",
    "PermissionUsage", "assess_permissions", "load_declared_permissions",
    "SharedDomain", "SimilarPair", "base_domain", "find_known_trackers",
    "hosts_by_package", "match_known_tracker", "shared_domains", "similar_apps",
]
