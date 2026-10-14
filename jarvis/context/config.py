"""Server config: TOML + env overrides, plus generators for example configs. Stdlib-only."""
from __future__ import annotations

import copy
import os
import secrets
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 8787, "db_path": "data/recall.db", "timezone": "local",
               "max_body_bytes": 4_000_000, "max_batch": 500},
    "auth": {"ingest_token": "", "query_token": "", "admin_token": ""},
    "privacy": {"block": [], "meta": [], "allow": [], "redact": True},
    "apps": {"labels": {}},
    "retention": {"raw_ttl_days": 30, "meta_ttl_days": 365},
    "embed": {"backend": "hash", "dim": 384, "base_url": "http://127.0.0.1:8081/v1",
              "model": "bge-small-en-v1.5", "min_score": None, "timeout_s": 30},
    "llm": {"enabled": False, "base_url": "http://127.0.0.1:8080/v1", "model": "qwen2.5-3b-instruct",
            "mode": "json_schema", "min_interval_s": 2.0, "min_chars": 24,
            "categories": ["msg", "email"], "timeout_s": 60},
    "neo4j": {"enabled": False, "uri": "bolt://127.0.0.1:7687", "user": "neo4j", "password": ""},
}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | os.PathLike[str] | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    base_dir = Path.cwd()
    if path:
        p = Path(path).expanduser()
        with open(p, "rb") as f:
            deep_merge(cfg, tomllib.load(f))
        base_dir = p.resolve().parent
    env_map = {"RECALL_DB": ("server", "db_path"), "RECALL_INGEST_TOKEN": ("auth", "ingest_token"),
               "RECALL_QUERY_TOKEN": ("auth", "query_token"), "RECALL_ADMIN_TOKEN": ("auth", "admin_token")}
    for env, (sec, key) in env_map.items():
        if os.environ.get(env):
            cfg[sec][key] = os.environ[env]
    if overrides:
        deep_merge(cfg, overrides)
    db = Path(str(cfg["server"]["db_path"])).expanduser()
    cfg["server"]["db_path"] = str(db if db.is_absolute() else base_dir / db)
    return cfg


def new_token() -> str:
    return secrets.token_urlsafe(32)


def example_server_toml(ingest: str, query: str, admin: str) -> str:
    return f'''# recall-layer server config. Keep this file chmod 600.
[server]
host = "127.0.0.1"          # bind to your Tailscale IP (`tailscale ip -4`) to accept the phone; never 0.0.0.0
port = 8787
db_path = "data/recall.db"  # put on encrypted storage (LUKS) -- the DB holds notification text
timezone = "local"          # or an IANA name, e.g. "Europe/Berlin"

[auth]                      # least privilege: the phone only holds the write-only ingest token
ingest_token = "{ingest}"
query_token  = "{query}"    # for Jarvis / CLI reads
admin_token  = "{admin}"    # forget / merge / retention

[privacy]                   # merged on top of built-in defaults (password managers blocked, banking metadata-only)
block = []
meta  = []
allow = []
redact = true

[apps.labels]               # package -> display name
# "com.example.app" = "Example"

[retention]
raw_ttl_days  = 30          # notification text/payload purged after this; digests + graph persist
meta_ttl_days = 365         # events deleted entirely after this

[embed]
backend = "hash"            # "hash" (no model, lexical) | "openai" (any OpenAI-compatible /v1/embeddings)
base_url = "http://127.0.0.1:8081/v1"
model = "bge-small-en-v1.5"

[llm]                       # optional entity/relation extraction; every output is evidence-checked
enabled = false
base_url = "http://127.0.0.1:8080/v1"
model = "qwen2.5-3b-instruct"
mode = "json_schema"        # or "json_object"
min_interval_s = 2.0

[neo4j]                     # optional read-replica of the graph (no message text is exported)
enabled = false
uri = "bolt://127.0.0.1:7687"
user = "neo4j"
password = ""
'''


def example_device_toml(ingest: str, server_url: str = "http://100.64.0.1:8787") -> str:
    return f'''# recall-layer device config (rooted Android). Keep chmod 600.
device_id = "9r"
db_path = "/data/local/tmp/recall/outbox.db"

[server]
url = "{server_url}"        # Nitro's Tailscale address
token = "{ingest}"          # write-only ingest token
batch = 200
timeout_s = 15
gzip = true

[poll]
focus_s = 2.0               # foreground app / screen state
notif_s = 8.0               # notification dump (also triggered instantly by logcat events when available)
idle_mult = 3               # slow down focus polling while the screen is off
heartbeat_s = 300

[capture]
emit_initial = true         # emit notifications already in the shade on first start
skip_ongoing = true         # media players, downloads, foreground services
skip_group_summary = true
ignore_pkgs = ["com.android.systemui"]
logcat_triggers = true

[privacy]                   # merged on top of built-in defaults; enforced BEFORE anything is queued
block = []
meta = []
allow = []
redact = true

[limits]
max_unsent = 200000
keep_sent = 5000
'''
