"""`recall` command line: setup, serving, ad-hoc queries, and the destructive admin operations."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import config as C
from . import resolve, retention
from .runtime import Runtime, build_runtime
from .tools import openai_tools


def _dump(o: Any) -> None:
    print(json.dumps(o, indent=2, ensure_ascii=False))


def _write_private(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to overwrite")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)


def cmd_init(a: argparse.Namespace) -> int:
    d = Path(a.dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    ing, qry, adm = C.new_token(), C.new_token(), C.new_token()
    _write_private(d / "server.toml", C.example_server_toml(ing, qry, adm), a.force)
    _write_private(d / "device.toml", C.example_device_toml(ing, a.server_url), a.force)
    print(f"wrote {d / 'server.toml'} and {d / 'device.toml'} (chmod 600)")
    print("next:\n"
          f"  1. set [server].host in server.toml to your Tailscale IP, and server_url in device.toml to http://<that ip>:{8787}\n"
          f"  2. recall -c {d / 'server.toml'} serve\n"
          f"  3. scripts/push_device.sh {d / 'device.toml'}     # installs + runs `doctor` on the phone\n"
          "  The phone only holds the write-only ingest token; keep query/admin tokens on the server.")
    return 0


def cmd_serve(a: argparse.Namespace, cfg: dict[str, Any]) -> int:
    import uvicorn

    from .api import create_app

    host, port = a.host or cfg["server"]["host"], a.port or int(cfg["server"]["port"])
    if host in ("0.0.0.0", "::"):
        print("warning: binding all interfaces; prefer your Tailscale IP", file=sys.stderr)
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    return 0


def _fmt_hits(hits: list[dict[str, Any]]) -> None:
    for h in hits:
        who = ",".join(e["name"] for e in h.get("entities", []) if e["type"] in ("person", "org", "thread"))[:28]
        print(f"{h['time'][:16].replace('T', ' ')}  {h['app_label'][:10]:<10}  {who:<28}  "
              f"{(h['text'] or '')[:90]!r}  #{h['event_id'][:10]}")


def cmd_tool(rt: Runtime, name: str, args: dict[str, Any], as_json: bool) -> int:
    res = rt.tools.call(name, {k: v for k, v in args.items() if v not in (None, [])})
    if not res["ok"]:
        print("error:", res["error"], file=sys.stderr)
        return 2
    r = res["result"]
    if as_json or name not in ("recall", "context_now"):
        _dump(r)
    elif name == "recall":
        _fmt_hits(r["hits"])
        if r.get("note"):
            print("note:", r["note"])
        if not r["hits"]:
            print("no matches")
    else:
        for d in r["devices"]:
            print(f"{d['device']}: screen={d['screen']} fg={d['foreground']} online={d['online']} (seen {d['last_seen_age_s']}s ago)")
        _fmt_hits(r["recent_notifications"])
    return 0


def cmd_forget(a: argparse.Namespace, rt: Runtime) -> int:
    kw = dict(labels=rt.labels, clock=rt.clock, app=a.app, entity=a.entity, since=a.since, until=a.until,
              event_ids=a.id or None, everything=a.everything)
    try:
        plan = retention.forget(rt.store, dry_run=True, **kw)
    except ValueError as e:
        print("error:", e, file=sys.stderr)
        return 2
    print(f"matches {plan['events_matched']} events" + (f", {plan['entities_targeted']} entity(ies) will be deleted" if plan["entities_targeted"] else ""))
    if a.dry_run:
        return 0
    if not a.yes:
        if not sys.stdin.isatty() or input("delete permanently, including derived data? [y/N] ").strip().lower() != "y":
            print("aborted (use --yes to skip the prompt)")
            return 1
    _dump(retention.forget(rt.store, vacuum=not a.no_vacuum, **kw))
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    from .bench import run_bench

    over: dict[str, Any] = {}
    if a.embed == "openai":
        over["embed"] = {"backend": "openai", "base_url": a.embed_url, "model": a.embed_model}
    _dump(run_bench(a.events, a.seed, a.days, over))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="recall", description="personal context graph: capture -> graph -> recall")
    ap.add_argument("-c", "--config", default=os.environ.get("RECALL_CONFIG"), help="server.toml (or $RECALL_CONFIG)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="write server.toml + device.toml with fresh tokens")
    p.add_argument("--dir", default=".")
    p.add_argument("--server-url", default="http://100.64.0.1:8787")
    p.add_argument("--force", action="store_true")
    sub.add_parser("token", help="print a fresh random token")
    p = sub.add_parser("serve")
    p.add_argument("--host")
    p.add_argument("--port", type=int)

    p = sub.add_parser("query", help="search history")
    p.add_argument("text", nargs="?", default="")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--app", action="append", dest="apps")
    p.add_argument("--entity")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("who-is")
    p.add_argument("name")
    p = sub.add_parser("timeline")
    p.add_argument("--since", default="today")
    p.add_argument("--until")
    p = sub.add_parser("changed", help="what_changed vs the previous window")
    p.add_argument("--since", default="24h")
    p.add_argument("--until")
    p.add_argument("--app")
    p = sub.add_parser("context")
    p.add_argument("--minutes", type=int, default=30)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("explain")
    p.add_argument("event_id")

    sub.add_parser("stats")
    p = sub.add_parser("enrich", help="run pending embedding/LLM enrichment now")
    p.add_argument("--limit", type=int, default=256)
    sub.add_parser("retention", help="digest + purge per the retention policy")
    p = sub.add_parser("forget", help="delete data and everything derived from it")
    p.add_argument("--app")
    p.add_argument("--entity")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--id", action="append", help="event id (repeatable)")
    p.add_argument("--everything", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-vacuum", action="store_true")
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("entities", help="entity maintenance")
    es = p.add_subparsers(dest="ecmd", required=True)
    es.add_parser("candidates")
    m = es.add_parser("merge")
    m.add_argument("keep", type=int)
    m.add_argument("drop", type=int)
    sub.add_parser("tools", help="print tool specs (OpenAI function format)")
    sub.add_parser("neo4j-sync")
    p = sub.add_parser("bench")
    p.add_argument("--events", type=int, default=2000)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--embed", choices=["hash", "openai"], default="hash")
    p.add_argument("--embed-url", default="http://127.0.0.1:8081/v1")
    p.add_argument("--embed-model", default="bge-small-en-v1.5")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "init":
        return cmd_init(a)
    if a.cmd == "token":
        print(C.new_token())
        return 0
    if a.cmd == "tools":
        _dump(openai_tools())
        return 0
    if a.cmd == "bench":
        return cmd_bench(a)
    cfg = C.load_config(a.config)
    if a.cmd == "serve":
        return cmd_serve(a, cfg)
    rt = build_runtime(cfg)
    try:
        if a.cmd == "query":
            return cmd_tool(rt, "recall", {"query": a.text, "since": a.since, "until": a.until, "apps": a.apps,
                                           "entity": a.entity, "limit": a.limit}, a.json)
        if a.cmd == "who-is":
            return cmd_tool(rt, "who_is", {"name": a.name}, True)
        if a.cmd == "timeline":
            return cmd_tool(rt, "timeline", {"since": a.since, "until": a.until}, True)
        if a.cmd == "changed":
            return cmd_tool(rt, "what_changed", {"since": a.since, "until": a.until, "app": a.app}, True)
        if a.cmd == "context":
            return cmd_tool(rt, "context_now", {"minutes": a.minutes}, a.json)
        if a.cmd == "explain":
            return cmd_tool(rt, "explain", {"event_id": a.event_id}, True)
        if a.cmd == "stats":
            _dump(rt.store.stats())
        elif a.cmd == "enrich":
            done, failed = rt.pipeline.enrich(a.limit)
            _dump({"completed": done, "failed": failed})
        elif a.cmd == "retention":
            _dump(retention.run_retention(rt.store, cfg, rt.clock, rt.labels))
        elif a.cmd == "forget":
            return cmd_forget(a, rt)
        elif a.cmd == "entities":
            if a.ecmd == "candidates":
                _dump(resolve.candidates(rt.store))
            else:
                _dump(resolve.merge_entities(rt.store, a.keep, a.drop))
        elif a.cmd == "neo4j-sync":
            from . import neo4j_sync

            try:
                _dump(neo4j_sync.sync_from_config(rt.store, cfg))
            except neo4j_sync.Neo4jUnavailable as e:
                print("error:", e, file=sys.stderr)
                return 2
        return 0
    except ValueError as e:
        print("error:", e, file=sys.stderr)
        return 2
    finally:
        rt.close()


if __name__ == "__main__":
    sys.exit(main())
