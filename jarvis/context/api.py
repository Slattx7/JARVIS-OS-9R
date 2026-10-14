"""HTTP API. Three token scopes, least privilege: `ingest` (the phone: write-only), `query` (agent: read-only tools),
`admin` (forget / merge / retention / stats; also satisfies the others)."""
from __future__ import annotations

import hmac
import json
import logging
import zlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import __version__, resolve, retention
from .runtime import Runtime, build_runtime
from .tools import SPECS, TOOLS, openai_tools

log = logging.getLogger("recall.api")
ALLOWED = {"ingest": {"ingest", "admin"}, "query": {"query", "admin"}, "admin": {"admin"}}


class Auth:
    def __init__(self, tokens: dict[str, str]):
        self.tokens = {k: v for k, v in tokens.items() if v}

    def granted(self, token: str) -> set[str]:
        got = set()
        for scope, tok in self.tokens.items():  # compare against every token: no early exit on the first hit
            if hmac.compare_digest(token.encode(), tok.encode()):
                got.add(scope)
        return got

    def require(self, scope: str):
        allowed = ALLOWED[scope]

        def dep(authorization: str = Header(default="")) -> None:
            token = authorization[7:].strip() if authorization[:7].lower() == "bearer " else ""
            granted = self.granted(token) if token else set()
            if not granted:
                raise HTTPException(401, "missing or invalid token", headers={"WWW-Authenticate": "Bearer"})
            if not granted & allowed:
                raise HTTPException(403, f"token lacks the '{scope}' scope")

        return dep


def _gunzip(data: bytes, limit: int) -> bytes:
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = d.decompress(data, limit + 1)
    except zlib.error:
        raise HTTPException(400, "bad gzip body") from None
    if len(out) > limit or d.unconsumed_tail:
        raise HTTPException(413, "decompressed body too large")
    return out


async def _json(request: Request, limit: int, *, empty_ok: bool = False) -> Any:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, "body too large")
    body = await request.body()
    if len(body) > limit:
        raise HTTPException(413, "body too large")
    if request.headers.get("content-encoding", "").lower() == "gzip":
        body = _gunzip(body, limit)
    if not body.strip():
        if empty_ok:
            return {}
        raise HTTPException(400, "empty body")
    try:
        return json.loads(body)
    except ValueError:
        raise HTTPException(400, "invalid JSON") from None


def create_app(cfg: dict[str, Any], *, rt: Runtime | None = None, start_worker: bool = True) -> FastAPI:
    a = cfg["auth"]
    auth = Auth({"ingest": a.get("ingest_token", ""), "query": a.get("query_token", ""), "admin": a.get("admin_token", "")})
    if not auth.tokens:
        raise RuntimeError("no auth tokens configured: run `recall init` or set [auth] / RECALL_*_TOKEN")
    rt = rt or build_runtime(cfg, start_worker=start_worker)
    max_body = int(cfg["server"]["max_body_bytes"])
    max_batch = int(cfg["server"]["max_batch"])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if rt.worker and not rt.worker.is_alive():
            rt.worker.start()
        yield
        rt.close()

    app = FastAPI(title="recall-layer", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.rt = rt

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.post("/v1/events", dependencies=[Depends(auth.require("ingest"))])
    async def post_events(request: Request) -> dict[str, Any]:
        payload = await _json(request, max_body)
        events = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(events, list):
            raise HTTPException(422, "body must be {\"device\": str, \"events\": [...]}")
        if len(events) > max_batch:
            raise HTTPException(413, f"too many events (max {max_batch})")
        res = await run_in_threadpool(rt.pipeline.ingest, events)
        if rt.worker:
            rt.worker.kick()
        return res

    @app.get("/v1/tools", dependencies=[Depends(auth.require("query"))])
    def list_tools() -> dict[str, Any]:
        return {"tools": TOOLS, "openai": openai_tools()}

    @app.post("/v1/tools/{name}", dependencies=[Depends(auth.require("query"))])
    async def call_tool(name: str, request: Request) -> JSONResponse:
        if name not in SPECS:
            return JSONResponse({"ok": False, "error": f"unknown tool {name!r}", "tools": sorted(SPECS)}, status_code=404)
        args = await _json(request, 64_000, empty_ok=True)
        if not isinstance(args, dict):
            raise HTTPException(422, "arguments must be a JSON object")
        res = await run_in_threadpool(rt.tools.call, name, args)
        return JSONResponse(res, status_code=200 if res.get("ok") else 400)

    admin = [Depends(auth.require("admin"))]

    @app.get("/v1/admin/stats", dependencies=admin)
    def stats() -> dict[str, Any]:
        return rt.store.stats()

    @app.post("/v1/admin/enrich", dependencies=admin)
    async def enrich(request: Request) -> dict[str, Any]:
        body = await _json(request, 4096, empty_ok=True)
        done, failed = await run_in_threadpool(rt.pipeline.enrich, int(body.get("limit", 64)))
        return {"completed": done, "failed": failed}

    @app.post("/v1/admin/retention", dependencies=admin)
    async def run_ret() -> dict[str, Any]:
        return await run_in_threadpool(retention.run_retention, rt.store, cfg, rt.clock, rt.labels)

    @app.post("/v1/admin/forget", dependencies=admin)
    async def forget(request: Request) -> dict[str, Any]:
        body = await _json(request, 1_000_000)
        allowed = {"app", "entity", "since", "until", "event_ids", "everything", "dry_run", "vacuum"}
        if not isinstance(body, dict) or set(body) - allowed:
            raise HTTPException(422, f"allowed fields: {sorted(allowed)}")
        try:
            return await run_in_threadpool(lambda: retention.forget(rt.store, labels=rt.labels, clock=rt.clock, **body))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/v1/admin/entities/candidates", dependencies=admin)
    def merge_candidates() -> dict[str, Any]:
        return {"candidates": resolve.candidates(rt.store)}

    @app.post("/v1/admin/entities/merge", dependencies=admin)
    async def merge(request: Request) -> dict[str, Any]:
        body = await _json(request, 4096)
        try:
            return await run_in_threadpool(resolve.merge_entities, rt.store, int(body["keep"]), int(body["drop"]))
        except (KeyError, TypeError):
            raise HTTPException(422, "body must be {\"keep\": id, \"drop\": id}") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/v1/admin/neo4j/sync", dependencies=admin)
    async def neo4j_sync() -> dict[str, Any]:
        from . import neo4j_sync as n4

        try:
            return await run_in_threadpool(n4.sync_from_config, rt.store, cfg)
        except n4.Neo4jUnavailable as e:
            raise HTTPException(409, str(e)) from None

    return app
