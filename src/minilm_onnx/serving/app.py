"""HTTP layer: OpenAI-compatible embeddings API, health, metrics.

    POST /v1/embeddings   {"input": str | [str], "encoding_format": "float" | "base64"}
    GET  /v1/models       loaded bundle identity
    GET  /livez           process is up (restart me if this fails)
    GET  /readyz          bundle verified, warmed up, golden self-test passed
    GET  /metrics         Prometheus exposition

The request/response shape matches OpenAI's embeddings API, so existing
clients (openai SDK, LangChain, LlamaIndex) can point base_url here.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import orjson
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .engine import Engine, NotReady, Overloaded
from .metrics import Metrics

log = logging.getLogger("embed.api")
access = logging.getLogger("embed.access")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class JSONResponse(Response):
    """orjson with native numpy support: vectors serialize without a Python-list detour."""

    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(default=None, description="must equal the model's dimension if given")
    user: str | None = Field(default=None, description="accepted for OpenAI compatibility; ignored, never logged")


def _error(status: int, message: str, etype: str, request: Request, headers: dict | None = None) -> JSONResponse:
    body = {
        "error": {
            "message": message,
            "type": etype,
            "code": status,
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    return JSONResponse(body, status_code=status, headers=headers)


class InputError(ValueError):
    pass


class BodyLimit:
    """Pure-ASGI guard: reject oversized or unsized request bodies before any parsing.

    Pydantic parses the whole JSON body before per-field limits can run, so
    the size cap has to sit in front of it. Chunked bodies (no Content-Length)
    get 411; clients that send JSON always know its length.
    """

    def __init__(self, app, max_bytes: int, paths: tuple[str, ...] = ("/v1/embeddings",)):
        self.app, self.max_bytes, self.paths = app, max_bytes, paths

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] in self.paths:
            length = dict(scope["headers"]).get(b"content-length")
            if length is None:
                return await self._reject(send, 411, "Content-Length is required")
            if not length.isdigit() or int(length) > self.max_bytes:
                return await self._reject(send, 413, f"request body exceeds {self.max_bytes} bytes")
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send, status: int, message: str) -> None:
        body = orjson.dumps({"error": {"message": message, "type": "invalid_request_error", "code": status}})
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


def validate_texts(raw: str | list[str], s: Settings) -> list[str]:
    texts = [raw] if isinstance(raw, str) else raw
    if not texts:
        raise InputError("input must contain at least one string")
    if len(texts) > s.max_texts_per_request:
        raise InputError(f"input has {len(texts)} strings; the limit is {s.max_texts_per_request} per request")
    for i, t in enumerate(texts):
        if not t or not t.strip():
            raise InputError(f"input[{i}] is empty")
        if len(t) > s.max_chars_per_text:
            raise InputError(f"input[{i}] has {len(t)} characters; the limit is {s.max_chars_per_text}")
        try:
            t.encode("utf-8")
        except UnicodeEncodeError:
            raise InputError(f"input[{i}] is not valid Unicode (unpaired surrogate)") from None
    return texts


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Any BundleError here aborts startup: better no pod than a wrong one.
        engine = Engine(settings, metrics)
        app.state.engine = engine
        await engine.start()
        yield
        log.info("draining")
        await engine.drain()
        log.info("stopped")

    app = FastAPI(
        title="MiniLM embedding service",
        version="1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.expose_docs else None,
        openapi_url="/openapi.json" if settings.expose_docs else None,
        redoc_url=None,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        # Accept a caller's id only if it's short and log-safe; otherwise mint one.
        supplied = request.headers.get("x-request-id", "")
        rid = supplied if REQUEST_ID_RE.match(supplied) else uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            dt = time.perf_counter() - t0
            if request.url.path == "/v1/embeddings":
                metrics.requests.labels(str(status)).inc()
                metrics.request_latency.observe(dt)
            if request.url.path not in ("/livez", "/readyz", "/metrics"):
                access.info(
                    "request",
                    extra={
                        "request_id": rid,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "duration_ms": round(dt * 1e3, 2),
                        "n_texts": getattr(request.state, "n_texts", None),
                        "tokens": getattr(request.state, "tokens", None),
                    },
                )
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, str(exc.detail), "invalid_request_error", request, getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(400, f"{loc}: {first.get('msg', 'invalid request')}", "invalid_request_error", request)

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest, request: Request):
        engine: Engine = request.app.state.engine
        try:
            texts = validate_texts(req.input, settings)
        except InputError as exc:
            return _error(400, str(exc), "invalid_request_error", request)
        dim = engine.manifest.embedding_dim
        if req.dimensions is not None and req.dimensions != dim:
            return _error(
                400,
                f"dimensions={req.dimensions} not supported; this model returns {dim}",
                "invalid_request_error",
                request,
            )
        request.state.n_texts = len(texts)
        try:
            vectors, tokens = await engine.embed(texts)
        except Overloaded:
            return _error(503, "server is at capacity, retry shortly", "overloaded", request, {"Retry-After": "1"})
        except NotReady as exc:
            return _error(503, f"not ready ({exc})", "unavailable", request, {"Retry-After": "5"})
        except asyncio.TimeoutError:
            return _error(504, f"embedding took longer than {settings.request_timeout_s}s", "timeout", request)
        except Exception:
            log.exception("embedding failed", extra={"request_id": request.state.request_id})
            return _error(500, "internal error", "server_error", request)
        request.state.tokens = tokens

        vectors = vectors.astype(np.float32, copy=False)
        if req.encoding_format == "base64":
            data = [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": base64.b64encode(v.astype("<f4").tobytes()).decode("ascii"),
                }
                for i, v in enumerate(vectors)
            ]
        else:
            data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]
        m = engine.manifest
        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "model": f"{m.name}@{m.version}",
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )

    @app.get("/v1/models")
    async def models(request: Request):
        m = request.app.state.engine.manifest
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": m.name,
                        "object": "model",
                        "owned_by": "self",
                        "version": m.version,
                        "variant": m.variant,
                        "weights": m.weights,
                        "embedding_dim": m.embedding_dim,
                        "max_seq_length": m.max_seq_length,
                        "source_model": m.source_model,
                        "created": m.created_utc,
                    }
                ],
            }
        )

    @app.get("/livez")
    async def livez(request: Request):
        engine: Engine | None = getattr(request.app.state, "engine", None)
        if engine is not None and not engine.healthy:
            return JSONResponse({"status": "unhealthy"}, status_code=503)  # batcher died: restart me
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request):
        engine: Engine | None = getattr(request.app.state, "engine", None)
        if engine is None or not engine.ready:
            state = "starting" if engine is None else engine.state()
            return JSONResponse({"status": state}, status_code=503)
        return {"status": "ready", "version": engine.manifest.version}

    @app.get("/metrics")
    async def prometheus():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.add_middleware(BodyLimit, max_bytes=settings.max_body_bytes)
    return app
