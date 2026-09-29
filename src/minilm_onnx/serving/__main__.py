"""Run the service:  python -m minilm_onnx.serving [--host 0.0.0.0] [--port 8080]

One process per container on purpose: ONNX Runtime already parallelizes each
model call across cores (intra-op threads), and a single process keeps one
batching queue, so concurrent requests actually share model calls. Scale out
with more replicas, not more workers per replica.
"""

from __future__ import annotations

import argparse
import math

import uvicorn

from .app import create_app
from .config import Settings
from .logs import configure


class _Server(uvicorn.Server):
    """uvicorn closes the listener on SIGTERM and only runs lifespan shutdown
    after in-flight requests finish. Flip readiness first, so /readyz goes red
    and new requests on kept-alive connections get 503 + Retry-After while
    everything already admitted completes."""

    def __init__(self, config: uvicorn.Config, app) -> None:
        super().__init__(config)
        self._app = app

    def handle_exit(self, sig, frame) -> None:
        engine = getattr(self._app.state, "engine", None)
        if engine is not None:
            engine.draining = True
            engine.ready = False
            engine.m.ready.set(0)
        super().handle_exit(sig, frame)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--loop", choices=["auto", "asyncio", "uvloop"], default="auto")
    args = ap.parse_args(argv)
    settings = Settings.from_env()
    configure(settings.log_level)
    app = create_app(settings)
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        loop=args.loop,
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=max(1, math.ceil(settings.drain_timeout_s)),
        timeout_keep_alive=30,
    )
    _Server(config, app).run()


if __name__ == "__main__":
    main()
