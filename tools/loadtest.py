"""Closed-loop load test for the embedding service.

    python tools/loadtest.py --url http://localhost:8080 --concurrency 1,4,16,64 --duration 15

Each of C workers sends one request, waits for the answer, sends the next,
for `duration` seconds after a short warmup. Requests carry a unique suffix
so the server's embedding cache never hits: this measures model serving,
not cache lookups. Reports throughput and latency percentiles per level,
plus the server's own view (request time, queue wait, model time, batch
size) scraped from /metrics, so client-side artifacts are visible.

The client is aiohttp, not httpx: under 64 concurrent connections with
bursty responses (which is exactly what batching produces), httpx's async
pool became the bottleneck, adding hundreds of ms per request that the
server never saw. `--client httpx` reproduces it; results/serving/
client_comparison.jsonl has the numbers. See README, "Measuring it honestly".
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minilm_onnx.corpus import DOCS, QUERIES  # noqa: E402

POOL = QUERIES + DOCS  # short queries and longer passages, 3-25 words


class _AiohttpClient:
    def __init__(self, concurrency: int):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=concurrency), timeout=aiohttp.ClientTimeout(total=30)
        )

    async def post(self, url: str, payload: dict) -> int:
        try:
            async with self.session.post(url, json=payload) as r:
                await r.read()
                return r.status
        except aiohttp.ClientError:
            return -1  # connection-level failure

    async def close(self) -> None:
        await self.session.close()


class _HttpxClient:
    """Kept only to reproduce the client-side bottleneck described in the README."""

    def __init__(self, concurrency: int):
        import httpx

        self._httpx = httpx
        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        self.client = httpx.AsyncClient(timeout=30, limits=limits)

    async def post(self, url: str, payload: dict) -> int:
        try:
            return (await self.client.post(url, json=payload)).status_code
        except self._httpx.HTTPError:
            return -1

    async def close(self) -> None:
        await self.client.aclose()


CLIENTS = {"aiohttp": _AiohttpClient, "httpx": _HttpxClient}


async def worker(client, url, texts_per_req, stop_at, record_from, counter, lat, errors):
    while (now := time.perf_counter()) < stop_at:
        n = next(counter)
        texts = [f"{POOL[(n + k) % len(POOL)]} [{n}-{k}]" for k in range(texts_per_req)]
        t0 = time.perf_counter()
        status = await client.post(url, {"input": texts})
        dt = time.perf_counter() - t0
        if now >= record_from:
            if status == 200:
                lat.append(dt)
            else:
                errors.append(status)


async def run_level(base_url, concurrency, duration, warmup, texts_per_req, client_name="aiohttp") -> dict:
    lat: list[float] = []
    errors: list[int] = []
    counter = itertools.count(int(time.time() * 1000) % 10**9)
    client = CLIENTS[client_name](concurrency)
    try:
        start = time.perf_counter()
        record_from, stop_at = start + warmup, start + warmup + duration
        await asyncio.gather(
            *(
                worker(client, f"{base_url}/v1/embeddings", texts_per_req, stop_at, record_from, counter, lat, errors)
                for _ in range(concurrency)
            )
        )
    finally:
        await client.close()
    arr = np.asarray(lat) * 1e3
    return {
        "client": client_name,
        "concurrency": concurrency,
        "texts_per_request": texts_per_req,
        "requests": int(arr.size),
        "errors": len(errors),
        "rps": arr.size / duration,
        "texts_per_s": arr.size * texts_per_req / duration,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else None,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else None,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else None,
    }


async def scrape_batch_stats(base_url) -> dict:
    async with aiohttp.ClientSession() as c, c.get(f"{base_url}/metrics") as r:
        text = await r.text()
    vals = {}
    for line in text.splitlines():
        if line.startswith(
            (
                "embed_batch_size_sum",
                "embed_batch_size_count",
                "embed_queue_wait_seconds_sum",
                "embed_queue_wait_seconds_count",
                "embed_model_latency_seconds_sum",
                "embed_model_latency_seconds_count",
                "embed_request_latency_seconds_sum",
                "embed_request_latency_seconds_count",
            )
        ):
            k, v = line.split()
            vals[k] = float(v)
    return vals


async def main_async(args) -> list[dict]:
    rows = []
    for c in [int(x) for x in args.concurrency.split(",")]:
        before = await scrape_batch_stats(args.url)
        row = await run_level(args.url, c, args.duration, args.warmup, args.texts_per_request, args.client)
        after = await scrape_batch_stats(args.url)
        calls = after.get("embed_batch_size_count", 0) - before.get("embed_batch_size_count", 0)
        texts = after.get("embed_batch_size_sum", 0) - before.get("embed_batch_size_sum", 0)
        d = {k: after.get(k, 0) - before.get(k, 0) for k in after}  # includes the warmup window
        row["mean_texts_per_model_call"] = texts / calls if calls else None
        qn = d.get("embed_queue_wait_seconds_count", 0)
        row["server_queue_wait_ms"] = 1e3 * d.get("embed_queue_wait_seconds_sum", 0) / qn if qn else None
        rn = d.get("embed_request_latency_seconds_count", 0)
        row["server_request_ms"] = 1e3 * d.get("embed_request_latency_seconds_sum", 0) / rn if rn else None
        mn = d.get("embed_model_latency_seconds_count", 0)
        row["server_model_call_ms"] = 1e3 * d.get("embed_model_latency_seconds_sum", 0) / mn if mn else None
        row["label"] = args.label
        row["rep"] = args.rep
        rows.append(row)
        print(
            f"[{args.label}] c={c:<3} rps={row['rps']:7.1f}  p50={row['p50_ms']:7.1f}ms  "
            f"p99={row['p99_ms']:7.1f}ms  batch~{row['mean_texts_per_model_call'] or 0:4.1f}  "
            f"server~{row['server_request_ms'] or 0:6.1f}ms  "
            f"queue~{row['server_queue_wait_ms'] or 0:6.1f}ms  model~{row['server_model_call_ms'] or 0:5.1f}ms  "
            f"errors={row['errors']}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--warmup", type=float, default=3)
    ap.add_argument("--texts-per-request", type=int, default=1)
    ap.add_argument("--label", default="run")
    ap.add_argument("--client", choices=sorted(CLIENTS), default="aiohttp")
    ap.add_argument("--rep", type=int, default=1, help="repetition index, recorded in the output")
    ap.add_argument("--out", help="append results as JSON lines")
    args = ap.parse_args()
    rows = asyncio.run(main_async(args))
    if args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
