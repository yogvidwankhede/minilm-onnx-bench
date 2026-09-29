"""Inference engine: tokenize -> dynamic micro-batching -> ONNX Runtime.

Why dynamic batching
    A single short request leaves most of the CPU's matrix units idle. The
    batcher holds the first queued text for at most `max_wait_ms` so texts
    from concurrent requests share one model call. Under low load the wait
    is the only cost; under high load it multiplies throughput.

Why a token budget, not just a batch size
    Cost scales with batch * longest sequence (everything pads to the
    longest). The batcher sorts a gathered batch by length and cuts it into
    sub-batches whose padded size stays under `max_batch_tokens`, so one
    256-token text doesn't make 31 short ones pay for 256 tokens each.

Concurrency model
    Admission, batching and bookkeeping run on the asyncio event loop (one
    thread, no locks needed). ORT runs in a single dedicated worker thread;
    it releases the GIL and uses `intra_op_threads` internally, so the loop
    keeps accepting requests while the model runs. Requests with more than
    TOKENIZE_OFFLOOP_CHARS characters tokenize on a small bounded pool so
    one large request can't stall everyone else's admission.

Failure model
    A failed model call fails only the requests in that call. If the
    batcher task itself ever dies, the engine marks itself unhealthy
    (readiness and liveness both fail, so the orchestrator restarts it) and
    fails every queued request immediately instead of letting them time out.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .bundle import BundleError, Manifest, load_golden, load_manifest
from .config import Settings
from .metrics import Metrics
from .providers import providers_for
from .tokenize import TextEncoder, pad_batch

log = logging.getLogger("embed.engine")


class Overloaded(RuntimeError):
    """Queue is full; caller should retry later (HTTP 503 + Retry-After)."""


class NotReady(RuntimeError):
    """Engine is starting up, draining, or unhealthy."""


class BatchFailed(RuntimeError):
    """The model call (or batch preparation) for this text failed."""


TOKENIZE_OFFLOOP_CHARS = 4096  # above this many characters per request, tokenize off the event loop


@dataclass
class _Item:
    ids: np.ndarray
    future: asyncio.Future
    enqueued: float = field(default_factory=time.perf_counter)


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: OrderedDict[bytes, np.ndarray] = OrderedDict()

    @staticmethod
    def key(text: str) -> bytes:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()

    def get(self, k: bytes):
        v = self._d.get(k)
        if v is not None:
            self._d.move_to_end(k)
        return v

    def put(self, k: bytes, v: np.ndarray) -> None:
        if self.capacity <= 0:
            return
        self._d[k] = v
        self._d.move_to_end(k)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)


def plan_sub_batches(lengths: list[int], max_batch_size: int, max_batch_tokens: int) -> list[list[int]]:
    """Group item indices (sorted by length) so that each group's padded size
    (len(group) * longest) stays within the token budget and batch size."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    groups: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        longest = lengths[i]  # ascending order: the newcomer is the longest
        if cur and (len(cur) + 1 > max_batch_size or (len(cur) + 1) * longest > max_batch_tokens):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


class Engine:
    def __init__(self, settings: Settings, metrics: Metrics | None = None):
        self.s = settings
        self.m = metrics or Metrics()
        bundle = Path(settings.bundle_dir)
        self.manifest: Manifest = load_manifest(bundle, verify=True)  # raises BundleError
        self.encoder = TextEncoder(bundle / "tokenizer.json", self.manifest.max_seq_length)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        if settings.intra_op_threads > 0:
            so.intra_op_num_threads = settings.intra_op_threads
        self.session = ort.InferenceSession(
            str(bundle / "model.onnx"), so, providers=providers_for(settings.execution_target)
        )
        self._inputs = {i.name for i in self.session.get_inputs()}
        dim = self.session.get_outputs()[0].shape[-1]
        if isinstance(dim, int) and dim != self.manifest.embedding_dim:
            raise BundleError(f"model outputs dim {dim}, manifest says {self.manifest.embedding_dim}")

        self.cache = LRUCache(settings.cache_size)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ort")
        self._tok_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tokenize")
        self._queue: asyncio.Queue[_Item] | None = None
        self._worker: asyncio.Task | None = None
        self.ready = False
        self.draining = False
        self.healthy = True  # False once the batcher dies; liveness reports it
        self._inflight = 0
        self.m.model_info.labels(
            self.manifest.name, self.manifest.version, self.manifest.variant, self.manifest.weights
        ).set(1)

    # ---------- lifecycle ----------

    async def start(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.s.max_queue_texts)
        self._worker = asyncio.create_task(self._batch_loop(), name="batcher")
        self._worker.add_done_callback(self._on_worker_exit)
        try:
            await self._warmup()
            self._self_test()
        except BaseException:
            await self._stop_worker()
            raise
        self.ready = True
        self.m.ready.set(1)
        log.info("engine ready", extra={"version": self.manifest.version, "variant": self.manifest.variant})

    async def drain(self) -> None:
        """Stop accepting work, let queued texts finish, then stop the worker."""
        self.draining = True
        self.ready = False
        self.m.ready.set(0)
        if self._queue is not None:
            deadline = time.monotonic() + self.s.drain_timeout_s
            # Shutdown-only, deadline-bounded poll; an Event would add bookkeeping to the hot path.
            while (not self._queue.empty() or self._inflight) and time.monotonic() < deadline:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        await self._stop_worker()

    async def _stop_worker(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        self._executor.shutdown(wait=True)
        self._tok_executor.shutdown(wait=True)

    def _on_worker_exit(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return  # normal shutdown
        exc = task.exception()
        log.critical("batcher task died; engine is unhealthy", exc_info=exc)
        self.healthy = False
        self.ready = False
        self.m.ready.set(0)
        while self._queue is not None and not self._queue.empty():
            it = self._queue.get_nowait()
            if not it.future.done():
                it.future.set_exception(BatchFailed("embedding worker stopped"))

    async def _warmup(self) -> None:
        """First ORT calls allocate arenas and pick kernels; don't let users pay for that."""
        loop = asyncio.get_running_loop()
        for _ in range(self.s.warmup_iters):
            for bs, seq in ((1, 16), (8, 64), (4, self.manifest.max_seq_length)):
                ids = [np.full(seq, 5, dtype=np.int64) for _ in range(bs)]
                await loop.run_in_executor(self._executor, self._run, pad_batch(ids, self.encoder.pad_id))

    def _self_test(self) -> None:
        """Re-embed the bundle's golden texts through the full serving path.

        Two checks: against the PyTorch reference at the variant's parity
        tolerance, and against the bundle's own build-time ONNX output at a
        tight tolerance (catches tokenizer / runtime drift that the looser
        int8 parity tolerance would miss). A few texts also run as batch-of-1,
        the most common production shape.
        """
        golden = load_golden(Path(self.s.bundle_dir))
        texts = golden["texts"]
        got = self._run(pad_batch(self.encoder.encode(texts), self.encoder.pad_id))
        singles = np.vstack([self._run(pad_batch(self.encoder.encode([t]), self.encoder.pad_id)) for t in texts[:3]])
        checks = (
            ("pytorch", got, golden["torch"], self.manifest.golden_min_cosine),
            ("onnx", got, golden["onnx"], self.manifest.golden_onnx_min_cosine),
            ("onnx batch-of-1", singles, golden["onnx"][:3], self.manifest.golden_onnx_min_cosine),
        )
        report = {}
        for name, vecs, ref, floor in checks:
            exp = np.asarray(ref, dtype=np.float32)
            cos = (vecs * exp).sum(1) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(exp, axis=1))
            worst = float(cos.min())
            report[name] = round(worst, 7)
            if worst < floor:
                raise BundleError(f"golden self-test failed ({name}): min cosine {worst:.7f} < {floor}")
        log.info("golden self-test passed", extra={"min_cosine": report, "n": len(texts)})

    # ---------- request path ----------

    async def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """Returns (embeddings[len(texts), dim], prompt_tokens)."""
        if not self.ready or self._queue is None:
            raise NotReady(self.state())
        out: list[np.ndarray | None] = [None] * len(texts)
        keys = [LRUCache.key(t) for t in texts]
        miss_idx = []
        for i, k in enumerate(keys):
            hit = self.cache.get(k)
            if hit is None:
                miss_idx.append(i)
            else:
                out[i] = hit
        self.m.cache_hits.inc(len(texts) - len(miss_idx))
        self.m.cache_misses.inc(len(miss_idx))

        tokens = 0
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            # Big requests tokenize off the loop (Rust releases the GIL) so they
            # don't stall other requests' admission.
            if sum(len(t) for t in miss_texts) > TOKENIZE_OFFLOOP_CHARS:
                loop = asyncio.get_running_loop()
                seqs = await loop.run_in_executor(self._tok_executor, self.encoder.encode, miss_texts)
            else:
                seqs = self.encoder.encode(miss_texts)
            if not self.ready:
                raise NotReady(self.state())
            # Admission + enqueue run without an await in between, so they are
            # atomic on the event loop: a request is never half-enqueued.
            if self._queue.qsize() + len(seqs) > self._queue.maxsize:
                raise Overloaded(f"queue full ({self._queue.qsize()}/{self._queue.maxsize})")
            loop = asyncio.get_running_loop()
            futures = []
            for seq in seqs:
                fut = loop.create_future()
                self._queue.put_nowait(_Item(seq, fut))
                futures.append(fut)
            self.m.queue_depth.set(self._queue.qsize())
            try:
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=self.s.request_timeout_s)
            finally:
                # On timeout, error or external cancellation, release this request's
                # queued work: the batcher skips cancelled items. No-op for done futures.
                for f in futures:
                    f.cancel()
            for i, seq, vec in zip(miss_idx, seqs, results):
                out[i] = vec
                self.cache.put(keys[i], vec)
                tokens += len(seq)
        self.m.texts.inc(len(texts))
        return np.stack(out), tokens

    def state(self) -> str:
        if not self.healthy:
            return "unhealthy"
        return "draining" if self.draining else "starting"

    # ---------- batcher ----------

    async def _gather(self) -> list[_Item]:
        assert self._queue is not None
        first = await self._queue.get()
        self._inflight = 1  # items held here are invisible to queue.empty(); drain must wait for them
        items = [first]
        deadline = first.enqueued + self.s.max_wait_ms / 1e3
        # Collect up to two model calls' worth; the planner splits by token budget.
        cap = self.s.max_batch_size * 2
        while len(items) < cap:
            try:
                items.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            # Poll rather than wait_for(queue.get()): cancelling a pending
            # Queue.get on timeout can lose an item on some Python versions.
            await asyncio.sleep(min(remaining, 0.0005))
        self._inflight = len(items)
        return [it for it in items if not it.future.cancelled()]

    async def _batch_loop(self) -> None:
        while True:
            items = await self._gather()
            self.m.queue_depth.set(self._queue.qsize())
            try:
                groups = plan_sub_batches([len(it.ids) for it in items], self.s.max_batch_size, self.s.max_batch_tokens)
                for g in groups:
                    await self._run_group([items[i] for i in g])
            except asyncio.CancelledError:
                raise
            except Exception:
                # Anything unexpected outside the model call: fail these items, keep serving.
                log.exception("batch processing failed")
                for it in items:
                    if not it.future.done():
                        it.future.set_exception(BatchFailed("batch processing failed"))
            finally:
                self._inflight = 0

    async def _run_group(self, group: list[_Item]) -> None:
        batch = pad_batch([it.ids for it in group], self.encoder.pad_id)
        now = time.perf_counter()
        for it in group:
            self.m.queue_wait.observe(now - it.enqueued)
        padded = batch["input_ids"].size
        self.m.batch_size.observe(len(group))
        self.m.batch_tokens.observe(padded)
        self.m.padding_efficiency.observe(int(batch["attention_mask"].sum()) / padded)
        t0 = time.perf_counter()
        try:
            emb = await asyncio.get_running_loop().run_in_executor(self._executor, self._run, batch)
            if len(emb) != len(group):
                raise RuntimeError(f"model returned {len(emb)} rows for {len(group)} inputs")
        except Exception:  # fail only this group's requests; keep serving
            log.exception("model call failed")
            for it in group:
                if not it.future.done():
                    it.future.set_exception(BatchFailed("model call failed"))  # fresh instance per request
            return
        self.m.model_latency.observe(time.perf_counter() - t0)
        for it, vec in zip(group, emb):
            if not it.future.done():
                it.future.set_result(vec.copy())  # own the row; don't pin the whole batch array

    def _run(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        return self.session.run(None, {k: v for k, v in batch.items() if k in self._inputs})[0]
