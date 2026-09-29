# ADR 0004: Dynamic batching with a token budget; default wait of 0 ms

**Status:** accepted · 2026-09-29

## Context

Search traffic is many small concurrent requests, usually one query each. A model call per request
leaves most of the CPU's vector units idle and pays per-call overhead each time. Inference servers
(Triton's dynamic batcher, Hugging Face text-embeddings-inference) merge concurrent requests into
shared model calls. The open questions were how to form the batches and how long to wait for them.

## Decision

1. **One batching queue, one ORT worker thread.** The asyncio loop admits, tokenizes and enqueues
   texts. A single batcher task gathers queued texts and runs ONNX Runtime in a dedicated thread,
   which releases the GIL and parallelizes internally with `intra_op_threads`. While the model runs,
   the loop keeps accepting requests.
2. **Token budget, not just batch size.** Cost scales with batch × longest sequence, because every row
   pads to the longest. Gathered texts are sorted by length and cut into sub-batches with
   `len(group) × longest ≤ max_batch_tokens` (default 8192) and `len(group) ≤ max_batch_size`
   (default 32). One 256-token passage doesn't make 31 short queries pay for 256 tokens each.
3. **All-or-nothing admission with a bounded queue.** A request enqueues all of its texts or none.
   Past `max_queue_texts` the server returns 503 with `Retry-After` instead of letting latency grow
   without bound.
4. **Default `max_wait_ms = 0`.** With a single worker, texts that arrive while a batch is running are
   all collected by the next gather, so batching happens on its own under load. Any positive wait
   adds that much latency to every request when the server is idle.

## Evidence

int8 stand-in bundle, server pinned to one vCPU, load generator on the other, single-text requests,
3 interleaved repetitions of each policy. Median, with the min–max range across repetitions where
it matters. Full table: `results/serving/loadtest.md`.

| Policy | c=1 p50 | c=16 req/s · p50 | c=64 req/s · p50 · p99 |
|---|---:|---:|---:|
| No batching | 7.6 ms | 151 · 104 ms | 144 · 442 ms · 523 ms |
| **Batching, wait 0 ms** | 7.7 ms | 222 · 69 ms | 220 · 279 ms · 482 ms |
| Batching, wait 2 ms | 9.5 ms | 224 · 68 ms | 208 · 298 ms · 377 ms |
| Batching, wait 5 ms | 13.0 ms | 210 · 71 ms | 196 · 321 ms · 437 ms |

What the data supports:

- **Batching vs none.** Throughput rises 47% at c=16 and 53% at c=64. The repetition ranges don't
  overlap (216–228 vs 135–154 req/s at c=16), and median latency drops by a third or more.
- **Waiting costs idle latency.** At c=1, wait 0 ms matches no batching (7.7 vs 7.6 ms), while
  2 ms and 5 ms of waiting raise p50 to 9.5 and 13.0 ms.
- **Wait 0 vs wait 2 under load is not resolved** by this data. wait 2 ms had lower median p99 at
  c=64 (377 vs 482 ms), but the ranges overlap (365–506 vs 377–484), and it looked slightly better at
  c=4. The default favors low-load latency; `max_wait_ms` remains a knob.

## Consequences

- Before changing `max_wait_ms`, re-run `tools/run_loadtest.sh` on the target hardware. On machines
  with more cores, where one model call is short relative to arrival gaps, a small wait may win.
- A request timeout frees the caller, but a model call already running in ORT can't be interrupted.
  The engine recovers when that call returns (`test_timeout_frees_client_and_engine_recovers`).
  Batch sizes are bounded, so a single call is bounded too.
- If the batcher task itself dies, queued requests fail immediately and `/livez` reports unhealthy,
  so the orchestrator restarts the process
  (`test_dead_batcher_marks_engine_unhealthy_and_fails_queued_work`).
