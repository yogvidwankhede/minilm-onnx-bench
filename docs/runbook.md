# Embedding service runbook

The service is `python -m minilm_onnx.serving`, shipped as the Docker image built from this repo.
Design rationale is in [`docs/adr/`](adr/).

## Endpoints

| Path | Purpose | Probe use |
|---|---|---|
| `POST /v1/embeddings` | OpenAI-compatible embeddings | — |
| `GET /v1/models` | loaded bundle: name, version, variant, dims | — |
| `GET /livez` | process is up and the batcher is alive (503 once the batcher has died) | liveness (restart on failure) |
| `GET /readyz` | bundle verified, warmed up, golden self-test passed, not draining | readiness (route traffic) |
| `GET /metrics` | Prometheus exposition | scrape |

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `EMBED_BUNDLE_DIR` | `/models/current` | bundle to serve |
| `EMBED_INTRA_OP_THREADS` | `0` (ORT default: all cores) | set to the container's CPU allotment |
| `EMBED_MAX_BATCH_SIZE` | `32` | texts per model call |
| `EMBED_MAX_BATCH_TOKENS` | `8192` | padded tokens per model call (batch × longest) |
| `EMBED_MAX_WAIT_MS` | `0` | extra wait for batch company; see ADR 0004 before raising |
| `EMBED_MAX_QUEUE_TEXTS` | `2048` | past this, 503 + `Retry-After: 1` |
| `EMBED_REQUEST_TIMEOUT_S` | `10` | per-request deadline (504 after) |
| `EMBED_MAX_TEXTS_PER_REQUEST` | `256` | 400 above this |
| `EMBED_MAX_CHARS_PER_TEXT` | `4096` | 400 above this; the model sees at most 256 tokens (~1–1.5k characters) |
| `EMBED_MAX_BODY_BYTES` | `8388608` | 413 above this, checked before JSON parsing; 411 if no `Content-Length` |
| `EMBED_CACHE_SIZE` | `10000` | in-process LRU of embeddings keyed by text hash; 0 disables |
| `EMBED_DRAIN_TIMEOUT_S` | `20` | on SIGTERM, time allowed to finish queued work |
| `EMBED_LOG_LEVEL` | `INFO` | JSON logs to stdout; request text is never logged |
| `EMBED_EXPOSE_DOCS` | `false` | serve `/docs` and `/openapi.json` |

A malformed or out-of-range value, such as `EMBED_MAX_BATCH_SIZE=lots` or `EMBED_REQUEST_TIMEOUT_S=0`,
fails startup with the variable's name in the error.

**Queue size vs timeout.** A text admitted at the back of a full queue waits about
`max_queue_texts / throughput` seconds. Keep that below `request_timeout_s`, or tail requests are
admitted only to time out. At ~220 texts/s per core, the default 2048-text queue is ~9.3 s against a
10 s timeout: fine on one core, generous on more. Size it as `timeout × measured throughput × 0.5`.

## Key metrics

| Metric | Read it as |
|---|---|
| `embed_request_latency_seconds` | what clients experience (histogram) |
| `embed_queue_wait_seconds` | time from enqueue to model call; this grows first when saturated |
| `embed_model_latency_seconds` | one ONNX Runtime call; rises with batch size and sequence length |
| `embed_batch_size`, `embed_batch_padded_tokens` | is batching happening? |
| `embed_padding_efficiency` | real tokens / padded tokens; low values mean mixed lengths are wasting compute |
| `embed_queue_depth` | texts waiting right now |
| `embed_requests_total{status}` | outcomes by HTTP status |
| `embed_cache_hits_total`, `embed_cache_misses_total` | cache effectiveness |
| `embed_ready`, `embed_model_info{version,...}` | readiness and which bundle is loaded |

### Suggested alerts (PromQL)

```promql
# Shedding load: sustained 503s
sum(rate(embed_requests_total{status="503"}[5m])) / sum(rate(embed_requests_total[5m])) > 0.01

# Latency SLO burn (example SLO: p99 < 250 ms)
histogram_quantile(0.99, sum by (le) (rate(embed_request_latency_seconds_bucket[5m]))) > 0.25

# Saturation leading indicator: queueing, not compute, dominates
histogram_quantile(0.9, sum by (le) (rate(embed_queue_wait_seconds_bucket[5m])))
  > 2 * histogram_quantile(0.9, sum by (le) (rate(embed_model_latency_seconds_bucket[5m])))

# Mixed model versions behind one endpoint (a rollout that never finished)
count(count by (version) (embed_model_info)) > 1

# Replica not ready
min(embed_ready) == 0
```

## Symptoms and actions

**Liveness failing (`/livez` 503, log line `batcher task died`).** A bug killed the batching task.
Queued requests were failed immediately; the orchestrator should restart the container. The
`CRITICAL` log line carries the traceback; file it.

**Pod never becomes ready.** Read the startup log. `checksum mismatch` or `bundle file missing` means
the bundle was corrupted or partially copied: redeploy the image or re-copy the bundle.
`golden self-test failed (pytorch | onnx | onnx batch-of-1)` means the graph, tokenizer, or ORT
version doesn't reproduce the reference vectors. Do not override it; roll back and investigate, because serving those vectors would silently
break search. `unsupported manifest schema` means the bundle was built by a newer packager.

**Rising 503s (`overloaded`).** The queue is full; replicas are saturated. Short term, add replicas.
Check `embed_queue_wait_seconds` against `embed_model_latency_seconds`: if queue wait dominates,
you're out of capacity; if model latency itself jumped, look for longer inputs
(`embed_batch_padded_tokens`) or CPU throttling. Raising `EMBED_MAX_QUEUE_TEXTS` only converts 503s
into slower responses.

**504s (timeouts).** The deadline passed while the request was queued or running. A model call already
running can't be interrupted, so each timeout still consumes its compute; persistent 504s mean
overload (see above) or pathological inputs.

**Latency up, no errors.** Check in order: queue wait (capacity), model latency (inputs got longer, or
CPU contention), padding efficiency (a traffic mix change such as long passages mixed with short
queries; consider separate deployments per traffic class).

**Vectors look different from last week.** Check `/v1/models` and `embed_model_info` for the version.
Under an int8 bundle, tiny differences (cosine ≥ 0.9999) between cached and fresh vectors are expected
(ADR 0003); anything larger means a model change.

## Deploy and roll back

1. Build a bundle: `python -m minilm_onnx.package --model <hf-id|dir> --variant int8`. It exits
   non-zero, writing nothing, if the parity gate fails.
2. Build the image with that bundle (`docker build --build-arg BUNDLE=bundles/<name> ...`), tagged with
   the bundle version.
3. Roll out gradually. Each replica verifies checksums and golden vectors before `/readyz` goes green,
   so a bad bundle stalls the rollout instead of taking traffic.
4. **Changing the model version changes the embedding space.** Vectors from different versions must
   not be compared. Re-embed the corpus into a new index, then switch reads, rather than mixing
   versions in one index.
5. Roll back by redeploying the previous image tag.

### Shutdown sequence

On SIGTERM:
1. Readiness flips to 503 immediately. New requests on already-open connections get
   503 + `Retry-After` and should be retried on another replica.
2. uvicorn stops accepting connections and waits up to `EMBED_DRAIN_TIMEOUT_S` for in-flight
   requests.
3. The engine drains anything still queued, again for up to `EMBED_DRAIN_TIMEOUT_S`, then exits.

A load balancer may keep routing to the pod for a few seconds after SIGTERM, until it notices. On
Kubernetes, add a `preStop` sleep (5–10 s) so endpoints are removed before the process stops
listening, and set `terminationGracePeriodSeconds` ≥ preStop + 2 × `EMBED_DRAIN_TIMEOUT_S`
(default: 10 + 40 = 50 s).

## Capacity planning

Measure on your own hardware with `tools/run_loadtest.sh <bundle>`. The load generator must run on
separate cores, or it steals CPU and skews results. Use its aiohttp client: httpx's pool becomes
the bottleneck at 64 connections (`results/serving/client_comparison.md`). On the reference stand-in
(int8, one pinned vCPU, single-text requests, median of 3 runs) one core sustained 222 req/s at
p99 123 ms with 16 concurrent clients, and 220 req/s at p99 482 ms with 64. For a latency SLO, size
replicas from the concurrency level where p99 still meets it, not from peak throughput.
