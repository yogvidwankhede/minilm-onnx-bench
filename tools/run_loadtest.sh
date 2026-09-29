#!/usr/bin/env bash
# Batching on vs off, same bundle, same machine. Writes results/serving/loadtest.jsonl.
# Server and load generator are pinned to disjoint CPUs so they don't steal cycles from each other.
# Usage: tools/run_loadtest.sh <bundle_dir> [threads] [duration_s] [repeats]
#   Policies are interleaved within each repetition so machine drift hits all of them.
#   SERVER_CPUS / CLIENT_CPUS override the pinning (defaults: 0 / 1).
set -euo pipefail
BUNDLE=${1:?bundle dir}
THREADS=${2:-1}  # match the number of SERVER_CPUS
DURATION=${3:-15}
REPEATS=${4:-3}
SERVER_CPUS=${SERVER_CPUS:-0}
CLIENT_CPUS=${CLIENT_CPUS:-1}
OUT=results/serving/loadtest.jsonl
mkdir -p results/serving && : > "$OUT"

run() {  # label, port, rep, extra env...
  local label=$1 port=$2 rep=$3; shift 3
  env EMBED_BUNDLE_DIR="$BUNDLE" EMBED_INTRA_OP_THREADS="$THREADS" EMBED_CACHE_SIZE=0 EMBED_LOG_LEVEL=WARNING "$@" \
    taskset -c "$SERVER_CPUS" python -m minilm_onnx.serving --port "$port" >/dev/null 2>&1 &
  local pid=$!
  for _ in $(seq 1 60); do curl -sf "localhost:$port/readyz" >/dev/null && break; sleep 0.5; done
  taskset -c "$CLIENT_CPUS" python tools/loadtest.py --url "http://127.0.0.1:$port" --concurrency 1,4,16,64 \
    --duration "$DURATION" --label "$label" --rep "$rep" --out "$OUT"
  kill -TERM "$pid"; wait "$pid" 2>/dev/null || true
}

for rep in $(seq 1 "$REPEATS"); do
  run "no batching"        8201 "$rep" EMBED_MAX_BATCH_SIZE=1  EMBED_MAX_WAIT_MS=0
  run "batching, wait 0ms" 8202 "$rep" EMBED_MAX_BATCH_SIZE=32 EMBED_MAX_WAIT_MS=0
  run "batching, wait 2ms" 8203 "$rep" EMBED_MAX_BATCH_SIZE=32 EMBED_MAX_WAIT_MS=2
  run "batching, wait 5ms" 8204 "$rep" EMBED_MAX_BATCH_SIZE=32 EMBED_MAX_WAIT_MS=5
done
echo "wrote $OUT"
