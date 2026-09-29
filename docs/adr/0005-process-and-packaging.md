# ADR 0005: One process per container, bundle baked into the image

**Status:** accepted · 2026-09-29

## Decision

1. **One uvicorn worker per container; scale out with replicas.** ONNX Runtime already spreads each
   model call across cores. Multiple worker processes would each hold their own batching queue,
   splitting concurrent traffic across queues and shrinking batches, the opposite of what ADR 0004
   relies on. They would also multiply memory for identical model copies.
2. **The bundle is baked into the image.** One image tag is one exact (code, model) pair, so deploys
   are reproducible and rollback means running the previous tag. To swap models without rebuilding,
   mount a bundle over `/models/current`. Startup verification (ADR 0002) protects both paths.
3. **The runtime image is minimal:** a `python:slim` base, dependencies pinned in
   `requirements/serve.lock`, no PyTorch, non-root user (uid 10001), read-only model files. Bundle
   permissions are normalized in the build stage with `chmod`, not `COPY --chmod`, whose effect on
   directories differs across BuildKit versions. The Docker `HEALTHCHECK` probes `/readyz`, not `/livez`, so a container only reports healthy after
   checksum verification, warmup and the golden self-test.

## Consequences

- Size capacity per replica using `intra_op_threads` equal to its CPU allotment, then add replicas.
  The runbook covers sizing.
- The CI `image` job builds this Dockerfile on every push to `main` and checks that the image has no
  torch, runs as uid 10001, can read its bundle, becomes healthy, serves a request, and drains on
  `docker stop`.
