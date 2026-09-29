# ADR 0001: ONNX Runtime on CPU, pooling inside the graph, un-fused graph in the bundle

**Status:** accepted · 2026-09-29

## Context

The embedder is a 22.6M-parameter MiniLM (6 layers, hidden 384). It serves retrieval queries: short
inputs, latency-sensitive, bursty. It is small enough that CPU inference is cheap and GPUs would sit
mostly idle, so the choice is between PyTorch eager and a compiled graph runtime on CPU.

## Decision

1. **Serve with ONNX Runtime on CPU.** Offline benchmarks on the same architecture measured fp32 ONNX at
   1.11x–2.68x and int8 ONNX at 1.96x–4.73x the speed of PyTorch eager across batch 1–32 and
   sequence 32–256 (`results/standin/results.md`). The serving path needs no PyTorch. During
   development, `pip install ".[serve]"` into a clean virtualenv came to about 245 MB of
   dependencies, and the service ran there with torch absent; CI checks the image has no torch.
2. **Mean pooling and L2 normalization are part of the exported graph.** The model's
   `modules.json` is Transformer → mean pool → normalize. Exporting only the encoder would make every
   consumer reimplement masked pooling, which is where silent divergence happens (forgetting the mask
   changes every vector). The graph returns final unit vectors.
3. **The bundle ships the un-fused graph; ORT fuses at session load.** ORT's fused graphs
   (`ORT_ENABLE_EXTENDED` output) contain contrib ops tied to the ORT version that produced them.
   Shipping the plain graph and letting the serving process optimize it keeps bundles portable across
   ORT upgrades. During development, a bundle built under ORT 1.25 passed its golden self-test when
   served by ORT 1.30 (the version pinned in `requirements/serve.lock`). The self-test (ADR 0002) is
   what makes a cross-version deploy safe to attempt.

## Consequences

- Build-time and serve-time dependencies are split (`[build]` vs `[serve]` extras).
- Session startup includes graph optimization (well under a second for this model).
- GPU serving, if ever needed, is a different execution provider plus a new benchmark, not a
  rewrite.
