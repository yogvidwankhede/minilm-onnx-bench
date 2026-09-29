# ADR 0006: Execution targets are measured per machine, not assumed

**Status:** accepted · 2026-09-29

## Context

"Serve with ONNX Runtime" is a CPU-shaped answer. Deployments also run on NVIDIA GPUs (CUDA), and
on desktops and laptops with Apple GPUs and Neural Engines. Which runtime and device is fastest
depends on the hardware, the batch size, and whether the input shapes are static, so the answer
has to be measured on each target rather than assumed.

## Decision

1. **The server takes an execution target:** `EMBED_EXECUTION_TARGET` = `cpu` (default), `coreml`,
   or `cuda` (`src/minilm_onnx/serving/providers.py`). Every target falls back to the CPU provider
   for unsupported operators. Asking for a target the installed onnxruntime build lacks fails
   startup loudly; it never silently falls back.
2. **Every target gets the fidelity checks, not just a speed number.** GPU and NPU paths often
   compute in reduced precision. The startup golden self-test runs on whichever target is
   configured.
3. **`tools/target_bench.py` benchmarks every target available on a machine** (PyTorch on CPU, MPS
   and CUDA; ONNX Runtime with CPU, CoreML and CUDA providers; int8 on CPU). It reports fidelity
   against PyTorch-CPU and a latency grid with bootstrap confidence intervals.

## Evidence: Apple M4 Max, fine-tuned weights

`results/healthmate/apple-m4-max/targets.md`. Every fp32 target (PyTorch-MPS, ONNX Runtime CPU and
CoreML) matched PyTorch-CPU to within 2e-07; neither GPU path was running in reduced precision.

| Workload | Fastest | vs PyTorch CPU |
|---|---|---:|
| Single query, 32 tokens | ONNX Runtime CPU, 1.24 ms | 2.00x |
| Single query, 128 tokens | PyTorch MPS 1.74x ≈ ONNX Runtime CPU 1.62x | — |
| Batch 8–32 | PyTorch on the Apple GPU (MPS) | 1.99x – 4.24x |
| Any shape | ONNX Runtime CoreML was slowest | 0.16x – 0.47x |

- On this machine **ONNX Runtime CPU is slower than PyTorch CPU on the larger shapes**: 0.72x–0.82x
  at batch 32 (32 or 128 tokens) and batch 8 × 128 tokens. It still wins at batch 8 × 32 tokens
  (1.41x). PyTorch uses Apple's Accelerate library for large matrix multiplies.
- **CoreML was slow here, most likely because of dynamic shapes** (not yet verified). The graph has
  variable batch and sequence axes, and CoreML compiles efficiently only for static shapes, so it
  likely partitions or falls back. Testing that means exporting fixed-shape buckets; not attempted.
- **CUDA is implemented but unmeasured.** No NVIDIA hardware was available. Its numbers must come
  from `target_bench.py` on a CUDA machine before anyone relies on it.

## Consequences

- On an Apple desktop, the right split is ONNX Runtime CPU for interactive single queries and
  PyTorch-MPS for bulk (re-)indexing. On x86 servers, ONNX Runtime CPU (ADR 0001).
- The perf gate (ADR 0007) tracks the CPU target on CI runners. Other targets are benchmarked on
  demand with `target_bench.py`, because CI has no GPUs.
