# ADR 0003: int8 dynamic quantization, with its batch-dependence stated

**Status:** accepted for the stand-in; **must be re-confirmed on the fine-tuned weights** · 2026-09-29

## Context

Dynamic int8 quantization (MatMul/Gemm weights stored as int8, activation scales computed on the fly)
needs no calibration data and is the standard first optimization for transformer encoders on CPU.

## Measurements (same-architecture stand-in)

- **Speed:** 1.96x–4.73x PyTorch eager across the offline benchmark grid
  (`results/standin/results.md`). That machine's CPU supports AVX-512 VNNI, the instruction set
  that accelerates int8 dot products. **On CPUs without VNNI, expect smaller int8 gains**; re-run
  `make bench` there before choosing int8.
- **Size:** 58.5 MB vs 90.3 MB. Only 35% smaller, because the 30,522 × 384 token-embedding table
  (~47 MB) is a Gather, not a MatMul, so it stays fp32.
- **Fidelity:** worst per-sentence cosine to fp32 PyTorch of about 0.99995, recorded in each
  bundle's `manifest.json` under `parity`.
- **Batch dependence:** activation scales are computed per model call, so a text's int8 vector depends
  slightly on which other texts share its batch. Measured with `tools/batch_drift.py` on the 55 golden
  texts, each embedded alone and then inside mixed-length batches (`results/serving/batch_drift.json`):

  | Variant | worst cosine (alone vs batched) | max abs coordinate difference |
  |---|---:|---:|
  | fp32 | 0.9999994 | 4.5e-08 |
  | int8 | 0.9999825 | 1.0e-03 |

## Decision

int8 is the default packaging variant, conditional on passing the release gate, whose retrieval
check applies to real weights. fp32 remains one flag away (`--variant fp32`).

## Consequences

- Under int8, the same text can come back with slightly different vectors: from the cache vs
  computed, or at different load levels. At cosine ≥ 0.99998 this is far below retrieval noise, but it
  rules out bit-exact equality checks on vectors.
  `test_int8_is_only_approximately_batch_invariant` asserts both that the drift exists and that it
  stays above cosine 0.9999. `test_concurrent_requests_keep_order_and_fp32_is_batch_invariant` holds
  fp32 to 1e-5.
- If an application needs bit-reproducible vectors, serve fp32.
- Static quantization, or quantizing the embedding table, could shrink the model further. Either one
  must pass the same gate.
