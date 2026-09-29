# ADR 0003: fp32 by default; int8 only as a searched, gated opt-in

**Status:** accepted · 2026-09-29 (revised after the first real-weights run)

## Context

Dynamic int8 quantization stores MatMul weights as int8 and quantizes activations on the fly. It
needs no calibration data and is the usual first optimization for transformer encoders on CPU. The
first version of this project shipped int8 by default. All of its evidence came from a
random-weight stand-in, which cannot show the one thing that matters here: whether quantization
preserves *this model's* retrieval behavior.

## What the fine-tuned weights showed

Measured on the real model with `tools/quant_search.py` (Apple M4 Max, 4 CPU threads;
`results/healthmate/quant_search.json`). "Speed" is relative to fp32 ONNX Runtime on the same
machine.

| int8 config | Gate | Worst cosine | Top-1 | Size | Speed b1·s32 | Speed b32·s128 |
|---|---|---:|---:|---:|---:|---:|
| per-tensor (ORT default) | **FAIL** | 0.9594 | 0.80 | 58.6 MB | 0.92x | 0.95x |
| weights only | **FAIL** | 0.9605 | 0.80 | 58.6 MB | 0.90x | 0.93x |
| weights only + per-channel | **FAIL** | 0.9986 | 0.87 | 58.7 MB | 0.90x | 0.93x |
| weights only, FFN down-proj in fp32 | PASS | 0.9975 | 1.00 | 69.2 MB | 0.98x | 1.02x |
| weights only, attention projections only | PASS | 0.9988 | 1.00 | 79.8 MB | 0.98x | 0.95x |
| weights only, last 2 layers in fp32 | **FAIL** | 0.9642 | 0.80 | 69.2 MB | 0.95x | 0.85x |

Three findings:

1. **Plain int8 breaks this model.** The release gate refused it: worst cosine 0.96, and 3 of 15
   queries changed their top document. Trained transformers develop activation outliers that one
   int8 scale can't represent; the FFN down-projections are the usual site, and keeping them in fp32
   fixed fidelity here. The random-weight
   stand-in passed the same gate at cosine 0.99995, which is why stand-in results were never
   allowed to stand in for fidelity.
2. **Keeping outlier-heavy MatMuls in fp32 fixes fidelity.** The two passing configs keep either the
   FFN down-projections or all FFN layers in fp32.
3. **On Apple Silicon, no int8 config was faster than fp32.** ARM lacks the x86 VNNI int8
   dot-product instructions, and ORT's fp32 path is already fast there. On the x86 sandbox, whose
   CPU has VNNI, the fully quantized configs were 1.47–1.84x faster than fp32 ONNX, and the configs
   that keep layers in fp32 kept part of that gain (1.04–1.53x)
   (`results/standin/quant_search_x86.json`, stand-in weights). So the trade-off is
   hardware-specific.

The retrieval check uses a 15-query × 40-document set, so one flipped near-tie moves top-1 by
0.07. That is why per-channel (cosine 0.9986) fails while FFN-in-fp32 (cosine 0.9975) passes. A
larger labeled query set would sharpen the gate; it is the obvious next investment.

## Decision

- **fp32 is the default bundle variant.** It passes with max |Δ| 1.9e-07, and on the one
  real-hardware target measured, int8 buys nothing.
- **int8 is opt-in:** `--variant int8 --quant <config>`, defaulting to the attention-only config
  that passed. It must still pass the gate on the target weights, and its speed must be re-measured
  on the target CPU (`tools/target_bench.py`) before it ships.

## Consequences

- Bundles are about 90 MB instead of 59 MB.
- The batch-dependence of dynamic int8 (a text's vector shifts slightly with its batch companions;
  worst cosine 0.99998 on the stand-in, `results/serving/batch_drift.json`) only applies to int8
  bundles. fp32 is batch-invariant to 4.5e-08.
- Static (calibrated) quantization, or a larger evaluation set, are the paths to revisit int8.
