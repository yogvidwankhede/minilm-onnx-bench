# ADR 0007: Performance is gated over time, not measured once

**Status:** accepted · 2026-09-29

## Context

The release gate (ADR 0002) proves a bundle computes the *same* embeddings as PyTorch. It says
nothing about speed. A dependency bump, an exporter change, or a new graph option can quietly make
inference slower. A one-off benchmark in a README can't catch that; only measuring every change
and comparing it with history can.

## Decision

Every push to `main`, and every pull request, runs `tools/perf_gate.py` in the CI `perf` job on
the **real fine-tuned weights**.

1. **Measure hardware-normalized speed.** The main metric is ONNX Runtime's speedup over PyTorch,
   timed interleaved in the same process on the same runner, at three shapes (batch 1 × 32 tokens,
   8 × 128, 32 × 128). Shared CI runners vary run to run, and a ratio cancels most of that because
   both sides see the same machine. Absolute p50 latencies are recorded for trends but not gated.
2. **Fail on three conditions:**
   - **Fidelity:** below the release-gate tolerances.
   - **Sudden regression:** any shape's speedup more than 20% below the median of the last 5
     passing runs on the same hardware class.
   - **Slow drift:** any shape below a committed floor in `perf/floors.json`. A rolling median
     would slowly absorb a regression that arrives a few percent at a time; the floor doesn't
     move.
3. **Keep the history in git.** On `main`, each run (passing or failing, flagged) is appended to
   `history.jsonl` on the `perf-history` branch. A trend chart and table are regenerated there, so
   the full record is browsable and diffable. Only passing runs become baselines. Pull requests
   compare against the history but don't write to it.

## Consequences

- A regression fails the PR that introduces it, with a job summary table showing current vs
  baseline speedup per shape.
- The 20% tolerance reflects measured noise: consecutive runs in the build workspace varied about
  12% on the batch-1 ratio. Tighten it per hardware class as the history grows.
- Floors are deliberately loose until CI history exists (only batch-1 has one, 1.3x). A single
  cross-hardware floor for larger shapes would be wrong, because on Apple Silicon ONNX Runtime CPU
  is slower than PyTorch there (ADR 0006).
- The gate logic is unit-tested to fail on each condition (`tests/test_perf_gate.py`), not only to
  pass on good runs.
