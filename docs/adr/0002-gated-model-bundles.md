# ADR 0002: Model bundles with a release gate and a startup self-test

**Status:** accepted · 2026-09-29

## Context

An embedding service fails silently. A wrong tokenizer setting, a truncated download, or a graph
exported from the wrong checkpoint all produce valid-looking 384-d unit vectors. Retrieval quality
degrades and nothing errors. Worse, vectors already stored in a vector index were computed by the old
model, so an unnoticed change corrupts search for every stored document.

## Decision

The deployable unit is a **bundle**: `model.onnx`, `tokenizer.json`, `golden.json`, `manifest.json`.

1. **Release gate at build time.** `python -m minilm_onnx.package` exports the model and compares
   ONNX Runtime output against PyTorch on the same token ids, produced by the same tokenizer code the
   server runs. Thresholds: fp32 needs max |Δ| ≤ 1e-4 and cosine ≥ 0.99999; int8 needs cosine ≥ 0.98
   and top-1 retrieval agreement ≥ 0.90 on a 15-query × 40-document set. On failure nothing is
   written. The bundle is assembled in a staging directory and renamed into place, so a crashed build
   never leaves a partial bundle.
2. **Content-addressed identity.** The version is `<variant>-<12 hex>` of a SHA-256 over every
   bundle file's checksum, so changing the model, tokenizer, or golden set changes the version. The
   manifest records every file's SHA-256, the parity report that let the bundle ship, and build
   provenance (git commit, library versions). Bundle files are world-readable (755/644) so a
   non-root server can load them, whether baked into an image or mounted.
3. **Startup verification.** The server recomputes every checksum and refuses to start on a mismatch.
   It then re-embeds the golden texts through the full serving path (tokenizer, padding, ONNX Runtime)
   and runs three checks. Readiness stays false unless all pass:
   - vs the **PyTorch** reference, at the variant's parity tolerance (cosine ≥ 0.98 for int8,
     ≥ 0.99999 for fp32);
   - vs the bundle's **own build-time ONNX output**, at cosine ≥ 0.9999;
   - the same, for texts run as **batch-of-1**, the most common production shape.

   The second check exists because the first is too loose for int8. During code review, disabling
   the tokenizer's lowercasing on the int8 stand-in bundle moved its vectors to cosine 0.975 vs
   PyTorch, failing the 0.98 floor by only 0.005. Against the bundle's own output the margin is wide:
   a healthy bundle reproduces it at 0.9999999 (reported in the startup log).

## Consequences

- A corrupted file, a swapped graph, or tokenizer drift fails the deploy instead of degrading search.
  Each case has a test: `test_tampered_model_is_rejected`,
  `test_wrong_weights_fail_golden_self_test` (valid checksums, wrong weights),
  `test_tokenizer_drift_fails_tight_self_test` (lowercasing disabled, checksums fixed up), and
  `test_gate_blocks_packaging_when_tolerance_is_impossible`.
- The version string shows up in `/v1/models`, every response's `model` field, and the
  `embed_model_info` metric. A vector store can record which model version produced each vector,
  which is what a safe re-embedding migration needs.
- Changing any file changes the version (`test_bundle_is_readable_by_non_root_and_version_covers_every_file`).
