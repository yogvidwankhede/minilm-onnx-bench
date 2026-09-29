"""Numerical parity between PyTorch and each ONNX variant.

Three levels, from strictest to most task-relevant:
  1. element-wise   max |a - b| over every embedding coordinate
  2. per-sentence   cosine(a_i, b_i); we report the worst sentence
  3. retrieval      does the ONNX model rank documents the same way?
                    top-1 agreement and mean top-k overlap vs PyTorch
"""

from __future__ import annotations

import numpy as np

from .runtimes import Batch

# fp32 exports should be numerically identical up to kernel reordering.
# int8 is lossy by design, so it gets task-level tolerances instead.
TOLERANCES = {
    "onnx_fp32": {"max_abs": 1e-4, "min_cos": 0.99999},
    "onnx_fp32_opt": {"max_abs": 1e-4, "min_cos": 0.99999},
    "onnx_int8": {"max_abs": None, "min_cos": 0.98, "min_top1": 0.90},
}


def elementwise(ref: np.ndarray, got: np.ndarray) -> dict:
    ref64, got64 = ref.astype(np.float64), got.astype(np.float64)
    cos = (ref64 * got64).sum(1) / (np.linalg.norm(ref64, axis=1) * np.linalg.norm(got64, axis=1))
    return {
        "max_abs_diff": float(np.abs(ref64 - got64).max()),
        "mean_abs_diff": float(np.abs(ref64 - got64).mean()),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
    }


def retrieval_agreement(ref_q, ref_d, got_q, got_d, k: int = 5) -> dict:
    """Compare document rankings for each query under both runtimes."""
    ref_scores, got_scores = ref_q @ ref_d.T, got_q @ got_d.T
    ref_rank = np.argsort(-ref_scores, axis=1)
    got_rank = np.argsort(-got_scores, axis=1)
    k = min(k, ref_d.shape[0])
    top1 = float((ref_rank[:, 0] == got_rank[:, 0]).mean())
    overlap = float(np.mean([len(set(r[:k]) & set(g[:k])) / k for r, g in zip(ref_rank, got_rank)]))
    return {
        "top1_agreement": top1,
        f"top{k}_overlap": overlap,
        "n_queries": int(ref_q.shape[0]),
        "n_docs": int(ref_d.shape[0]),
    }


def check(name: str, metrics: dict) -> dict:
    tol = TOLERANCES.get(name, {})
    fails = []
    if tol.get("max_abs") is not None and metrics["max_abs_diff"] > tol["max_abs"]:
        fails.append(f"max_abs_diff {metrics['max_abs_diff']:.2e} > {tol['max_abs']:.0e}")
    if metrics["min_cosine"] < tol.get("min_cos", -1):
        fails.append(f"min_cosine {metrics['min_cosine']:.6f} < {tol['min_cos']}")
    r = metrics.get("retrieval")
    if r and "min_top1" in tol and r["top1_agreement"] < tol["min_top1"]:
        fails.append(f"top1_agreement {r['top1_agreement']:.3f} < {tol['min_top1']}")
    return {"pass": not fails, "failures": fails, "tolerance": tol}


def concat(batches: list[Batch], fn) -> np.ndarray:
    return np.concatenate([fn(b) for b in batches], axis=0)
