"""Latency / throughput benchmark with warmup, fixed threads and bootstrap CIs.

Method
  * Every runtime gets the same pre-built input arrays (tokenization is
    identical for both and is excluded, so we measure the model only).
  * Each (runtime, batch, seq) cell: `warmup` untimed calls, then timed calls
    until both `min_iters` and `min_seconds` are met.
  * Runtimes are measured in interleaved rounds within each cell, so slow
    drift in machine state (thermal, noisy neighbours) hits all of them.
  * Speedup = median(PyTorch) / median(runtime), with a 95% percentile
    bootstrap CI over the per-call latencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .runtimes import synthetic_batch


@dataclass
class BenchConfig:
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32])
    seq_lens: list[int] = field(default_factory=lambda: [32, 128, 256])
    warmup: int = 5
    min_iters: int = 20
    min_seconds: float = 2.0
    rounds: int = 3
    bootstrap: int = 2000
    seed: int = 0


def _time_calls(fn, batch, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(batch)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _calls_for_budget(fn, batch, cfg: BenchConfig) -> int:
    t0 = time.perf_counter()
    fn(batch)
    one = max(time.perf_counter() - t0, 1e-5)
    per_round_s = cfg.min_seconds / cfg.rounds
    return max(int(np.ceil(cfg.min_iters / cfg.rounds)), int(per_round_s / one))


def bootstrap_speedup(base: np.ndarray, other: np.ndarray, n: int, rng) -> tuple[float, float]:
    b = rng.choice(base, size=(n, base.size), replace=True)
    o = rng.choice(other, size=(n, other.size), replace=True)
    ratios = np.median(b, axis=1) / np.median(o, axis=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(lat_ms: np.ndarray, batch: int) -> dict:
    p50 = float(np.percentile(lat_ms, 50))
    return {
        "n": int(lat_ms.size),
        "p50_ms": p50,
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
        "mean_ms": float(lat_ms.mean()),
        "std_ms": float(lat_ms.std(ddof=1)) if lat_ms.size > 1 else 0.0,
        "throughput_sent_per_s": batch / (p50 / 1e3),
    }


def run(runners: list, cfg: BenchConfig, vocab: int = 30522, baseline: str = "pytorch_fp32", log=print) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for seq in cfg.seq_lens:
        for bs in cfg.batch_sizes:
            batch = synthetic_batch(bs, seq, vocab=vocab, seed=cfg.seed)
            lat: dict[str, list[float]] = {r.name: [] for r in runners}
            n_calls = {}
            for r in runners:
                for _ in range(cfg.warmup):
                    r(batch)
                n_calls[r.name] = _calls_for_budget(r, batch, cfg)
            for _ in range(cfg.rounds):
                for r in runners:
                    lat[r.name] += _time_calls(r, batch, n_calls[r.name])
            base = np.asarray(lat[baseline])
            for r in runners:
                arr = np.asarray(lat[r.name])
                row = {"runtime": r.name, "batch": bs, "seq": seq, **summarize(arr, bs)}
                if r.name != baseline:
                    row["speedup_vs_pytorch"] = float(np.median(base) / np.median(arr))
                    lo, hi = bootstrap_speedup(base, arr, cfg.bootstrap, rng)
                    row["speedup_ci95"] = [lo, hi]
                rows.append(row)
            cell = "  ".join(f"{r.name}={np.median(lat[r.name]):.2f}ms" for r in runners)
            log(f"  batch={bs:<3} seq={seq:<4} {cell}")
    return rows
