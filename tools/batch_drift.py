"""Measure how much a text's embedding changes with its batch companions.

    python tools/batch_drift.py bundles/<fp32-bundle> bundles/<int8-bundle> > results/serving/batch_drift.json

Each golden text is embedded alone (batch of one), then all texts are sent
concurrently so they share mixed-length batches. fp32 should be invariant up
to float rounding; int8 dynamic quantization picks activation scales per
call, so its vectors move slightly (ADR 0003).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minilm_onnx.serving.bundle import load_golden, load_manifest  # noqa: E402
from minilm_onnx.serving.config import Settings  # noqa: E402
from minilm_onnx.serving.engine import Engine  # noqa: E402


async def measure(bundle: str) -> dict:
    texts = load_golden(Path(bundle))["texts"]
    eng = Engine(Settings.from_env(bundle_dir=bundle, cache_size=0, max_wait_ms=30, warmup_iters=1))
    await eng.start()
    try:
        solo = np.stack([(await eng.embed([t]))[0][0] for t in texts])
        together = np.stack([r[0][0] for r in await asyncio.gather(*(eng.embed([t]) for t in texts))])
    finally:
        await eng.drain()
    cos = (solo * together).sum(1)
    m = load_manifest(Path(bundle))
    return {
        "bundle": m.version,
        "variant": m.variant,
        "weights": m.weights,
        "n_texts": len(texts),
        "min_cosine_alone_vs_batched": float(cos.min()),
        "mean_cosine_alone_vs_batched": float(cos.mean()),
        "max_abs_diff": float(np.abs(solo - together).max()),
    }


def main() -> None:
    results = [asyncio.run(measure(b)) for b in sys.argv[1:]]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
