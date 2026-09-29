"""Search int8 quantization configs for one that keeps retrieval fidelity.

    python tools/quant_search.py --model <hf-id|dir> --out results/healthmate/quant_search.json

Plain dynamic int8 (one scale per tensor) can fail on trained transformers:
their activations develop large outliers that a single scale can't represent.
This tries the standard mitigations and measures, for each config, the same
fidelity numbers the release gate uses plus latency:

  per_tensor        baseline: ORT dynamic int8 on every MatMul, incl. attention scores
  weights_only      only MatMuls against constant weights (MatMulConstBOnly)
  +per_channel      one weight scale per output channel
  +skip_ffn_out     keep each layer's FFN down-projection in fp32 (the usual outlier site)
  +attention_only   quantize only Q/K/V/output projections
  +skip_last_2      keep the last two encoder layers in fp32

The config to ship is the fastest one that passes the gate; `package.py
--quant <name>` builds it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minilm_onnx import export, parity  # noqa: E402
from minilm_onnx.corpus import DOCS, QUERIES  # noqa: E402
from minilm_onnx.model import MAX_SEQ_LENGTH, build_standin_embedder, load_embedder  # noqa: E402
from minilm_onnx.package import real_tokenizer, standin_tokenizer  # noqa: E402
from minilm_onnx.quant import QUANT_CONFIGS, quantize  # noqa: E402
from minilm_onnx.runtimes import OrtRunner, TorchRunner, synthetic_batch  # noqa: E402
from minilm_onnx.serving.tokenize import TextEncoder, pad_batch  # noqa: E402


def latency_ms(runner, batch, iters: int = 30) -> float:
    for _ in range(5):
        runner(batch)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        runner(batch)
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1e3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id, local dir, or 'standin'")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    standin = args.model == "standin"
    model = build_standin_embedder() if standin else load_embedder(args.model)
    work = Path(tempfile.mkdtemp())
    tok = work / "tokenizer.json"
    (standin_tokenizer(tok, QUERIES + DOCS) if standin else real_tokenizer(args.model, tok))
    enc = TextEncoder(tok, MAX_SEQ_LENGTH)
    qb, db = pad_batch(enc.encode(QUERIES), enc.pad_id), pad_batch(enc.encode(DOCS), enc.pad_id)

    fp32 = export.export_fp32(model, work / "model.onnx")
    ref = TorchRunner(model, args.threads)
    ref_q, ref_d = ref(qb), ref(db)
    vocab = model.encoder.config.vocab_size
    speed_batches = {"b1_s32": synthetic_batch(1, 32, vocab), "b32_s128": synthetic_batch(32, 128, vocab)}
    base_rt = OrtRunner(fp32, "fp32", args.threads)
    base_ms = {k: latency_ms(base_rt, b) for k, b in speed_batches.items()}

    graph = onnx.load(str(fp32))
    rows = []
    for name in QUANT_CONFIGS:
        path = quantize(fp32, work / f"{name}.onnx", name, graph)
        rt = OrtRunner(path, name, args.threads)
        got_q, got_d = rt(qb), rt(db)
        m = parity.elementwise(np.vstack([ref_q, ref_d]), np.vstack([got_q, got_d]))
        m["retrieval"] = parity.retrieval_agreement(ref_q, ref_d, got_q, got_d)
        verdict = parity.check("onnx_int8", m)
        ms = {k: latency_ms(rt, b) for k, b in speed_batches.items()}
        row = {
            "config": name,
            "gate": "PASS" if verdict["pass"] else "FAIL",
            "min_cosine": m["min_cosine"],
            "top1_agreement": m["retrieval"]["top1_agreement"],
            "top5_overlap": m["retrieval"]["top5_overlap"],
            "size_mb": export.file_mb(path),
            "speedup_vs_onnx_fp32": {k: base_ms[k] / ms[k] for k in ms},
            "latency_ms": ms,
        }
        rows.append(row)
        print(
            f"{name:26} {row['gate']}  min cos {m['min_cosine']:.5f}  top1 {row['top1_agreement']:.2f}  "
            f"top5 {row['top5_overlap']:.2f}  {row['size_mb']:5.1f} MB  "
            + "  ".join(f"{k} {v:.2f}x" for k, v in row["speedup_vs_onnx_fp32"].items()),
            flush=True,
        )
    passing = [r for r in rows if r["gate"] == "PASS"]
    best = max(passing, key=lambda r: r["speedup_vs_onnx_fp32"]["b1_s32"], default=None)
    out = {
        "model": args.model,
        "threads": args.threads,
        "onnx_fp32_latency_ms": base_ms,
        "gate": parity.TOLERANCES["onnx_int8"],
        "configs": rows,
        "recommended": best["config"] if best else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"recommended: {out['recommended']}  -> {args.out}")


if __name__ == "__main__":
    main()
