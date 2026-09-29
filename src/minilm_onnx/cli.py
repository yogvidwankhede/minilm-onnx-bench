"""CLI: export -> parity -> benchmark -> report.

Examples
  # real fine-tuned model (needs Hugging Face access or a local copy)
  python -m minilm_onnx --model yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold

  # same-architecture random-init stand-in (offline; speed numbers only)
  python -m minilm_onnx --model standin
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import transformers

from . import bench, export, parity
from .model import DEFAULT_MODEL_ID, MAX_SEQ_LENGTH, build_standin_embedder, describe, load_embedder
from .report import write_markdown
from .runtimes import OrtRunner, TorchRunner, synthetic_batch


def environment(threads: int) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "threads": threads,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "transformers": transformers.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def parity_inputs(model_id: str, standin: bool, vocab: int):
    """Returns (query_batches, doc_batches, source_label)."""
    if not standin:
        from transformers import AutoTokenizer

        from .corpus import DOCS, QUERIES

        tok = AutoTokenizer.from_pretrained(model_id)

        def enc(texts):
            e = tok(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="np")
            if "token_type_ids" not in e:
                e["token_type_ids"] = np.zeros_like(e["input_ids"])
            return [{k: e[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")}]

        return enc(QUERIES), enc(DOCS), "built-in medical corpus (15 queries x 40 docs), real tokenizer"
    q = [synthetic_batch(8, 24, vocab, seed=s, pad_frac=0.5) for s in range(2)]
    d = [synthetic_batch(8, 64, vocab, seed=100 + s, pad_frac=0.6) for s in range(5)]
    return q, d, "synthetic token ids with variable padding (16 queries x 40 docs)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="minilm_onnx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF id, local dir, or 'standin'")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--artifacts", default="artifacts", help="where ONNX files are written")
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--seq-lens", default="32,128,256")
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke testing")
    args = ap.parse_args(argv)

    standin = args.model == "standin"
    out, art = Path(args.out), Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    print(f"[1/4] loading model: {args.model}")
    model = build_standin_embedder() if standin else load_embedder(args.model)
    info = describe(model)
    vocab = model.encoder.config.vocab_size

    print(f"[2/4] exporting ONNX -> {art}/")
    paths = export.export_all(model, art, int8=not args.no_int8)
    sizes = {k: export.file_mb(p) for k, p in paths.items()}
    sizes["pytorch_fp32"] = round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 2)

    runners = [TorchRunner(model, args.threads)]
    runners += [OrtRunner(paths["onnx_fp32"], "onnx_fp32", args.threads, optimize=False)]
    runners += [OrtRunner(paths["onnx_fp32_opt"], "onnx_fp32_opt", args.threads)]
    if "onnx_int8" in paths:
        runners += [OrtRunner(paths["onnx_int8"], "onnx_int8", args.threads)]

    print("[3/4] parity vs PyTorch")
    qb, db, source = parity_inputs(args.model, standin, vocab)
    ref_q, ref_d = parity.concat(qb, runners[0]), parity.concat(db, runners[0])
    par = {}
    for r in runners[1:]:
        got_q, got_d = parity.concat(qb, r), parity.concat(db, r)
        m = parity.elementwise(np.vstack([ref_q, ref_d]), np.vstack([got_q, got_d]))
        m["retrieval"] = parity.retrieval_agreement(ref_q, ref_d, got_q, got_d)
        if standin:
            m["retrieval"]["note"] = "random weights: ranking is not semantically meaningful; not gated"
        verdict = parity.check(r.name, {**m, "retrieval": None if standin else m["retrieval"]})
        par[r.name] = {**m, **verdict}
        flag = "PASS" if verdict["pass"] else "FAIL " + "; ".join(verdict["failures"])
        print(
            f"  {r.name:<14} max|d|={m['max_abs_diff']:.2e}  min cos={m['min_cosine']:.6f}  "
            f"top1={m['retrieval']['top1_agreement']:.2f}  {flag}"
        )

    print("[4/4] benchmarking")
    cfg = bench.BenchConfig(
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        seq_lens=[int(x) for x in args.seq_lens.split(",")],
        min_seconds=args.min_seconds,
    )
    if args.quick:
        cfg.batch_sizes, cfg.seq_lens, cfg.min_seconds, cfg.min_iters, cfg.bootstrap = [1, 4], [16, 64], 0.3, 6, 300
    rows = bench.run(runners, cfg, vocab=vocab)

    result = {
        "model": args.model,
        "weights": "random-init stand-in (same architecture)" if standin else "fine-tuned",
        "architecture": info,
        "environment": environment(args.threads),
        "bench_config": cfg.__dict__,
        "sizes_mb": sizes,
        "parity_source": source,
        "parity": par,
        "benchmark": rows,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    write_markdown(result, out / "results.md")
    print(f"\nwrote {out / 'results.json'} and {out / 'results.md'}")
    return 0 if all(v["pass"] for v in par.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
