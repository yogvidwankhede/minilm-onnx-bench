"""Benchmark every execution target available on this machine.

    python tools/target_bench.py --model <hf-id|dir|standin> --out results/<machine>/targets

Targets (each skipped with a note if unavailable here):
  pytorch_fp32     PyTorch eager on CPU (the baseline and parity reference)
  pytorch_mps      PyTorch on the Apple GPU (Metal)
  pytorch_cuda     PyTorch on an NVIDIA GPU
  ort_cpu          ONNX Runtime, CPU provider
  ort_coreml       ONNX Runtime, CoreML provider (Apple GPU / Neural Engine)
  ort_cuda         ONNX Runtime, CUDA provider
  ort_cpu_int8     ONNX Runtime CPU, int8 (config from --int8-config)

GPU/NPU paths often compute in reduced precision, so each target gets the
same fidelity checks as the release gate, not just a speed number.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minilm_onnx import bench, export, parity  # noqa: E402
from minilm_onnx.corpus import DOCS, QUERIES  # noqa: E402
from minilm_onnx.model import MAX_SEQ_LENGTH, build_standin_embedder, load_embedder  # noqa: E402
from minilm_onnx.package import real_tokenizer, standin_tokenizer  # noqa: E402
from minilm_onnx.quant import QUANT_CONFIGS, quantize  # noqa: E402
from minilm_onnx.runtimes import OrtRunner, TorchRunner, torch_devices  # noqa: E402
from minilm_onnx.serving.providers import available_targets  # noqa: E402
from minilm_onnx.serving.tokenize import TextEncoder, pad_batch  # noqa: E402

LABELS = {
    "pytorch_fp32": "PyTorch, CPU",
    "pytorch_mps": "PyTorch, Apple GPU (MPS)",
    "pytorch_cuda": "PyTorch, NVIDIA GPU",
    "ort_cpu": "ONNX Runtime, CPU",
    "ort_coreml": "ONNX Runtime, CoreML",
    "ort_cuda": "ONNX Runtime, CUDA",
    "ort_cpu_int8": "ONNX Runtime, CPU, int8",
}


def machine() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
    }
    if sys.platform == "darwin":
        import subprocess

        info["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    else:
        info["cpu"] = platform.processor() or "unknown"
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--int8-config", default="weights_only+attention_only", choices=sorted(QUANT_CONFIGS))
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--seq-lens", default="32,128")
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--out", required=True, help="output path prefix; writes .json and .md")
    args = ap.parse_args()

    standin = args.model == "standin"
    model = build_standin_embedder() if standin else load_embedder(args.model)
    work = Path(tempfile.mkdtemp())
    tok = work / "tokenizer.json"
    (standin_tokenizer(tok, QUERIES + DOCS) if standin else real_tokenizer(args.model, tok))
    enc = TextEncoder(tok, MAX_SEQ_LENGTH)
    qb, db = pad_batch(enc.encode(QUERIES), enc.pad_id), pad_batch(enc.encode(DOCS), enc.pad_id)

    fp32 = export.export_fp32(model, work / "model.onnx")
    int8 = quantize(fp32, work / "model.int8.onnx", args.int8_config)

    runners, skipped = [TorchRunner(model, args.threads)], {}
    for dev in ("mps", "cuda"):
        if dev in torch_devices():
            # .to(device) moves a module in place: give each device its own copy
            runners.append(TorchRunner(copy.deepcopy(model), args.threads, device=dev))
        else:
            skipped[f"pytorch_{dev}"] = "device not available"
    targets = available_targets()
    for t in ("cpu", "coreml", "cuda"):
        if t in targets:
            runners.append(OrtRunner(fp32, f"ort_{t}", args.threads, target=t))
        else:
            skipped[f"ort_{t}"] = "provider not in this onnxruntime build"
    runners.append(OrtRunner(int8, "ort_cpu_int8", args.threads))

    print("[parity vs PyTorch CPU]")
    ref = runners[0]
    ref_q, ref_d = ref(qb), ref(db)
    fid = {}
    for r in runners[1:]:
        got_q, got_d = r(qb), r(db)
        m = parity.elementwise(np.vstack([ref_q, ref_d]), np.vstack([got_q, got_d]))
        m["retrieval"] = parity.retrieval_agreement(ref_q, ref_d, got_q, got_d)
        fid[r.name] = m
        print(
            f"  {r.name:14} max|d| {m['max_abs_diff']:.2e}  min cos {m['min_cosine']:.6f}  "
            f"top1 {m['retrieval']['top1_agreement']:.2f}",
            flush=True,
        )

    print("[latency]")
    cfg = bench.BenchConfig(
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        seq_lens=[int(x) for x in args.seq_lens.split(",")],
        min_seconds=args.min_seconds,
    )
    rows = bench.run(runners, cfg, vocab=model.encoder.config.vocab_size, baseline="pytorch_fp32")

    result = {
        "model": args.model,
        "weights": "random-init stand-in" if standin else "fine-tuned",
        "machine": machine(),
        "threads": args.threads,
        "int8_config": args.int8_config,
        "skipped": skipped,
        "fidelity_vs_pytorch_cpu": fid,
        "benchmark": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    out.with_suffix(".md").write_text(render(result))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


def render(res: dict) -> str:
    mc = res["machine"]
    lines = [
        f"# Execution targets: `{res['model']}` ({res['weights']} weights)",
        "",
        f"Machine: {mc['cpu']} ({mc['machine']}, {mc['cpu_count']} cores) · torch {mc['torch']} · "
        f"onnxruntime {mc['onnxruntime']} · {res['threads']} CPU threads · int8 config `{res['int8_config']}`",
        "",
        "## Fidelity vs PyTorch on CPU",
        "",
        "| Target | max abs diff | worst cosine | top-1 agreement | top-5 overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in res["fidelity_vs_pytorch_cpu"].items():
        r = m["retrieval"]
        lines.append(
            f"| {LABELS.get(name, name)} | {m['max_abs_diff']:.2e} | {m['min_cosine']:.6f} | "
            f"{r['top1_agreement']:.2f} | {r['top5_overlap']:.2f} |"
        )
    if res["skipped"]:
        lines += ["", "Not measured here: " + "; ".join(f"{LABELS.get(k, k)} ({v})" for k, v in res["skipped"].items())]
    lines += [
        "",
        "## Latency (model only)",
        "",
        "Speedup is median(PyTorch CPU) / median(target) with a 95% bootstrap CI.",
        "",
        "| batch | seq | Target | p50 ms | p95 ms | Speedup vs PyTorch CPU |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for r in res["benchmark"]:
        sp = (
            "1.00x (baseline)"
            if "speedup_vs_pytorch" not in r
            else f"{r['speedup_vs_pytorch']:.2f}x [{r['speedup_ci95'][0]:.2f}, {r['speedup_ci95'][1]:.2f}]"
        )
        lines.append(
            f"| {r['batch']} | {r['seq']} | {LABELS.get(r['runtime'], r['runtime'])} | {r['p50_ms']:.2f} | "
            f"{r['p95_ms']:.2f} | {sp} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
