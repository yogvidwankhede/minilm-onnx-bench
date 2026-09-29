"""Performance-over-time gate.

    python tools/perf_gate.py --model <hf-id|dir|standin> --history perf-history/history.jsonl \
        --floors perf/floors.json [--record] [--summary summary.md]

Each run benchmarks the shipped configuration (ONNX Runtime CPU, fp32 bundle
graph) against PyTorch eager *in the same process on the same machine*, and
records, per input shape:

  speedup   median(PyTorch) / median(ONNX Runtime)   hardware-normalized: shared
            CI runners vary run to run, but both sides see the same machine
  p50_ms    absolute ONNX Runtime latency             recorded for trends only

plus the parity-gate fidelity of the exported graph on the real tokenizer.

It fails (exit 1) when, for this hardware class:
  * fidelity is below the release-gate tolerances, or
  * any shape's speedup is > --tolerance below the median of the last
    --window passing runs (a sudden regression), or
  * any shape's speedup is below the committed floor in perf/floors.json
    (slow drift that a rolling baseline would absorb).

With --record the run is appended to the history (failing runs too, flagged,
so they're visible but never become the baseline).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minilm_onnx import export, parity  # noqa: E402
from minilm_onnx.corpus import DOCS, QUERIES  # noqa: E402
from minilm_onnx.model import MAX_SEQ_LENGTH, build_standin_embedder, load_embedder  # noqa: E402
from minilm_onnx.package import real_tokenizer, standin_tokenizer  # noqa: E402
from minilm_onnx.runtimes import OrtRunner, TorchRunner, synthetic_batch  # noqa: E402
from minilm_onnx.serving.tokenize import TextEncoder, pad_batch  # noqa: E402

SHAPES = ((1, 32), (8, 128), (32, 128))


def hardware_class() -> str:
    """Coarse key so runs are only compared with runs on similar hardware."""
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "linux":
            with open("/proc/cpuinfo") as f:
                names = [line.split(":", 1)[1].strip() for line in f if line.startswith("model name")]
            cpu = names[0] if names else cpu
        elif sys.platform == "darwin":
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
    except OSError:
        pass
    runner = "github" if os.environ.get("GITHUB_ACTIONS") == "true" else "local"
    return f"{runner}|{platform.system()}-{platform.machine()}|{os.cpu_count()}cpu|{cpu}"


def measure(model, fp32_path: Path, threads: int, rounds: int, iters: int) -> dict:
    torch_rt, ort_rt = TorchRunner(model, threads), OrtRunner(fp32_path, "ort_cpu", threads)
    out = {}
    for bs, seq in SHAPES:
        batch = synthetic_batch(bs, seq, model.encoder.config.vocab_size)
        for r in (torch_rt, ort_rt):
            for _ in range(3):
                r(batch)
        t_torch, t_ort = [], []
        for _ in range(rounds):  # interleaved, so drift hits both sides equally
            for rt, sink in ((torch_rt, t_torch), (ort_rt, t_ort)):
                for _ in range(iters):
                    t0 = time.perf_counter()
                    rt(batch)
                    sink.append(time.perf_counter() - t0)
        key = f"b{bs}_s{seq}"
        out[key] = {
            "speedup": float(np.median(t_torch) / np.median(t_ort)),
            "p50_ms": float(np.median(t_ort) * 1e3),
            "pytorch_p50_ms": float(np.median(t_torch) * 1e3),
        }
    return out


def fidelity(model, fp32_path: Path, tok_path: Path, threads: int) -> dict:
    enc = TextEncoder(tok_path, MAX_SEQ_LENGTH)
    qb, db = pad_batch(enc.encode(QUERIES), enc.pad_id), pad_batch(enc.encode(DOCS), enc.pad_id)
    ref, rt = TorchRunner(model, threads), OrtRunner(fp32_path, "ort", threads)
    rq, rd, gq, gd = ref(qb), ref(db), rt(qb), rt(db)
    m = parity.elementwise(np.vstack([rq, rd]), np.vstack([gq, gd]))
    m["retrieval"] = parity.retrieval_agreement(rq, rd, gq, gd)
    return {**m, **parity.check("onnx_fp32", m)}


def evaluate(record: dict, history: list[dict], floors: dict, window: int, tolerance: float) -> list[str]:
    fails = []
    if not record["fidelity"]["pass"]:
        fails.append("fidelity: " + "; ".join(record["fidelity"]["failures"]))
    same = [h for h in history if h["hardware"] == record["hardware"] and h.get("pass")][-window:]
    floor = floors.get(record["hardware"]) or floors.get("default", {})
    for shape, m in record["perf"].items():
        if same:
            base = median(h["perf"][shape]["speedup"] for h in same if shape in h["perf"])
            if m["speedup"] < base * (1 - tolerance):
                fails.append(
                    f"{shape}: speedup {m['speedup']:.2f}x is more than {tolerance:.0%} below the "
                    f"median of the last {len(same)} runs ({base:.2f}x)"
                )
        if shape in floor and m["speedup"] < floor[shape]:
            fails.append(f"{shape}: speedup {m['speedup']:.2f}x is below the committed floor {floor[shape]:.2f}x")
    return fails


def summary(record: dict, history: list[dict], fails: list[str], window: int) -> str:
    same = [h for h in history if h["hardware"] == record["hardware"] and h.get("pass")][-window:]
    lines = [
        f"### Performance gate: {'PASS' if not fails else 'FAIL'}",
        "",
        f"`{record['hardware']}` · commit `{record['commit'][:7]}` · baseline = median of last {len(same)} "
        f"passing runs on this hardware class",
        "",
        "| Shape | ORT p50 ms | Speedup vs PyTorch | Baseline speedup |",
        "|---|---:|---:|---:|",
    ]
    for shape, m in record["perf"].items():
        base = median(h["perf"][shape]["speedup"] for h in same if shape in h["perf"]) if same else None
        lines.append(
            f"| {shape} | {m['p50_ms']:.2f} | {m['speedup']:.2f}x | {f'{base:.2f}x' if base else 'n/a (first run)'} |"
        )
    f = record["fidelity"]
    lines += [
        "",
        f"Fidelity vs PyTorch: max abs diff {f['max_abs_diff']:.2e}, worst cosine {f['min_cosine']:.7f}, "
        f"top-1 agreement {f['retrieval']['top1_agreement']:.2f}",
    ]
    if fails:
        lines += ["", "**Failures**", ""] + [f"- {x}" for x in fails]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--history", required=True, type=Path)
    ap.add_argument("--floors", type=Path, default=Path("perf/floors.json"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=0.20)
    ap.add_argument("--record", action="store_true", help="append this run to the history file")
    ap.add_argument("--summary", type=Path, help="write a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()

    standin = args.model == "standin"
    model = build_standin_embedder() if standin else load_embedder(args.model)
    work = Path(tempfile.mkdtemp())
    tok = work / "tokenizer.json"
    (standin_tokenizer(tok, QUERIES + DOCS) if standin else real_tokenizer(args.model, tok))
    fp32 = export.export_fp32(model, work / "model.onnx")

    history = []
    if args.history.exists():
        history = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    floors = json.loads(args.floors.read_text()) if args.floors.exists() else {}
    commit = (
        os.environ.get("GITHUB_SHA")
        or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    )

    record = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit or "unknown",
        "model": args.model,
        "hardware": hardware_class(),
        "threads": args.threads,
        "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__},
        "fidelity": fidelity(model, fp32, tok, args.threads),
        "perf": measure(model, fp32, args.threads, args.rounds, args.iters),
    }
    fails = evaluate(record, history, floors, args.window, args.tolerance)
    record["pass"] = not fails
    record["failures"] = fails

    text = summary(record, history, fails, args.window)
    print(text)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    if args.record:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        with open(args.history, "a") as f:
            f.write(json.dumps(record) + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
