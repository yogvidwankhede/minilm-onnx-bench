"""Render results.json as a Markdown report (and a latency chart if matplotlib is present)."""

from __future__ import annotations

from pathlib import Path

LABELS = {
    "pytorch_fp32": "PyTorch fp32 (eager)",
    "onnx_fp32": "ONNX Runtime fp32, no graph opts",
    "onnx_fp32_opt": "ONNX Runtime fp32, fused graph",
    "onnx_int8": "ONNX Runtime int8 (dynamic quant)",
}


def _fmt_ci(row) -> str:
    if "speedup_vs_pytorch" not in row:
        return "1.00x (baseline)"
    lo, hi = row["speedup_ci95"]
    return f"{row['speedup_vs_pytorch']:.2f}x [{lo:.2f}, {hi:.2f}]"


def write_markdown(res: dict, path: Path) -> None:
    env, arch = res["environment"], res["architecture"]
    L = [
        f"# ONNX Runtime vs PyTorch: `{res['model']}`",
        "",
        f"- Weights: **{res['weights']}**",
        f"- Architecture: {arch['layers']} layers, hidden {arch['hidden']}, {arch['heads']} heads, "
        f"{arch['params_m']}M params",
        f"- Machine: {env['platform']} ({env['machine']}), {env['cpu_count']} logical CPUs, "
        f"**{env['threads']} threads** for every runtime",
        f"- Versions: torch {env['torch']}, onnxruntime {env['onnxruntime']}, "
        f"transformers {env['transformers']}, Python {env['python']}",
        f"- Run at {env['timestamp_utc']}",
        "",
        "## Parity vs PyTorch",
        "",
        f"Inputs: {res['parity_source']}.",
        "",
        "| Variant | max abs diff | min cosine | top-1 agreement | top-5 overlap | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, p in res["parity"].items():
        r = p["retrieval"]
        gate = "PASS" if p["pass"] else "FAIL: " + "; ".join(p["failures"])
        L.append(
            f"| {LABELS.get(name, name)} | {p['max_abs_diff']:.2e} | {p['min_cosine']:.6f} | "
            f"{r['top1_agreement']:.2f} | {r.get('top5_overlap', float('nan')):.2f} | {gate} |"
        )
    if "random" in res["weights"]:
        L += ["", "_Retrieval agreement uses random weights here, so it is reported but not gated._"]

    L += ["", "## Model size on disk", "", "| Variant | MB |", "|---|---:|"]
    for k, v in res["sizes_mb"].items():
        L.append(f"| {LABELS.get(k, k)} | {v} |")

    L += [
        "",
        "## Latency (model only, tokenization excluded)",
        "",
        "Speedup is median(PyTorch) / median(variant) with a 95% bootstrap CI.",
        "",
        "| batch | seq | Runtime | p50 ms | p95 ms | sent/s | Speedup vs PyTorch |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in res["benchmark"]:
        L.append(
            f"| {row['batch']} | {row['seq']} | {LABELS.get(row['runtime'], row['runtime'])} | "
            f"{row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['throughput_sent_per_s']:.0f} | {_fmt_ci(row)} |"
        )
    path.write_text("\n".join(L) + "\n")
    _maybe_plot(res, path.with_name("latency.png"))


def _maybe_plot(res: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = res["benchmark"]
    runtimes = list(dict.fromkeys(r["runtime"] for r in rows))
    cells = list(dict.fromkeys((r["batch"], r["seq"]) for r in rows))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(cells)), 3.6), dpi=150)
    width = 0.8 / len(runtimes)
    colors = ["#8a8f98", "#9bb7d4", "#2f6fb0", "#e0833a"]
    for i, rt in enumerate(runtimes):
        vals = [next(r["p50_ms"] for r in rows if r["runtime"] == rt and (r["batch"], r["seq"]) == c) for c in cells]
        ax.bar([j + i * width for j in range(len(cells))], vals, width, label=LABELS.get(rt, rt), color=colors[i % 4])
    ax.set_xticks([j + width * (len(runtimes) - 1) / 2 for j in range(len(cells))])
    ax.set_xticklabels([f"b{b}\ns{s}" for b, s in cells], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("p50 latency (ms, log)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title(f"{res['architecture']['layers']}-layer MiniLM, {res['environment']['threads']} threads", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
