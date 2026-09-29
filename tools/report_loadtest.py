"""Render results/serving/loadtest.jsonl as a Markdown table + a two-panel chart."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical palette in fixed order; validated for CVD separation on the light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]  # secondary encoding, so identity is never color-alone
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def load(path: Path) -> dict[str, list[dict]]:
    """Group rows by policy and aggregate repetitions: median, plus min-max range."""
    raw: dict[str, dict[int, list[dict]]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        raw.setdefault(r["label"], {}).setdefault(r["concurrency"], []).append(r)
    runs: dict[str, list[dict]] = {}
    for label, by_c in raw.items():
        rows = []
        for c, reps in sorted(by_c.items()):
            row = {"concurrency": c, "reps": len(reps), "errors": sum(r["errors"] for r in reps)}
            for k in ("rps", "p50_ms", "p95_ms", "p99_ms", "mean_texts_per_model_call", "server_request_ms"):
                vals = [r[k] for r in reps]
                row[k], row[k + "_min"], row[k + "_max"] = median(vals), min(vals), max(vals)
            rows.append(row)
        runs[label] = rows
    return runs


def _rng(r: dict, k: str, fmt: str) -> str:
    if r["reps"] == 1:
        return format(r[k], fmt)
    return f"{format(r[k], fmt)} ({format(r[k + '_min'], fmt)}–{format(r[k + '_max'], fmt)})"


def table(runs: dict[str, list[dict]]) -> str:
    reps = max(r["reps"] for rows in runs.values() for r in rows)
    out = [
        f"Median of {reps} interleaved repetitions; min–max across repetitions in parentheses.",
        "",
        "| Policy | Concurrency | Req/s | p50 ms | p99 ms | Server-side mean ms | Texts per model call | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, rows in runs.items():
        for r in rows:
            out.append(
                f"| {label} | {r['concurrency']} | {_rng(r, 'rps', '.0f')} | {_rng(r, 'p50_ms', '.1f')} | "
                f"{_rng(r, 'p99_ms', '.1f')} | {r['server_request_ms']:.1f} | "
                f"{r['mean_texts_per_model_call']:.1f} | {r['errors']} |"
            )
    return "\n".join(out)


def _label_ends(fig, ax, ends, min_gap_pt: float = 10.5) -> None:
    """Direct-label line ends, nudging labels apart vertically so they never overlap."""
    fig.canvas.draw()
    min_gap_px = min_gap_pt * fig.dpi / 72  # one text line at 7.5pt, plus leading
    to_px = ax.transData.transform
    pts = sorted(((label, *to_px((x, y))) for label, x, y in ends), key=lambda p: p[2])
    placed: list[float] = []
    for _label, _, py in pts:
        y = py if not placed or py - placed[-1] >= min_gap_px else placed[-1] + min_gap_px
        placed.append(y)
    for (label, px, py), y in zip(pts, placed):
        ax.annotate(
            label,
            xy=ax.transData.inverted().transform((px, py)),
            xytext=(7, (y - py) * 72 / fig.dpi),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK,
        )


def chart(runs: dict[str, list[dict]], path: Path, title: str) -> None:
    plt.rcParams.update(
        {"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160, facecolor=SURFACE)
    panels = [("rps", "Throughput (requests/s)", False), ("p99_ms", "p99 latency (ms, log scale)", True)]
    for ax, (key, ylabel, logy) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ends = []
        for i, (label, rows) in enumerate(runs.items()):
            xs = [r["concurrency"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(
                xs,
                ys,
                color=SERIES[i],
                lw=2,
                marker=MARKERS[i],
                ms=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label,
                zorder=3,
            )
            if rows[0].get("reps", 1) > 1:
                lo = [r[key + "_min"] for r in rows]
                hi = [r[key + "_max"] for r in rows]
                ax.fill_between(xs, lo, hi, color=SERIES[i], alpha=0.12, linewidth=0, zorder=2)
            ends.append((label, xs[-1], ys[-1]))
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        ax.set_xlim(0.8, 300)
        if logy:
            ax.set_yscale("log")
            ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
            ax.set_yticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
        else:
            ax.set_ylim(0, None)
        _label_ends(fig, ax, ends)
        ax.set_xlabel("concurrent clients")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    fig.suptitle(title + "  (median; band = min–max over repetitions)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/serving/loadtest.jsonl")
    title = sys.argv[2] if len(sys.argv) > 2 else "Dynamic batching policies, same bundle and hardware"
    runs = load(src)
    chart(runs, src.with_suffix(".png"), title)
    src.with_suffix(".md").write_text(table(runs) + "\n")
    print(f"wrote {src.with_suffix('.png')} and {src.with_suffix('.md')}")


if __name__ == "__main__":
    main()
