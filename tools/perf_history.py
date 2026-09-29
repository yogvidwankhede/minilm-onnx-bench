"""Render the performance history (perf-history branch) as a trend chart and table.

    python tools/perf_history.py history/history.jsonl history/

Writes trend.png and README.md next to the history file. CI runs this after
every recorded run on main, so the perf-history branch always shows the
latest trend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order
MARKERS = ["o", "s", "D", "^"]
INK, INK_2, GRID, SURFACE, BAD = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#cf222e"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chart(runs: list[dict], out: Path) -> None:
    by_hw: dict[str, list[dict]] = {}
    for r in runs:
        by_hw.setdefault(r["hardware"], []).append(r)
    fig, axes = plt.subplots(len(by_hw), 1, figsize=(10, 3.2 * len(by_hw)), dpi=150, facecolor=SURFACE, squeeze=False)
    for ax, (hw, rs) in zip(axes[:, 0], by_hw.items()):
        ax.set_facecolor(SURFACE)
        shapes = list(rs[-1]["perf"])
        xs = list(range(len(rs)))
        for i, shape in enumerate(shapes):
            ys = [r["perf"].get(shape, {}).get("speedup") for r in rs]
            ax.plot(xs, ys, color=SERIES[i % 4], lw=2, marker=MARKERS[i % 4], ms=5, label=shape, zorder=3)
            for x, y, r in zip(xs, ys, rs):
                if not r.get("pass") and y is not None:
                    ax.scatter([x], [y], s=110, facecolors="none", edgecolors=BAD, linewidths=1.8, zorder=4)
        ax.axhline(1.0, color=INK_2, lw=0.8, ls="--")
        ax.text(xs[-1] + 0.3 if xs else 0, 1.0, "PyTorch parity", va="center", fontsize=7, color=INK_2)
        ax.set_xticks(xs, [r["commit"][:7] for r in rs], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("ONNX Runtime speedup\nvs PyTorch (x)", color=INK_2, fontsize=8)
        ax.set_title(hw.split("|", 1)[-1], fontsize=8, color=INK_2, loc="right")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(shapes))
    fig.suptitle("Performance over time (red ring = gate failed)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def table(runs: list[dict], last: int = 30) -> str:
    shapes = list(runs[-1]["perf"]) if runs else []
    head = "| When (UTC) | Commit | Hardware | " + " | ".join(f"{s} speedup · p50" for s in shapes) + " | Gate |"
    lines = [head, "|" + "---|" * (4 + len(shapes))]
    for r in reversed(runs[-last:]):
        cells = []
        for s in shapes:
            m = r["perf"].get(s)
            cells.append(f"{m['speedup']:.2f}x · {m['p50_ms']:.1f} ms" if m else "n/a")
        gate = "PASS" if r.get("pass") else "FAIL: " + "; ".join(r.get("failures", []))
        lines.append(
            f"| {r['time_utc']} | `{r['commit'][:7]}` | {r['hardware'].split('|', 1)[-1]} | "
            + " | ".join(cells)
            + f" | {gate} |"
        )
    return "\n".join(lines)


def main() -> None:
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    runs = load(src)
    if not runs:
        return
    chart(runs, out_dir / "trend.png")
    (out_dir / "README.md").write_text(
        "# Performance history\n\n"
        "Written by CI on every push to `main` (see `tools/perf_gate.py` on the main branch). Each run times "
        "ONNX Runtime and PyTorch on the same runner with the real fine-tuned weights; the gate fails on a "
        "sudden drop against the rolling median, a breach of the committed floor, or any fidelity loss.\n\n"
        "![trend](trend.png)\n\n## Most recent runs\n\n" + table(runs) + "\n"
    )
    print(f"wrote {out_dir / 'trend.png'} and {out_dir / 'README.md'} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
