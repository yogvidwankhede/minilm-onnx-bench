"""Generate the README architecture diagram as light and dark SVGs.

    python tools/draw_architecture.py        # writes docs/architecture-{light,dark}.svg

Hand-laid-out on a fixed grid (rather than Mermaid auto-layout) so it renders
identically everywhere; one layout, two palettes matching GitHub's themes.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1300, 724
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

PALETTES = {
    "light": dict(
        lane="#f6f8fa",
        lane_stroke="#d0d7de",
        node="#ffffff",
        node_stroke="#afb8c1",
        text="#1f2328",
        muted="#59636e",
        page="#ffffff",
        accent="#0969da",
        accent_bg="#ddf4ff",
        ok="#1a7f37",
        ok_bg="#dafbe1",
        bad="#cf222e",
        bad_bg="#ffebe9",
        line="#6e7781",
    ),
    "dark": dict(
        lane="#161b22",
        lane_stroke="#30363d",
        node="#0d1117",
        node_stroke="#3d444d",
        text="#e6edf3",
        muted="#9198a1",
        page="#0d1117",
        accent="#4493f8",
        accent_bg="#0c2d6b",
        ok="#3fb950",
        ok_bg="#0f2d1a",
        bad="#f85149",
        bad_bg="#3c1618",
        line="#8b949e",
    ),
}


class Svg:
    def __init__(self, p: dict):
        self.p, self.parts = p, []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, stroke, r=8, sw=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def text(self, x, y, s, size=13, color=None, weight=400, anchor="middle", font=SANS, spacing=None):
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(
            f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color or self.p["text"]}" text-anchor="{anchor}"{ls}>{escape(s)}</text>'
        )

    def node(self, x, y, w, h, title, sub=None, kind="plain"):
        p = self.p
        fill, stroke, sw = {
            "plain": (p["node"], p["node_stroke"], 1),
            "accent": (p["accent_bg"], p["accent"], 1.5),
            "ok": (p["ok_bg"], p["ok"], 1.5),
            "bad": (p["bad_bg"], p["bad"], 1.2),
        }[kind]
        self.rect(x, y, w, h, fill, stroke, sw=sw)
        cx = x + w / 2
        if sub:
            self.text(cx, y + h / 2 - 3, title, 14, weight=600)
            self.text(cx, y + h / 2 + 15, sub, 11.5, p["muted"])
        else:
            self.text(cx, y + h / 2 + 5, title, 14, weight=600)

    def path(self, pts, color, marker, dash=None, sw=1.5):
        d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da} '
            f'stroke-linejoin="round" marker-end="url(#{marker})"/>'
        )

    def label(self, x, y, s, color, bg=None):
        """Text on a line: small, with a background knock-out so the line doesn't strike through it."""
        w = 7.0 * len(s) + 16
        self.rect(x - w / 2, y - 12, w, 18, bg or self.p["lane"], "none", r=4)
        self.text(x, y + 1.5, s, 11.5, color, weight=500)

    def render(self) -> str:
        p = self.p
        markers = "".join(
            f'<marker id="{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>'
            for n, c in (("a", p["line"]), ("acc", p["accent"]), ("ok", p["ok"]), ("bad", p["bad"]))
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-label="Architecture: build-time release gate and serve-time inference path">'
            f"<defs>{markers}</defs>" + "".join(self.parts) + "</svg>"
        )


def lane(s: Svg, y, h, title, command):
    p = s.p
    s.rect(20, y, W - 40, h, p["lane"], p["lane_stroke"], r=12)
    s.text(44, y + 32, title.upper(), 12.5, p["muted"], weight=700, anchor="start", spacing="0.08em")
    s.text(W - 44, y + 32, command, 12.5, p["muted"], anchor="end", font=MONO)


def draw(p: dict) -> str:
    s = Svg(p)
    L, A = p["line"], p["accent"]

    # Both lanes first, so connectors that cross lane edges are painted on top.
    lane(s, 20, 232, "Build time", "python -m minilm_onnx.package")
    lane(s, 300, 404, "Serve time", "python -m minilm_onnx.serving")

    # ---------------- build time ----------------
    ry, rh = 108, 64
    mid = ry + rh / 2
    s.node(44, ry, 180, rh, "PyTorch checkpoint", "fine-tuned MiniLM-L6")
    s.node(260, ry, 196, rh, "ONNX export", "mean pool + L2 norm in graph")
    s.node(492, ry, 170, rh, "int8 (opt-in)", "fp32 is the default")
    s.node(698, ry, 186, rh, "Parity gate", "vs PyTorch, same tokenizer", kind="accent")
    for x0, x1 in ((224, 260), (456, 492), (662, 698)):
        s.path([(x0, mid), (x1 - 2, mid)], L, "a")

    # bundle card
    bx, by, bw, bh = 930, 76, 326, 144
    s.rect(bx, by, bw, bh, p["ok_bg"], p["ok"], sw=1.5)
    s.text(bx + 18, by + 26, "Release bundle", 14, weight=600, anchor="start")
    s.text(bx + bw - 18, by + 26, "content-addressed", 11.5, p["muted"], anchor="end")
    files = (
        ("model.onnx", "graph, fp32 or int8"),
        ("tokenizer.json", "exact serving tokenizer"),
        ("golden.json", "reference embeddings"),
        ("manifest.json", "sha256 · version · parity"),
    )
    for i, (f, what) in enumerate(files):
        yy = by + 54 + i * 22
        s.text(bx + 18, yy, f, 12.5, anchor="start", font=MONO)
        s.text(bx + bw - 18, yy, what, 11.5, p["muted"], anchor="end")
    s.path([(884, mid), (bx - 2, mid)], p["ok"], "ok")
    s.text(907, mid - 10, "pass", 11.5, p["ok"], weight=600)

    # fail branch
    s.path([(791, ry + rh), (791, 194)], p["bad"], "bad")
    s.node(686, 196, 210, 36, "fail: nothing written", kind="bad")

    # bundle -> deploy -> verify (orthogonal elbow through the lane gap)
    verify_cx = 325
    s.path([(bx + bw / 2, by + bh), (bx + bw / 2, 276), (verify_cx, 276), (verify_cx, 390)], A, "acc")
    s.label((bx + bw / 2 + verify_cx) / 2, 276, "deploy: baked into the container image", A, bg=p["page"])

    # ---------------- serve time ----------------

    # row labels
    s.text(44, 418, "Startup", 13, weight=600, anchor="start")
    s.text(44, 436, "once per process", 11.5, p["muted"], anchor="start")
    s.text(44, 558, "Request path", 13, weight=600, anchor="start")
    s.text(44, 576, "every call", 11.5, p["muted"], anchor="start")

    sy, sh = 392, 60
    smid = sy + sh / 2
    s.node(230, sy, 190, sh, "Verify bundle", "SHA-256 of every file")
    s.node(460, sy, 160, sh, "Warm up", "allocate ORT arenas")
    s.node(660, sy, 240, sh, "Golden self-test", "vs PyTorch · own ONNX · batch-of-1")
    s.node(940, sy, 170, sh, "Ready", "/readyz returns 200", kind="ok")
    for x0, x1 in ((420, 460), (620, 660), (900, 940)):
        s.path([(x0, smid), (x1 - 2, smid)], L, "a")
    s.text(1130, smid - 2, "any failure:", 11.5, p["bad"], anchor="start", weight=600)
    s.text(1130, smid + 14, "never becomes ready", 11.5, p["bad"], anchor="start")

    qy, qh = 532, 60
    qmid = qy + qh / 2
    widths = (
        ("Client", 120, "OpenAI SDK, curl", "plain"),
        ("FastAPI", 172, "validate · admit or 503", "plain"),
        ("Tokenizer", 128, "Rust, shared code", "plain"),
        ("Bounded queue", 146, "backpressure", "plain"),
        ("Dynamic batcher", 184, "length-sorted batches", "accent"),
        ("ONNX Runtime", 146, "CPU · CoreML · CUDA", "plain"),
    )
    gap, x, row = 26, 230, []
    for title, w, sub, kind in widths:
        row.append((x, w, title, sub, kind))
        x += w + gap
    for x, w, t, sub, kind in row:
        s.node(x, qy, w, qh, t, sub, kind)
    for (x0, w0, *_), (x1, *_rest) in zip(row, row[1:]):
        s.path([(x0 + w0, qmid), (x1 - 2, qmid)], L, "a")

    # readiness gates admission
    api_cx = row[1][0] + row[1][1] / 2
    s.path([(1025, sy + sh), (1025, 494), (api_cx, 494), (api_cx, qy - 2)], p["ok"], "ok", dash="5 4")
    s.label((1025 + api_cx) / 2, 494, "admits traffic only when ready", p["ok"])

    # response path
    ort_cx, client_cx = row[-1][0] + row[-1][1] / 2, row[0][0] + row[0][1] / 2
    s.path([(ort_cx, qy + qh), (ort_cx, 632), (client_cx, 632), (client_cx, qy + qh + 2)], A, "acc", dash="5 4")
    s.label((ort_cx + client_cx) / 2, 632, "response: 384-d unit vectors, float or base64", A)

    # footer: operations
    s.add(f'<line x1="44" y1="660" x2="{W - 44}" y2="660" stroke="{p["lane_stroke"]}" stroke-width="1"/>')
    s.text(44, 686, "OPERATIONS", 11.5, p["muted"], weight=700, anchor="start", spacing="0.08em")
    s.text(
        150,
        686,
        "/metrics for Prometheus  ·  JSON logs without request text  ·  /livez fails if the batcher dies  ·  "
        "503 + Retry-After when the queue is full  ·  drain on SIGTERM",
        12,
        p["muted"],
        anchor="start",
    )
    return s.render()


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "docs"
    out.mkdir(exist_ok=True)
    for name, palette in PALETTES.items():
        (out / f"architecture-{name}.svg").write_text(draw(palette))
        print(f"wrote docs/architecture-{name}.svg")


if __name__ == "__main__":
    main()
