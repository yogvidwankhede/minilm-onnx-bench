"""Model bundle: the unit that gets built once, verified, and deployed.

    <bundle>/
      model.onnx       the graph (fp32 or int8), pooling + normalize inside
      tokenizer.json   Rust `tokenizers` file, truncation/padding set by the server
      golden.json      reference embeddings for a fixed text set: PyTorch's, and this
                       bundle's own ONNX output at build time
      manifest.json    identity, checksums, shapes, and the parity report that let it ship

The server refuses to become ready unless every checksum matches and the
golden set re-embeds through the full serving path (tokenizer + ONNX) to
(a) within the parity tolerance of PyTorch and (b) within a tight tolerance
of the bundle's own build-time ONNX output. (b) is what catches tokenizer or
runtime drift that (a)'s looser int8 tolerance would let through.
Nothing in this module imports torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_SCHEMA = 3
FILES = ("model.onnx", "tokenizer.json", "golden.json")


class BundleError(RuntimeError):
    """Bundle is missing, corrupted, or incompatible. The server must not start."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Manifest:
    name: str
    version: str
    source_model: str
    weights: str
    variant: str
    embedding_dim: int
    max_seq_length: int
    opset: int
    golden_min_cosine: float  # vs PyTorch reference (parity tolerance for the variant)
    golden_onnx_min_cosine: float  # vs this bundle's own build-time ONNX output
    quant_config: str | None  # int8 config (src/minilm_onnx/quant.py); None for fp32
    sha256: dict
    parity: dict
    created_utc: str
    build: dict
    schema: int = MANIFEST_SCHEMA

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        if d.get("schema") != MANIFEST_SCHEMA:
            raise BundleError(f"unsupported manifest schema {d.get('schema')!r}, expected {MANIFEST_SCHEMA}")
        try:
            return cls(**d)
        except TypeError as exc:
            raise BundleError(f"malformed manifest: {exc}") from exc


def load_manifest(bundle_dir: Path, verify: bool = True) -> Manifest:
    bundle_dir = Path(bundle_dir)
    mpath = bundle_dir / "manifest.json"
    if not mpath.is_file():
        raise BundleError(f"no manifest.json in {bundle_dir}")
    manifest = Manifest.from_dict(json.loads(mpath.read_text()))
    if verify:
        for fname in FILES:
            fpath = bundle_dir / fname
            if not fpath.is_file():
                raise BundleError(f"bundle file missing: {fname}")
            expected = manifest.sha256.get(fname)
            actual = sha256_file(fpath)
            if expected != actual:
                raise BundleError(f"checksum mismatch for {fname}: manifest {expected}, file {actual}")
    return manifest


def bundle_version(variant: str, sha256: dict) -> str:
    """Content address over every bundle file: any change to any file changes the version."""
    digest = hashlib.sha256(json.dumps(sha256, sort_keys=True).encode()).hexdigest()
    return f"{variant}-{digest[:12]}"


def load_golden(bundle_dir: Path) -> dict:
    """{"texts": [...], "torch": [[...]], "onnx": [[...]]}"""
    return json.loads((Path(bundle_dir) / "golden.json").read_text())
