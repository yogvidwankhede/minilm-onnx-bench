"""Build a deployable model bundle, gated on parity with PyTorch.

    python -m minilm_onnx.package --model <hf-id|dir|standin> [--variant fp32|int8] [--quant CONFIG]

fp32 is the default. int8 is opt-in: on the fine-tuned weights plain dynamic
int8 fails this gate, and on Apple Silicon no int8 config was faster than
fp32 (ADR 0003). --quant picks the int8 config (tools/quant_search.py).

Nothing is written unless the chosen variant passes its parity gate through
the real serving path (TextEncoder -> ONNX Runtime) against PyTorch on the
same token ids. The bundle is assembled in a staging dir and renamed into
place, so a crashed build never leaves a half-written bundle behind.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import transformers

from . import export, parity
from .corpus import DOCS, QUERIES
from .model import DEFAULT_MODEL_ID, MAX_SEQ_LENGTH, SentenceEmbedder, build_standin_embedder, load_embedder
from .quant import QUANT_CONFIGS, quantize
from .runtimes import OrtRunner, TorchRunner
from .serving.bundle import FILES, MANIFEST_SCHEMA, bundle_version, sha256_file
from .serving.tokenize import TextEncoder, pad_batch

VARIANT_FILES = {"fp32": "model.onnx", "int8": "model.int8.onnx"}
DEFAULT_QUANT = "weights_only+attention_only"  # passed the gate on the fine-tuned weights
VARIANT_TOLERANCE_KEY = {"fp32": "onnx_fp32", "int8": "onnx_int8"}
# Serving must reproduce the bundle's own build-time ONNX output almost exactly.
# Headroom covers int8's batch-composition drift (~2e-5 in cosine, ADR 0003).
GOLDEN_ONNX_MIN_COSINE = 0.9999


class GateFailed(RuntimeError):
    pass


def standin_tokenizer(path: Path, texts: list[str]) -> Path:
    """BERT-style WordPiece tokenizer trained on the corpus, for offline builds.

    Special-token ids differ from the real vocab ([CLS]=2 here vs 101), which
    is fine: the stand-in model has random weights and only needs valid ids.
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok.train_from_iterator(
        texts * 4, trainers.WordPieceTrainer(vocab_size=2000, special_tokens=specials, show_progress=False)
    )
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    tok.save(str(path))
    return path


def real_tokenizer(model_id: str, path: Path) -> Path:
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model_id).backend_tokenizer.save(str(path))
    return path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_bundle(
    model_id: str,
    variant: str,
    out_root: Path,
    name: str = "healthmate-minilm",
    threads: int = 2,
    log=print,
    model: SentenceEmbedder | None = None,
    quant: str = DEFAULT_QUANT,
) -> Path:
    """`model` overrides loading (tests pass a tiny stand-in); `model_id` must then be 'standin'."""
    if variant not in VARIANT_FILES:
        raise ValueError(f"variant must be one of {sorted(VARIANT_FILES)}")
    standin = model_id == "standin"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=out_root))
    try:
        log(f"[1/4] loading {model_id}")
        if model is None:
            model = build_standin_embedder() if standin else load_embedder(model_id)
        elif not standin:
            raise ValueError("a model override is only allowed with model_id='standin'")
        dim = model.encoder.config.hidden_size

        tok_path = staging / "tokenizer.json"
        (standin_tokenizer(tok_path, QUERIES + DOCS) if standin else real_tokenizer(model_id, tok_path))
        enc = TextEncoder(tok_path, MAX_SEQ_LENGTH)

        log(f"[2/4] exporting ({variant})")
        work = staging / "_export"
        src = export.export_fp32(model, work / "model.onnx")
        if variant == "int8":
            src = quantize(src, work / "model.int8.onnx", quant)
        shutil.copyfile(src, staging / "model.onnx")
        shutil.rmtree(work)

        log("[3/4] parity gate (serving tokenizer -> ONNX Runtime vs PyTorch)")
        q_batch = pad_batch(enc.encode(QUERIES), enc.pad_id)
        d_batch = pad_batch(enc.encode(DOCS), enc.pad_id)
        ref, onnx_rt = TorchRunner(model, threads), OrtRunner(staging / "model.onnx", variant, threads)
        ref_q, ref_d = ref(q_batch), ref(d_batch)
        got_q, got_d = onnx_rt(q_batch), onnx_rt(d_batch)
        metrics = parity.elementwise(np.vstack([ref_q, ref_d]), np.vstack([got_q, got_d]))
        metrics["retrieval"] = parity.retrieval_agreement(ref_q, ref_d, got_q, got_d)
        key = VARIANT_TOLERANCE_KEY[variant]
        verdict = parity.check(key, {**metrics, "retrieval": None if standin else metrics["retrieval"]})
        log(
            f"      max|d|={metrics['max_abs_diff']:.2e} min cos={metrics['min_cosine']:.6f} "
            f"top1={metrics['retrieval']['top1_agreement']:.2f} -> {'PASS' if verdict['pass'] else 'FAIL'}"
        )
        if not verdict["pass"]:
            raise GateFailed("parity gate failed, bundle not written: " + "; ".join(verdict["failures"]))

        log("[4/4] writing golden set + manifest")
        texts = QUERIES + DOCS
        golden = np.vstack([ref_q, ref_d])
        # The bundle's own output through the exact serving configuration
        # (ORT_ENABLE_ALL, one batch of all golden texts, as the self-test runs it).
        serving_rt = OrtRunner(staging / "model.onnx", "serving", threads, optimize=True)
        golden_onnx = serving_rt(pad_batch(enc.encode(texts), enc.pad_id))
        (staging / "golden.json").write_text(
            json.dumps(
                {
                    "texts": texts,
                    "torch": [[round(float(x), 7) for x in row] for row in golden],
                    "onnx": [[round(float(x), 7) for x in row] for row in golden_onnx],
                }
            )
        )

        sums = {f: sha256_file(staging / f) for f in FILES}
        version = bundle_version(variant, sums)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "name": name,
            "version": version,
            "source_model": model_id,
            "weights": "random-init stand-in (same architecture)" if standin else "fine-tuned",
            "variant": variant,
            "quant_config": quant if variant == "int8" else None,
            "embedding_dim": dim,
            "max_seq_length": MAX_SEQ_LENGTH,
            "opset": 17,
            "golden_min_cosine": parity.TOLERANCES[key]["min_cos"],
            "golden_onnx_min_cosine": GOLDEN_ONNX_MIN_COSINE,
            "sha256": sums,
            "parity": {**metrics, **verdict},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build": {
                "git_commit": _git_commit(),
                "torch": torch.__version__,
                "onnxruntime": ort.__version__,
                "transformers": transformers.__version__,
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # mkdtemp makes the staging dir 0700; a bundle must be readable by the
        # non-root serving user (uid 10001), whether baked into an image or mounted.
        os.chmod(staging, 0o755)
        for f in (*FILES, "manifest.json"):
            os.chmod(staging / f, 0o644)

        final = out_root / f"{name}-{version}"
        if final.exists():
            shutil.rmtree(final)  # version hashes every file: an existing dir is byte-identical
        os.replace(staging, final)
        log(f"bundle ready: {final}")
        return final
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF id, local dir, or 'standin'")
    ap.add_argument("--variant", choices=sorted(VARIANT_FILES), default="fp32")
    ap.add_argument("--quant", choices=sorted(QUANT_CONFIGS), default=DEFAULT_QUANT, help="int8 config")
    ap.add_argument("--out", default="bundles")
    ap.add_argument("--name", default="healthmate-minilm")
    args = ap.parse_args(argv)
    try:
        build_bundle(args.model, args.variant, Path(args.out), args.name, quant=args.quant)
    except GateFailed as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
