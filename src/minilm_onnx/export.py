"""Export the embedder to ONNX and derive optimized / quantized variants.

Produces up to three graphs:
  model.onnx            fp32, straight from torch.onnx.export
  model.opt.onnx        fp32, ONNX Runtime graph optimizations baked in
                        (attention/GELU/LayerNorm fusion)
  model.int8.onnx       dynamic int8 quantization of the optimized graph
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import warnings
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .model import SentenceEmbedder

INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]
OUTPUT_NAMES = ["sentence_embedding"]
DYNAMIC_AXES = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "token_type_ids": {0: "batch", 1: "seq"},
    "sentence_embedding": {0: "batch"},
}


@contextlib.contextmanager
def _cwd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def dummy_inputs(batch: int = 2, seq: int = 16, vocab: int = 30522) -> tuple[torch.Tensor, ...]:
    ids = torch.randint(1000, vocab, (batch, seq), dtype=torch.long)
    mask = torch.ones(batch, seq, dtype=torch.long)
    mask[-1, seq // 2 :] = 0  # one padded row, so masking is traced, not constant-folded
    types = torch.zeros(batch, seq, dtype=torch.long)
    return ids, mask, types


def export_fp32(model: SentenceEmbedder, out_path: Path, opset: int = 17) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = dummy_inputs(vocab=model.encoder.config.vocab_size)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            args,
            str(out_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=DYNAMIC_AXES,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # TorchScript exporter: stable dynamic_axes for BERT
        )
    onnx.checker.check_model(str(out_path))
    return out_path


def optimize(src: Path, dst: Path) -> Path:
    """Run ORT's extended graph optimizations once and save the fused graph."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    return dst


def quantize_int8(src: Path, dst: Path) -> Path:
    """Dynamic (weight-only static, activation-dynamic) int8 quantization.

    Quantizes MatMul/Gemm weights; activations are quantized on the fly.
    No calibration data needed, which is why it's the usual first try for
    transformer encoders on CPU.
    """
    # ORT's recommended pre-processing (shape inference + cleanup). It can write
    # temp files to the current directory, so run it from a throwaway one.
    src, dst = Path(src).resolve(), Path(dst).resolve()
    prepped = dst.with_name(dst.stem + ".prep.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        # Symbolic shape inference can't resolve BERT's dynamic attention
        # reshapes; ONNX's standard shape inference is enough here.
        with tempfile.TemporaryDirectory() as tmp, _cwd(tmp):
            quant_pre_process(str(src), str(prepped), skip_optimization=False, skip_symbolic_shape=True)
        src = prepped
    except Exception as exc:  # pre-processing is an improvement, not a requirement
        warnings.warn(f"quant_pre_process skipped: {exc}")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    prepped.unlink(missing_ok=True)
    return dst


def export_all(model: SentenceEmbedder, out_dir: Path, opset: int = 17, int8: bool = True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    paths = {"onnx_fp32": export_fp32(model, out_dir / "model.onnx", opset)}
    paths["onnx_fp32_opt"] = optimize(paths["onnx_fp32"], out_dir / "model.opt.onnx")
    if int8:
        # Quantize the un-fused graph: dynamic quant of ORT-fused (contrib-op)
        # graphs is less reliable across ORT versions.
        paths["onnx_int8"] = quantize_int8(paths["onnx_fp32"], out_dir / "model.int8.onnx")
    return paths


def file_mb(path: Path) -> float:
    return round(Path(path).stat().st_size / 1e6, 2)
