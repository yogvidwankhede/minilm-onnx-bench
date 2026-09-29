"""int8 quantization configurations, shared by the packager and tools/quant_search.py.

Dynamic int8 stores MatMul weights as int8 and quantizes activations on the
fly. On trained transformers a single scale per tensor can't cover activation
outliers, so the configs below trade some speed for fidelity by using
per-channel weight scales and/or keeping outlier-heavy MatMuls in fp32.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

# name -> (weights_only, per_channel, rule selecting MatMuls to KEEP in fp32)
#   weights_only: quantize only MatMuls whose second input is a constant weight
#   (ORT's MatMulConstBOnly). Without it, dynamic quantization also quantizes
#   the attention-score products (activation x activation), which are
#   error-prone because both operands carry outliers.
QUANT_CONFIGS: dict[str, tuple[bool, bool, str | None]] = {
    "per_tensor": (False, False, None),
    "weights_only": (True, False, None),
    "weights_only+per_channel": (True, True, None),
    "weights_only+skip_ffn_out": (True, False, "ffn_out"),
    "weights_only+per_channel+skip_ffn_out": (True, True, "ffn_out"),
    "weights_only+attention_only": (True, False, "non_attention"),
    "weights_only+skip_last_2": (True, False, "last_2"),
}

# BERT MatMul names from torch.onnx.export look like:
#   /encoder/layer.3/attention/self/query/MatMul   /encoder/layer.3/attention/output/dense/MatMul
#   /encoder/layer.3/intermediate/dense/MatMul     /encoder/layer.3/output/dense/MatMul  (FFN down-projection)
_LAYER = re.compile(r"layer\.(\d+)/")


def _excluded(graph: onnx.ModelProto, rule: str | None) -> list[str]:
    if rule is None:
        return []
    names = [n.name for n in graph.graph.node if n.op_type in ("MatMul", "Gemm")]
    layers = [int(m.group(1)) for n in names if (m := _LAYER.search(n))]
    last = max(layers) if layers else -1
    out = []
    for n in names:
        is_attn = "/attention/" in n
        is_ffn_out = "/output/dense/" in n and not is_attn
        m = _LAYER.search(n)
        in_last_2 = m is not None and int(m.group(1)) >= last - 1
        if (
            (rule == "ffn_out" and is_ffn_out)
            or (rule == "non_attention" and not is_attn)
            or (rule == "last_2" and in_last_2)
        ):
            out.append(n)
    if rule == "ffn_out" and not out:
        raise ValueError("no FFN down-projection MatMuls found; graph naming changed?")
    return out


@contextlib.contextmanager
def _cwd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def quantize(src: Path, dst: Path, config: str, graph: onnx.ModelProto | None = None) -> Path:
    if config not in QUANT_CONFIGS:
        raise ValueError(f"unknown quant config {config!r}; choose from {sorted(QUANT_CONFIGS)}")
    weights_only, per_channel, rule = QUANT_CONFIGS[config]
    src, dst = Path(src).resolve(), Path(dst).resolve()
    graph = graph or onnx.load(str(src))
    prepped = dst.with_name(dst.stem + ".prep.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        with tempfile.TemporaryDirectory() as tmp, _cwd(tmp):
            quant_pre_process(str(src), str(prepped), skip_optimization=False, skip_symbolic_shape=True)
        model_in = prepped
    except Exception:
        model_in = src
    quantize_dynamic(
        model_input=str(model_in),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_exclude=_excluded(graph, rule),
        extra_options={"MatMulConstBOnly": weights_only},
    )
    prepped.unlink(missing_ok=True)
    return dst
