import tempfile
from pathlib import Path

import onnx
import pytest

from minilm_onnx import export
from minilm_onnx.model import build_standin_embedder
from minilm_onnx.quant import QUANT_CONFIGS, _excluded, quantize
from minilm_onnx.serving.providers import TargetUnavailable, available_targets, providers_for

TINY = dict(num_hidden_layers=3, hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=2000)


@pytest.fixture(scope="module")
def fp32_graph():
    path = export.export_fp32(build_standin_embedder(**TINY), Path(tempfile.mkdtemp()) / "m.onnx")
    return path, onnx.load(str(path))


def test_cpu_is_always_available_and_every_target_falls_back_to_cpu():
    assert "cpu" in available_targets()
    assert providers_for("cpu") == ["CPUExecutionProvider"]
    from minilm_onnx.serving.providers import TARGETS

    assert all(chain[-1] == "CPUExecutionProvider" for chain in TARGETS.values())


def test_unknown_or_missing_targets_fail_loudly():
    with pytest.raises(ValueError, match="unknown execution target"):
        providers_for("tpu")
    for t in ("coreml", "cuda"):
        if t not in available_targets():
            with pytest.raises(TargetUnavailable):
                providers_for(t)


def test_exclusion_rules_select_the_intended_matmuls(fp32_graph):
    _, g = fp32_graph
    weight_matmuls = 6 * 3  # q, k, v, attention out, ffn up, ffn down per layer
    assert len(_excluded(g, "ffn_out")) == 3
    assert all("/output/dense/" in n and "/attention/" not in n for n in _excluded(g, "ffn_out"))
    assert len(_excluded(g, "non_attention")) == 2 * 3  # ffn up + down per layer
    assert all("layer.1/" in n or "layer.2/" in n for n in _excluded(g, "last_2"))
    assert _excluded(g, None) == [] and weight_matmuls == 18


@pytest.mark.parametrize("config", sorted(QUANT_CONFIGS))
def test_every_quant_config_produces_a_loadable_smaller_or_equal_graph(fp32_graph, config, tmp_path):
    src, _ = fp32_graph
    out = quantize(src, tmp_path / "q.onnx", config)
    onnx.checker.check_model(str(out))
    assert out.stat().st_size <= src.stat().st_size


def test_keeping_layers_in_fp32_keeps_them_as_float_matmuls(fp32_graph, tmp_path):
    src, _ = fp32_graph
    full = onnx.load(str(quantize(src, tmp_path / "a.onnx", "weights_only")))
    partial = onnx.load(str(quantize(src, tmp_path / "b.onnx", "weights_only+skip_ffn_out")))
    count = lambda m, op: sum(n.op_type == op for n in m.graph.node)  # noqa: E731
    assert count(partial, "MatMulInteger") == count(full, "MatMulInteger") - 3
