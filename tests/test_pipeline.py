import numpy as np
import pytest
import torch

from minilm_onnx import bench, export, parity
from minilm_onnx.model import build_standin_embedder, mean_pool
from minilm_onnx.report import write_markdown
from minilm_onnx.runtimes import OrtRunner, TorchRunner, synthetic_batch

TINY = dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=2000)


@pytest.fixture(scope="module")
def tiny():
    return build_standin_embedder(seed=1, **TINY)


@pytest.fixture(scope="module")
def exported(tiny, tmp_path_factory):
    return export.export_all(tiny, tmp_path_factory.mktemp("onnx"))


def test_mean_pool_ignores_padding():
    tok = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(tok, mask), torch.tensor([[2.0, 2.0]]))


def test_embeddings_are_unit_norm(tiny):
    b = synthetic_batch(4, 12, vocab=2000, pad_frac=0.5)
    emb = TorchRunner(tiny, 1)(b)
    assert emb.shape == (4, 64)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_padding_does_not_change_embedding(tiny):
    b = synthetic_batch(1, 10, vocab=2000)
    padded = {k: np.pad(v, ((0, 0), (0, 22))) for k, v in b.items()}  # pads ids/mask/types with 0
    r = TorchRunner(tiny, 1)
    np.testing.assert_allclose(r(b), r(padded), atol=1e-5)


@pytest.mark.parametrize("variant", ["onnx_fp32", "onnx_fp32_opt"])
@pytest.mark.parametrize("bs,seq", [(1, 5), (3, 40), (7, 128)])
def test_fp32_onnx_matches_pytorch_on_unseen_shapes(tiny, exported, variant, bs, seq):
    """Dynamic axes: shapes differ from the (2, 16) export trace."""
    b = synthetic_batch(bs, seq, vocab=2000, seed=bs * seq, pad_frac=0.5)
    ref = TorchRunner(tiny, 1)(b)
    got = OrtRunner(exported[variant], variant, 1)(b)
    m = parity.elementwise(ref, got)
    assert parity.check(variant, m)["pass"], m


def test_int8_is_close_but_smaller(tiny, exported):
    b = synthetic_batch(8, 32, vocab=2000, pad_frac=0.5)
    m = parity.elementwise(TorchRunner(tiny, 1)(b), OrtRunner(exported["onnx_int8"], "onnx_int8", 1)(b))
    assert m["min_cosine"] > 0.98
    assert export.file_mb(exported["onnx_int8"]) < export.file_mb(exported["onnx_fp32"])


def test_check_reports_failures():
    bad = {"max_abs_diff": 1.0, "min_cosine": 0.5}
    v = parity.check("onnx_fp32", bad)
    assert not v["pass"] and len(v["failures"]) == 2


def test_retrieval_agreement_identical_is_perfect():
    rng = np.random.default_rng(0)
    q, d = rng.normal(size=(5, 8)), rng.normal(size=(12, 8))
    r = parity.retrieval_agreement(q, d, q, d)
    assert r["top1_agreement"] == 1.0 and r["top5_overlap"] == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    base, fast = rng.normal(10, 0.5, 400), rng.normal(5, 0.25, 400)
    lo, hi = bench.bootstrap_speedup(base, fast, 1000, rng)
    assert lo < np.median(base) / np.median(fast) < hi
    assert 1.8 < lo and hi < 2.2


def test_bench_and_report_end_to_end(tiny, exported, tmp_path):
    runners = [TorchRunner(tiny, 1), OrtRunner(exported["onnx_fp32_opt"], "onnx_fp32_opt", 1)]
    cfg = bench.BenchConfig(batch_sizes=[1, 2], seq_lens=[8], warmup=1, min_iters=3, min_seconds=0.05,
                            rounds=1, bootstrap=100)
    rows = bench.run(runners, cfg, vocab=2000, log=lambda *_: None)
    assert len(rows) == 4
    ort_row = next(r for r in rows if r["runtime"] == "onnx_fp32_opt")
    assert ort_row["p50_ms"] > 0 and len(ort_row["speedup_ci95"]) == 2
    res = {
        "model": "tiny", "weights": "random-init stand-in", "parity_source": "synthetic",
        "architecture": {"layers": 2, "hidden": 64, "heads": 4, "params_m": 0.2},
        "environment": {"platform": "x", "machine": "x", "cpu_count": 1, "threads": 1, "torch": "x",
                        "onnxruntime": "x", "transformers": "x", "python": "x", "timestamp_utc": "x"},
        "sizes_mb": {"onnx_fp32": 1.0},
        "parity": {"onnx_fp32_opt": {"max_abs_diff": 0.0, "min_cosine": 1.0, "pass": True, "failures": [],
                                     "retrieval": {"top1_agreement": 1.0, "top5_overlap": 1.0}}},
        "benchmark": rows,
    }
    write_markdown(res, tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text()
    assert "Speedup vs PyTorch" in text and "PASS" in text
