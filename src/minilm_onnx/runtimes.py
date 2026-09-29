"""Uniform runners so PyTorch and ONNX Runtime are timed through the same code path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .model import SentenceEmbedder

Batch = dict[str, np.ndarray]  # input_ids / attention_mask / token_type_ids, int64


class TorchRunner:
    def __init__(self, model: SentenceEmbedder, threads: int):
        self.name = "pytorch_fp32"
        self.model = model.eval()
        self.threads = threads

    def __call__(self, batch: Batch) -> np.ndarray:
        torch.set_num_threads(self.threads)
        with torch.inference_mode():
            out = self.model(**{k: torch.from_numpy(v) for k, v in batch.items()})
        return out.numpy()


class OrtRunner:
    def __init__(self, path: Path, name: str, threads: int, optimize: bool = True):
        self.name = name
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def __call__(self, batch: Batch) -> np.ndarray:
        feed = {k: v for k, v in batch.items() if k in self.input_names}
        return self.sess.run(None, feed)[0]


def synthetic_batch(batch: int, seq: int, vocab: int = 30522, seed: int = 0, pad_frac: float = 0.0) -> Batch:
    """Token-id batch with [CLS] ... [SEP] framing and optional right padding."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(1000, vocab, size=(batch, seq), dtype=np.int64)
    ids[:, 0] = 101  # [CLS]
    mask = np.ones((batch, seq), dtype=np.int64)
    if pad_frac > 0:
        for row in range(batch):
            keep = max(2, int(seq * (1 - rng.uniform(0, pad_frac))))
            ids[row, keep - 1] = 102  # [SEP]
            ids[row, keep:] = 0
            mask[row, keep:] = 0
    else:
        ids[:, -1] = 102
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "token_type_ids": np.zeros((batch, seq), dtype=np.int64),
    }
