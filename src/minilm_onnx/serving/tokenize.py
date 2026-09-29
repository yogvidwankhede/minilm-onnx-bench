"""Tokenization and padding shared by the packager and the server.

Using one code path for both is deliberate: the golden embeddings stored in
the bundle are produced through exactly the tokenization the server runs,
so a tokenizer config drift (lowercasing, truncation length, special
tokens) fails the startup self-test instead of silently shifting vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


class TextEncoder:
    def __init__(self, tokenizer_path: Path, max_seq_length: int):
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.no_padding()  # the batcher pads per sub-batch, to that sub-batch's longest input
        self.tok.enable_truncation(max_length=max_seq_length)
        self.max_seq_length = max_seq_length
        pad = self.tok.token_to_id("[PAD]")
        self.pad_id = 0 if pad is None else pad

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        """Token ids per text ([CLS] ... [SEP], truncated). Rust-side batch encode."""
        return [np.asarray(e.ids, dtype=np.int64) for e in self.tok.encode_batch(texts)]


def pad_batch(seqs: list[np.ndarray], pad_id: int = 0) -> dict[str, np.ndarray]:
    """Right-pad to the longest sequence; build mask and token types."""
    n, width = len(seqs), max(len(s) for s in seqs)
    ids = np.full((n, width), pad_id, dtype=np.int64)
    mask = np.zeros((n, width), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, : len(s)] = 1
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
