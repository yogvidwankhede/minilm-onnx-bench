"""PyTorch reference model: BERT encoder + mean pooling + L2 normalize.

This reproduces the sentence-transformers pipeline declared in the model's
modules.json (Transformer -> Pooling(mean) -> Normalize) as a single
nn.Module, so the exported ONNX graph returns final sentence embeddings
rather than raw token states. Keeping pooling inside the graph means the
serving side has nothing to reimplement and nothing to get subtly wrong.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel

DEFAULT_MODEL_ID = "yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold"
MAX_SEQ_LENGTH = 256  # from sentence_bert_config.json

# all-MiniLM-L6-v2 architecture. Used to build a same-shape stand-in when the
# real weights are unreachable (offline tests, sandboxed CI).
MINILM_L6_CONFIG = dict(
    vocab_size=30522,
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=1536,
    max_position_embeddings=512,
    type_vocab_size=2,
    hidden_act="gelu",
)


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors, ignoring padding (sentence-transformers mean pooling)."""
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class SentenceEmbedder(nn.Module):
    def __init__(self, encoder: BertModel, normalize: bool = True):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        emb = mean_pool(out.last_hidden_state, attention_mask)
        if self.normalize:
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


def load_embedder(model_id_or_path: str = DEFAULT_MODEL_ID) -> SentenceEmbedder:
    """Load the fine-tuned model from the Hub or a local directory."""
    encoder = AutoModel.from_pretrained(model_id_or_path, attn_implementation="eager")
    return SentenceEmbedder(encoder).eval()


def build_standin_embedder(seed: int = 0, **overrides) -> SentenceEmbedder:
    """Random-init model with the MiniLM-L6 architecture (or overrides).

    Latency depends on architecture, not weight values, so this is a valid
    stand-in for speed measurements. It is NOT a stand-in for embedding
    quality: never report parity or retrieval numbers from it as properties
    of the fine-tuned model.
    """
    torch.manual_seed(seed)
    cfg = BertConfig(**{**MINILM_L6_CONFIG, **overrides})
    cfg._attn_implementation = "eager"
    return SentenceEmbedder(BertModel(cfg, add_pooling_layer=False)).eval()


def is_local_dir(path: str) -> bool:
    return Path(path).is_dir() and (Path(path) / "config.json").exists()


def describe(model: SentenceEmbedder) -> dict:
    cfg = model.encoder.config
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "layers": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "heads": cfg.num_attention_heads,
        "params_m": round(n_params / 1e6, 2),
    }


__all__ = [
    "DEFAULT_MODEL_ID",
    "MAX_SEQ_LENGTH",
    "SentenceEmbedder",
    "build_standin_embedder",
    "describe",
    "load_embedder",
    "mean_pool",
]
