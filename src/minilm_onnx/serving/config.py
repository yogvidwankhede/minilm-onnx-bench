"""Service configuration, read once from environment variables (12-factor).

Every knob has a production-safe default; the ones that matter for capacity
planning are documented in docs/runbook.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


def _caster(default):
    if isinstance(default, bool):  # bool("false") is True; parse it properly
        return lambda raw: {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}[raw.lower()]
    return type(default)


def _env(name: str, default, cast):
    raw = os.environ.get(f"EMBED_{name.upper()}")
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"EMBED_{name.upper()}={raw!r}: {exc}") from exc


@dataclass(frozen=True)
class Settings:
    bundle_dir: str = "/models/current"
    # batching
    max_batch_size: int = 32  # texts per model call
    max_wait_ms: float = 0.0  # extra wait for company; 0 = batch what queued while the model ran (ADR 0004)
    max_batch_tokens: int = 8192  # padded tokens per model call (batch * longest seq)
    # capacity / protection
    max_queue_texts: int = 2048  # beyond this, shed load with 503 + Retry-After
    request_timeout_s: float = 10.0
    max_texts_per_request: int = 256
    # 256 tokens is ~1-1.5k characters of English; 4096 leaves headroom without letting
    # one request burn seconds of tokenizer CPU on text the model will truncate anyway.
    max_chars_per_text: int = 4096
    max_body_bytes: int = 8 * 1024 * 1024  # checked before JSON parsing (413 above, 411 if unsized)
    # compute
    intra_op_threads: int = 0  # 0 = ORT default (all physical cores)
    cache_size: int = 10000  # LRU embeddings keyed by text hash; 0 disables
    warmup_iters: int = 3
    drain_timeout_s: float = 20.0
    log_level: str = "INFO"
    expose_docs: bool = False  # /docs and /openapi.json; off in production

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        vals = {f.name: _env(f.name, f.default, _caster(f.default)) for f in fields(cls)}
        vals["log_level"] = str(vals["log_level"]).upper()
        vals.update(overrides)
        s = cls(**vals)
        s.validate()
        return s

    def validate(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if self.max_batch_tokens < 256:
            raise ValueError("max_batch_tokens must fit at least one max-length sequence (256)")
        if self.max_texts_per_request < 1 or self.max_chars_per_text < 1:
            raise ValueError("max_texts_per_request and max_chars_per_text must be >= 1")
        if self.request_timeout_s <= 0 or self.drain_timeout_s < 0:
            raise ValueError("request_timeout_s must be > 0 and drain_timeout_s >= 0")
        if self.cache_size < 0 or self.intra_op_threads < 0 or self.warmup_iters < 0:
            raise ValueError("cache_size, intra_op_threads and warmup_iters must be >= 0")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"log_level {self.log_level!r} is not a logging level")
        if self.max_queue_texts < self.max_texts_per_request:
            raise ValueError("max_queue_texts must be >= max_texts_per_request, or large requests always fail")
