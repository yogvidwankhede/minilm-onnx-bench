"""Prometheus metrics. One registry per app instance so tests can build many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5, 10)


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter("embed_requests", "Embedding requests by outcome", ["status"], registry=r)
        self.texts = Counter("embed_texts", "Texts embedded (including cache hits)", registry=r)
        self.request_latency = Histogram(
            "embed_request_latency_seconds", "End-to-end request latency", buckets=LATENCY_BUCKETS, registry=r
        )
        self.queue_wait = Histogram(
            "embed_queue_wait_seconds", "Time a text waited before its model call", buckets=LATENCY_BUCKETS, registry=r
        )
        self.model_latency = Histogram(
            "embed_model_latency_seconds", "ONNX Runtime call latency", buckets=LATENCY_BUCKETS, registry=r
        )
        self.batch_size = Histogram(
            "embed_batch_size", "Texts per model call", buckets=(1, 2, 4, 8, 16, 32, 64, 128), registry=r
        )
        self.batch_tokens = Histogram(
            "embed_batch_padded_tokens",
            "Padded tokens per model call",
            buckets=(64, 256, 1024, 2048, 4096, 8192, 16384),
            registry=r,
        )
        self.padding_efficiency = Histogram(
            "embed_padding_efficiency",
            "Real tokens / padded tokens per call",
            buckets=(0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
            registry=r,
        )
        self.queue_depth = Gauge("embed_queue_depth", "Texts waiting for the model", registry=r)
        self.cache_hits = Counter("embed_cache_hits", "Embedding cache hits", registry=r)
        self.cache_misses = Counter("embed_cache_misses", "Embedding cache misses", registry=r)
        self.ready = Gauge("embed_ready", "1 when the service passes readiness", registry=r)
        self.model_info = Gauge(
            "embed_model_info", "Loaded bundle", ["name", "version", "variant", "weights"], registry=r
        )
