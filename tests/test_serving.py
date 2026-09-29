"""Service tests on tiny bundles (2-layer model), so they run offline in seconds."""

import asyncio
import json
import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from minilm_onnx.model import build_standin_embedder
from minilm_onnx.package import GateFailed, build_bundle
from minilm_onnx.serving import engine as engine_mod
from minilm_onnx.serving.app import create_app
from minilm_onnx.serving.bundle import BundleError, bundle_version, load_manifest, sha256_file
from minilm_onnx.serving.config import Settings
from minilm_onnx.serving.engine import BatchFailed, Engine, LRUCache, NotReady, Overloaded, plan_sub_batches

TINY = dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=2000)
QUIET = dict(warmup_iters=1, intra_op_threads=1)


@pytest.fixture(scope="session")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundles")
    model = build_standin_embedder(seed=3, **TINY)
    return {
        v: build_bundle("standin", v, root, name="tiny", log=lambda *_: None, model=model) for v in ("fp32", "int8")
    }


def settings(bundle, **kw):
    return Settings.from_env(bundle_dir=str(bundle), **{**QUIET, **kw})


def run_engine(bundle, coro_fn, **kw):
    """Start an engine, run coro_fn(engine), always drain."""

    async def main():
        eng = Engine(settings(bundle, **kw))
        await eng.start()
        try:
            return await coro_fn(eng)
        finally:
            await eng.drain()

    return asyncio.run(main())


# ---------------- bundle integrity ----------------


def test_manifest_records_passing_gate(bundles):
    m = load_manifest(bundles["int8"])
    assert m.parity["pass"] and m.variant == "int8" and m.embedding_dim == 64
    assert m.version == bundle_version("int8", m.sha256)


def test_tampered_model_is_rejected(bundles, tmp_path):
    b = shutil.copytree(bundles["fp32"], tmp_path / "b")
    with open(b / "model.onnx", "ab") as f:
        f.write(b"\0")
    with pytest.raises(BundleError, match="checksum mismatch for model.onnx"):
        load_manifest(b)


def test_missing_file_and_bad_schema_are_rejected(bundles, tmp_path):
    b = shutil.copytree(bundles["fp32"], tmp_path / "b")
    (b / "golden.json").unlink()
    with pytest.raises(BundleError, match="missing"):
        load_manifest(b)
    b2 = shutil.copytree(bundles["fp32"], tmp_path / "b2")
    m = json.loads((b2 / "manifest.json").read_text())
    m["schema"] = 99
    (b2 / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BundleError, match="schema"):
        load_manifest(b2)


def test_wrong_weights_fail_golden_self_test(bundles, tmp_path):
    """A validly-checksummed bundle whose graph doesn't match its golden set must not become ready."""
    other = build_bundle(
        "standin",
        "fp32",
        tmp_path / "other",
        name="other",
        log=lambda *_: None,
        model=build_standin_embedder(seed=99, **TINY),
    )
    b = shutil.copytree(bundles["fp32"], tmp_path / "b")
    shutil.copyfile(other / "model.onnx", b / "model.onnx")
    m = json.loads((b / "manifest.json").read_text())
    m["sha256"]["model.onnx"] = sha256_file(b / "model.onnx")
    (b / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BundleError, match="golden self-test failed"):
        run_engine(b, lambda e: asyncio.sleep(0))


def test_gate_blocks_packaging_when_tolerance_is_impossible(tmp_path, monkeypatch):
    from minilm_onnx import parity

    monkeypatch.setitem(parity.TOLERANCES, "onnx_fp32", {"max_abs": 0.0, "min_cos": 1.1})
    with pytest.raises(GateFailed):
        build_bundle("standin", "fp32", tmp_path, log=lambda *_: None, model=build_standin_embedder(**TINY))
    assert not [p for p in tmp_path.iterdir() if not p.name.startswith(".")], "no bundle may be left behind"


def test_bundle_is_readable_by_non_root_and_version_covers_every_file(bundles):
    b = bundles["fp32"]
    assert oct(b.stat().st_mode & 0o777) == "0o755"
    assert all(oct(f.stat().st_mode & 0o777) == "0o644" for f in b.iterdir())
    m = load_manifest(b)
    assert m.version == bundle_version("fp32", m.sha256)
    changed = {**m.sha256, "tokenizer.json": "0" * 64}  # a tokenizer-only change
    assert bundle_version("fp32", changed) != m.version


def test_tokenizer_drift_fails_tight_self_test(bundles, tmp_path):
    """Disable lowercasing, re-checksum so only the golden check can catch it."""
    b = shutil.copytree(bundles["int8"], tmp_path / "b")
    tok = json.loads((b / "tokenizer.json").read_text())
    tok["normalizer"]["lowercase"] = False
    (b / "tokenizer.json").write_text(json.dumps(tok))
    m = json.loads((b / "manifest.json").read_text())
    m["sha256"]["tokenizer.json"] = sha256_file(b / "tokenizer.json")
    (b / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BundleError, match="golden self-test failed"):
        run_engine(b, lambda e: asyncio.sleep(0))


# ---------------- batching logic ----------------


@pytest.mark.parametrize("seed", range(5))
def test_sub_batch_plan_respects_limits_and_covers_every_item(seed):
    rng = np.random.default_rng(seed)
    lengths = list(rng.integers(3, 257, size=int(rng.integers(1, 80))))
    groups = plan_sub_batches(lengths, max_batch_size=16, max_batch_tokens=2048)
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
    for g in groups:
        assert len(g) <= 16
        assert len(g) == 1 or len(g) * max(lengths[i] for i in g) <= 2048


def test_lru_cache_evicts_least_recent():
    c = LRUCache(2)
    a, b, d = (LRUCache.key(t) for t in "abd")
    c.put(a, np.zeros(1))
    c.put(b, np.ones(1))
    c.get(a)
    c.put(d, np.ones(1))
    assert c.get(b) is None and c.get(a) is not None and len(c) == 2


def test_concurrent_requests_keep_order_and_fp32_is_batch_invariant(bundles):
    texts = [f"patient {i} reports {'persistent ' * (i % 7)}cough" for i in range(40)]

    async def go(eng):
        solo = [(await eng.embed([t]))[0][0] for t in texts]  # sequential, batch of one
        together = await asyncio.gather(*(eng.embed([t]) for t in texts))
        return np.stack(solo), np.stack([v[0][0] for v in together])

    solo, together = run_engine(bundles["fp32"], go, cache_size=0, max_wait_ms=20)
    # fp32 ONNX is batch-invariant: same vector whether a text ran alone or padded among 39 others
    np.testing.assert_allclose(solo, together, atol=1e-5)


def test_batching_actually_batches(bundles):
    async def go(eng):
        await asyncio.gather(*(eng.embed([f"text number {i}"]) for i in range(32)))
        h = eng.m.batch_size
        return h._sum.get(), sum(b.get() for b in h._buckets)  # texts, model calls

    texts, calls = run_engine(bundles["fp32"], go, cache_size=0, max_wait_ms=20)
    assert texts == 32 and calls < 32 / 4, f"{calls} model calls for 32 concurrent texts"


def test_int8_is_only_approximately_batch_invariant(bundles):
    """Dynamic quantization picks activation scales per call, so batch companions
    nudge a text's vector. Bound the drift; see docs/adr/0003."""
    texts = [f"symptom {i} " * (1 + i % 9) for i in range(24)]

    async def go(eng):
        solo = np.stack([(await eng.embed([t]))[0][0] for t in texts])
        eng.cache = LRUCache(0)
        both = await asyncio.gather(*(eng.embed([t]) for t in texts))
        return solo, np.stack([b[0][0] for b in both])

    solo, together = run_engine(bundles["int8"], go, cache_size=0, max_wait_ms=20)
    cos = (solo * together).sum(1)
    assert cos.min() > 0.9999, cos.min()  # measured on the full-size stand-in: 0.99998 (ADR 0003)
    assert np.abs(solo - together).max() > 0  # and it really is batch-dependent, unlike fp32


def test_overload_sheds_whole_requests(bundles, monkeypatch):
    async def go(eng):
        real = eng._run
        monkeypatch.setattr(eng, "_run", lambda b: (time.sleep(0.2), real(b))[1])
        results = await asyncio.gather(*(eng.embed([f"t{i}a", f"t{i}b"]) for i in range(10)), return_exceptions=True)
        return results

    res = run_engine(bundles["fp32"], go, cache_size=0, max_queue_texts=4, max_texts_per_request=2, max_batch_size=2)
    shed = [r for r in res if isinstance(r, Overloaded)]
    ok = [r for r in res if isinstance(r, tuple)]
    assert shed and ok and len(shed) + len(ok) == 10
    assert all(r[0].shape == (2, 64) for r in ok)  # admitted requests are complete, never partial


def test_timeout_frees_client_and_engine_recovers(bundles, monkeypatch):
    """A timeout releases the caller, but an ORT call already running can't be
    interrupted: the engine recovers once that call returns (see runbook)."""

    async def go(eng):
        real = eng._run
        monkeypatch.setattr(eng, "_run", lambda b: (time.sleep(0.3), real(b))[1])
        t0 = time.perf_counter()
        with pytest.raises(asyncio.TimeoutError):
            await eng.embed(["slow"])
        assert time.perf_counter() - t0 < 0.25  # caller got its answer at the deadline
        monkeypatch.setattr(eng, "_run", real)
        await asyncio.sleep(0.35)  # the in-flight slow call drains
        return await eng.embed(["fine afterwards"])

    vec, _ = run_engine(bundles["fp32"], go, cache_size=0, request_timeout_s=0.1)
    assert vec.shape == (1, 64)


def test_batch_preparation_error_fails_fast_and_engine_keeps_serving(bundles, monkeypatch):
    calls = {"n": 0}
    real = engine_mod.plan_sub_batches

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return real(*a, **k)

    async def go(eng):
        monkeypatch.setattr(engine_mod, "plan_sub_batches", flaky)
        t0 = time.perf_counter()
        with pytest.raises(BatchFailed):
            await eng.embed(["first"])
        assert time.perf_counter() - t0 < 1  # failed immediately, not at the 10 s timeout
        return await eng.embed(["second"])

    vec, _ = run_engine(bundles["fp32"], go, cache_size=0)
    assert vec.shape == (1, 64)


def test_wrong_row_count_from_model_fails_the_request(bundles, monkeypatch):
    async def go(eng):
        monkeypatch.setattr(eng, "_run", lambda b: np.zeros((0, 64), dtype=np.float32))
        with pytest.raises(BatchFailed):
            await eng.embed(["x"])

    run_engine(bundles["fp32"], go, cache_size=0)


def test_dead_batcher_marks_engine_unhealthy_and_fails_queued_work(bundles, monkeypatch):
    async def main():
        eng = Engine(settings(bundles["fp32"], cache_size=0))
        await eng.start()
        real = eng._run
        monkeypatch.setattr(eng, "_run", lambda b: (time.sleep(0.2), real(b))[1])
        first = asyncio.ensure_future(eng.embed(["in the model when the batcher dies"]))
        await asyncio.sleep(0.05)  # batcher is now inside the slow model call

        async def explode():
            raise RuntimeError("batcher bug")

        eng._gather = explode  # its next iteration dies
        queued = asyncio.ensure_future(eng.embed(["queued at the moment of death"]))
        (vec, _) = await first  # the in-flight call still completes
        assert vec.shape == (1, 64)
        with pytest.raises(BatchFailed):
            await asyncio.wait_for(queued, 1)  # failed immediately, not at the timeout
        assert not eng.healthy and not eng.ready and eng.state() == "unhealthy"
        with pytest.raises(NotReady, match="unhealthy"):
            await eng.embed(["after death"])
        await eng.drain()

    asyncio.run(main())


def test_cancelled_request_releases_its_queued_work(bundles, monkeypatch):
    async def go(eng):
        real = eng._run
        monkeypatch.setattr(eng, "_run", lambda b: (time.sleep(0.2), real(b))[1])
        blocker = asyncio.ensure_future(eng.embed(["occupies the model"]))
        await asyncio.sleep(0.02)
        victim = asyncio.ensure_future(eng.embed(["a", "b", "c"]))
        await asyncio.sleep(0.02)
        victim.cancel()
        await asyncio.sleep(0)
        queued = list(eng._queue._queue)
        assert queued and all(it.future.cancelled() for it in queued)  # batcher will skip them
        await blocker

    run_engine(bundles["fp32"], go, cache_size=0)


def test_drained_engine_refuses_work(bundles):
    async def main():
        eng = Engine(settings(bundles["fp32"]))
        await eng.start()
        await eng.drain()
        with pytest.raises(NotReady):
            await eng.embed(["late"])

    asyncio.run(main())


def test_settings_validation_and_env(monkeypatch, bundles):
    monkeypatch.setenv("EMBED_MAX_WAIT_MS", "7.5")
    monkeypatch.setenv("EMBED_BUNDLE_DIR", str(bundles["fp32"]))
    assert Settings.from_env().max_wait_ms == 7.5
    with pytest.raises(ValueError):
        Settings.from_env(max_batch_size=0)
    monkeypatch.setenv("EMBED_LOG_LEVEL", "info")
    monkeypatch.setenv("EMBED_EXPOSE_DOCS", "false")
    s = Settings.from_env()
    assert s.log_level == "INFO" and s.expose_docs is False
    with pytest.raises(ValueError):
        Settings.from_env(request_timeout_s=0)
    monkeypatch.setenv("EMBED_MAX_BATCH_SIZE", "lots")
    with pytest.raises(ValueError, match="EMBED_MAX_BATCH_SIZE"):
        Settings.from_env()


# ---------------- HTTP API ----------------


@pytest.fixture(scope="module")
def client(bundles):
    with TestClient(create_app(settings(bundles["fp32"], max_texts_per_request=8, max_chars_per_text=200))) as c:
        yield c


def test_health_and_models(client):
    assert client.get("/livez").json() == {"status": "ok"}
    r = client.get("/readyz").json()
    assert r["status"] == "ready" and r["version"].startswith("fp32-")
    m = client.get("/v1/models").json()["data"][0]
    assert m["embedding_dim"] == 64 and m["variant"] == "fp32"


def test_openai_compatible_response(client):
    r = client.post(
        "/v1/embeddings",
        json={"input": ["fever and chills", "knee pain"], "model": "x"},
        headers={"x-request-id": "abc123"},
    )
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc123"
    body = r.json()
    assert body["object"] == "list" and [d["index"] for d in body["data"]] == [0, 1]
    v = np.asarray(body["data"][0]["embedding"])
    assert v.shape == (64,) and abs(np.linalg.norm(v) - 1) < 1e-5
    assert body["usage"]["prompt_tokens"] > 0 and body["model"].startswith("tiny@fp32-")


def test_base64_matches_float(client):
    import base64

    f = client.post("/v1/embeddings", json={"input": "same text"}).json()["data"][0]["embedding"]
    b = client.post("/v1/embeddings", json={"input": "same text", "encoding_format": "base64"}).json()
    decoded = np.frombuffer(base64.b64decode(b["data"][0]["embedding"]), "<f4")
    np.testing.assert_allclose(decoded, f, atol=1e-7)


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({"input": []}, "at least one"),
        ({"input": ["ok", ""]}, "input[1] is empty"),
        ({"input": ["x"] * 9}, "limit is 8"),
        ({"input": "y" * 201}, "limit is 200"),
        ({"input": 5}, "input"),
        ({}, "input"),
        ({"input": "x", "encoding_format": "int8"}, "encoding_format"),
    ],
)
def test_bad_input_is_400_with_openai_error_shape(client, payload, fragment):
    r = client.post("/v1/embeddings", json=payload)
    assert r.status_code == 400
    err = r.json()["error"]
    assert fragment in err["message"] and err["type"] == "invalid_request_error" and err["request_id"]


def test_invalid_unicode_is_400_not_500(client):
    body = b'{"input": ["ok", "bad \\ud800 surrogate"]}'
    r = client.post("/v1/embeddings", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400 and "not valid Unicode" in r.json()["error"]["message"]


def test_body_limits(bundles):
    s = settings(bundles["fp32"], max_body_bytes=2048)
    with TestClient(create_app(s)) as c:
        big = c.post("/v1/embeddings", json={"input": ["x" * 3000]})
        assert big.status_code == 413 and big.json()["error"]["code"] == 413
        chunked = c.post(
            "/v1/embeddings", content=iter([b'{"input": "hi"}']), headers={"content-type": "application/json"}
        )
        assert chunked.status_code == 411


def test_request_id_is_sanitized(client):
    ok = client.post("/v1/embeddings", json={"input": "a"}, headers={"x-request-id": "trace-01.ab_C"})
    assert ok.headers["x-request-id"] == "trace-01.ab_C"
    for bad in ("x" * 65, "has space", "inject\ttab"):
        r = client.post("/v1/embeddings", json={"input": "a"}, headers={"x-request-id": bad})
        assert r.headers["x-request-id"] != bad and len(r.headers["x-request-id"]) == 32


def test_dimensions_parameter(client):
    assert client.post("/v1/embeddings", json={"input": "a", "dimensions": 64}).status_code == 200
    r = client.post("/v1/embeddings", json={"input": "a", "dimensions": 32})
    assert r.status_code == 400 and "returns 64" in r.json()["error"]["message"]


def test_unknown_routes_use_openai_error_shape_and_docs_are_off(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == 404
    assert client.get("/v1/embeddings").status_code == 405
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_liveness_fails_when_worker_is_dead(bundles):
    with TestClient(create_app(settings(bundles["fp32"]))) as c:
        c.app.state.engine.healthy = False
        assert c.get("/livez").status_code == 503
        c.app.state.engine.healthy = True


def test_metrics_exposed(client):
    client.post("/v1/embeddings", json={"input": "metrics probe"})
    text = client.get("/metrics").text
    for name in (
        "embed_requests_total",
        "embed_request_latency_seconds_bucket",
        "embed_batch_size_bucket",
        "embed_padding_efficiency_bucket",
        "embed_queue_depth",
        "embed_ready 1.0",
        "embed_model_info",
    ):
        assert name in text, name
