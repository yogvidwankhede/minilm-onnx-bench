# syntax=docker/dockerfile:1.7
#
# Serving image: ONNX Runtime + Rust tokenizer + FastAPI. No PyTorch.
#
#   docker build --build-arg BUNDLE=bundles/healthmate-minilm-int8-<hash> -t minilm-embed .
#   docker run -p 8080:8080 minilm-embed
#
# The bundle is baked in so one image tag = one exact (code, model) pair:
# deploys are reproducible and rollback is "run the previous tag". To swap
# models without rebuilding, mount a bundle over /models/current instead.

# 3.11+ required by the pinned lock file (numpy 2.4).
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
ARG BUNDLE
RUN test -n "$BUNDLE" || (echo "build with --build-arg BUNDLE=bundles/<name>" >&2; exit 1)
WORKDIR /src
# Normalize bundle permissions here instead of COPY --chmod, whose effect on
# directories differs across BuildKit versions (moby/buildkit#5943).
COPY ${BUNDLE}/ /bundle/
RUN chmod 0755 /bundle && chmod 0444 /bundle/* && test -f /bundle/manifest.json
COPY requirements/serve.lock requirements/serve.lock
RUN pip install --no-cache-dir --prefix=/install -r requirements/serve.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --prefix=/install .

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBED_BUNDLE_DIR=/models/current
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /install /usr/local
COPY --from=build /bundle/ /models/current/
USER 10001
EXPOSE 8080
# Readiness, not liveness: the container is "healthy" only after checksum
# verification, warmup and the golden self-test have passed.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=2)"]
ENTRYPOINT ["python", "-m", "minilm_onnx.serving"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
