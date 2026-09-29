MODEL ?= yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold
VARIANT ?= int8
THREADS ?= 2
BUNDLE ?= $(firstword $(wildcard bundles/*-$(VARIANT)-*))

.PHONY: install lint test bundle serve loadtest bench bench-standin image

install:
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

test:
	pytest -q

# Gated release bundle. MODEL=standin builds from a same-architecture random-init model (offline).
bundle:
	python -m minilm_onnx.package --model $(MODEL) --variant $(VARIANT) --out bundles

serve:
	EMBED_BUNDLE_DIR=$(BUNDLE) EMBED_INTRA_OP_THREADS=$(THREADS) python -m minilm_onnx.serving --port 8080

# Batching policies compared on pinned CPUs; writes results/serving/
loadtest:
	tools/run_loadtest.sh $(BUNDLE) 1 10
	python tools/report_loadtest.py results/serving/loadtest.jsonl

# Offline ONNX Runtime vs PyTorch benchmark
bench:
	python -m minilm_onnx --model $(MODEL) --threads $(THREADS) --out results/healthmate

bench-standin:
	python -m minilm_onnx --model standin --threads $(THREADS) --out results/standin

image:
	docker build --build-arg BUNDLE=$(BUNDLE) -t minilm-embed:$(notdir $(BUNDLE)) .
