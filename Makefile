MODEL ?= yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold
THREADS ?= 4

.PHONY: install test bench bench-standin

install:
	pip install -e ".[dev]"

test:
	pytest -q

# Real fine-tuned weights (needs Hugging Face access)
bench:
	python -m minilm_onnx --model $(MODEL) --threads $(THREADS) --out results/healthmate

# Same-architecture random-init model: speed only, runs offline
bench-standin:
	python -m minilm_onnx --model standin --threads $(THREADS) --out results/standin
