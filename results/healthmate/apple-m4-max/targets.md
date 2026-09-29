# Execution targets: `yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold` (fine-tuned weights)

Machine: Apple M4 Max (arm64, 14 cores) · torch 2.14.0 · onnxruntime 1.30.0 · 4 CPU threads · int8 config `weights_only+attention_only`

## Fidelity vs PyTorch on CPU

| Target | max abs diff | worst cosine | top-1 agreement | top-5 overlap |
|---|---:|---:|---:|---:|
| PyTorch, Apple GPU (MPS) | 1.64e-07 | 1.000000 | 1.00 | 1.00 |
| ONNX Runtime, CPU | 1.86e-07 | 1.000000 | 1.00 | 1.00 |
| ONNX Runtime, CoreML | 1.78e-07 | 1.000000 | 1.00 | 1.00 |
| ONNX Runtime, CPU, int8 | 9.21e-03 | 0.998777 | 1.00 | 1.00 |

Not measured here: PyTorch, NVIDIA GPU (device not available); ONNX Runtime, CUDA (provider not in this onnxruntime build)

## Latency (model only)

Speedup is median(PyTorch CPU) / median(target) with a 95% bootstrap CI.

| batch | seq | Target | p50 ms | p95 ms | Speedup vs PyTorch CPU |
|---:|---:|---|---:|---:|---|
| 1 | 32 | PyTorch, CPU | 2.49 | 2.87 | 1.00x (baseline) |
| 1 | 32 | PyTorch, Apple GPU (MPS) | 2.91 | 4.26 | 0.85x [0.85, 0.86] |
| 1 | 32 | ONNX Runtime, CPU | 1.24 | 1.35 | 2.00x [1.99, 2.02] |
| 1 | 32 | ONNX Runtime, CoreML | 5.30 | 5.69 | 0.47x [0.47, 0.47] |
| 1 | 32 | ONNX Runtime, CPU, int8 | 1.24 | 1.29 | 2.01x [2.00, 2.03] |
| 8 | 32 | PyTorch, CPU | 8.10 | 8.32 | 1.00x (baseline) |
| 8 | 32 | PyTorch, Apple GPU (MPS) | 4.08 | 5.12 | 1.99x [1.96, 2.03] |
| 8 | 32 | ONNX Runtime, CPU | 5.75 | 7.09 | 1.41x [1.40, 1.42] |
| 8 | 32 | ONNX Runtime, CoreML | 25.84 | 27.10 | 0.31x [0.31, 0.32] |
| 8 | 32 | ONNX Runtime, CPU, int8 | 6.23 | 6.67 | 1.30x [1.29, 1.31] |
| 32 | 32 | PyTorch, CPU | 20.24 | 21.55 | 1.00x (baseline) |
| 32 | 32 | PyTorch, Apple GPU (MPS) | 4.90 | 5.53 | 4.13x [4.10, 4.16] |
| 32 | 32 | ONNX Runtime, CPU | 24.63 | 28.25 | 0.82x [0.82, 0.83] |
| 32 | 32 | ONNX Runtime, CoreML | 99.46 | 103.46 | 0.20x [0.20, 0.21] |
| 32 | 32 | ONNX Runtime, CPU, int8 | 22.80 | 24.31 | 0.89x [0.88, 0.90] |
| 1 | 128 | PyTorch, CPU | 5.69 | 6.28 | 1.00x (baseline) |
| 1 | 128 | PyTorch, Apple GPU (MPS) | 3.28 | 5.46 | 1.74x [1.68, 1.84] |
| 1 | 128 | ONNX Runtime, CPU | 3.50 | 3.80 | 1.62x [1.62, 1.63] |
| 1 | 128 | ONNX Runtime, CoreML | 16.03 | 17.67 | 0.35x [0.35, 0.36] |
| 1 | 128 | ONNX Runtime, CPU, int8 | 3.66 | 3.95 | 1.55x [1.55, 1.56] |
| 8 | 128 | PyTorch, CPU | 20.35 | 20.92 | 1.00x (baseline) |
| 8 | 128 | PyTorch, Apple GPU (MPS) | 5.48 | 5.71 | 3.71x [3.70, 3.72] |
| 8 | 128 | ONNX Runtime, CPU | 25.11 | 28.18 | 0.81x [0.81, 0.81] |
| 8 | 128 | ONNX Runtime, CoreML | 111.06 | 113.91 | 0.18x [0.18, 0.18] |
| 8 | 128 | ONNX Runtime, CPU, int8 | 23.57 | 24.50 | 0.86x [0.86, 0.87] |
| 32 | 128 | PyTorch, CPU | 74.37 | 81.12 | 1.00x (baseline) |
| 32 | 128 | PyTorch, Apple GPU (MPS) | 17.54 | 17.99 | 4.24x [4.23, 4.26] |
| 32 | 128 | ONNX Runtime, CPU | 103.89 | 116.84 | 0.72x [0.69, 0.72] |
| 32 | 128 | ONNX Runtime, CoreML | 450.76 | 453.97 | 0.16x [0.16, 0.17] |
| 32 | 128 | ONNX Runtime, CPU, int8 | 102.18 | 114.15 | 0.73x [0.68, 0.78] |
