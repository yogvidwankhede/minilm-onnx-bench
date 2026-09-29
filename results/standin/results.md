# ONNX Runtime vs PyTorch: `standin`

- Weights: **random-init stand-in (same architecture)**
- Architecture: 6 layers, hidden 384, 12 heads, 22.57M params
- Machine: Linux-6.18.44-fc-v37-x86_64-with-glibc2.39 (x86_64), 2 logical CPUs, **2 threads** for every runtime
- Versions: torch 2.14.0+cu130, onnxruntime 1.25.0, transformers 4.57.6, Python 3.11.15
- Run at 2026-09-29T19:57:59Z

## Parity vs PyTorch

Inputs: synthetic token ids with variable padding (16 queries x 40 docs).

| Variant | max abs diff | min cosine | top-1 agreement | top-5 overlap | Gate |
|---|---:|---:|---:|---:|---|
| ONNX Runtime fp32, no graph opts | 7.45e-08 | 1.000000 | 1.00 | 1.00 | PASS |
| ONNX Runtime fp32, fused graph | 7.45e-08 | 1.000000 | 1.00 | 1.00 | PASS |
| ONNX Runtime int8 (dynamic quant) | 1.68e-03 | 0.999955 | 1.00 | 0.97 | PASS |

_Retrieval agreement uses random weights here, so it is reported but not gated._

## Model size on disk

| Variant | MB |
|---|---:|
| ONNX Runtime fp32, no graph opts | 90.25 |
| ONNX Runtime fp32, fused graph | 90.2 |
| ONNX Runtime int8 (dynamic quant) | 58.5 |
| PyTorch fp32 (eager) | 90.26 |

## Latency (model only, tokenization excluded)

Speedup is median(PyTorch) / median(variant) with a 95% bootstrap CI.

| batch | seq | Runtime | p50 ms | p95 ms | sent/s | Speedup vs PyTorch |
|---:|---:|---|---:|---:|---:|---|
| 1 | 32 | PyTorch fp32 (eager) | 24.42 | 60.66 | 41 | 1.00x (baseline) |
| 1 | 32 | ONNX Runtime fp32, no graph opts | 10.03 | 14.30 | 100 | 2.43x [2.32, 2.59] |
| 1 | 32 | ONNX Runtime fp32, fused graph | 9.11 | 15.46 | 110 | 2.68x [2.56, 2.88] |
| 1 | 32 | ONNX Runtime int8 (dynamic quant) | 5.16 | 9.71 | 194 | 4.73x [4.49, 5.05] |
| 8 | 32 | PyTorch fp32 (eager) | 54.44 | 100.66 | 147 | 1.00x (baseline) |
| 8 | 32 | ONNX Runtime fp32, no graph opts | 49.55 | 62.74 | 161 | 1.10x [1.02, 1.29] |
| 8 | 32 | ONNX Runtime fp32, fused graph | 41.90 | 76.76 | 191 | 1.30x [1.19, 1.52] |
| 8 | 32 | ONNX Runtime int8 (dynamic quant) | 23.46 | 52.33 | 341 | 2.32x [2.13, 2.78] |
| 32 | 32 | PyTorch fp32 (eager) | 175.15 | 234.55 | 183 | 1.00x (baseline) |
| 32 | 32 | ONNX Runtime fp32, no graph opts | 178.46 | 200.42 | 179 | 0.98x [0.91, 1.06] |
| 32 | 32 | ONNX Runtime fp32, fused graph | 157.94 | 176.86 | 203 | 1.11x [0.99, 1.18] |
| 32 | 32 | ONNX Runtime int8 (dynamic quant) | 83.40 | 115.62 | 384 | 2.10x [1.88, 2.26] |
| 1 | 128 | PyTorch fp32 (eager) | 38.80 | 64.81 | 26 | 1.00x (baseline) |
| 1 | 128 | ONNX Runtime fp32, no graph opts | 30.93 | 41.40 | 32 | 1.25x [1.07, 1.45] |
| 1 | 128 | ONNX Runtime fp32, fused graph | 28.49 | 41.38 | 35 | 1.36x [1.15, 1.60] |
| 1 | 128 | ONNX Runtime int8 (dynamic quant) | 13.30 | 21.12 | 75 | 2.92x [2.41, 3.49] |
| 8 | 128 | PyTorch fp32 (eager) | 211.41 | 251.57 | 38 | 1.00x (baseline) |
| 8 | 128 | ONNX Runtime fp32, no graph opts | 181.38 | 204.56 | 44 | 1.17x [1.05, 1.32] |
| 8 | 128 | ONNX Runtime fp32, fused graph | 178.59 | 204.85 | 45 | 1.18x [1.06, 1.35] |
| 8 | 128 | ONNX Runtime int8 (dynamic quant) | 87.12 | 116.78 | 92 | 2.43x [2.11, 2.76] |
| 32 | 128 | PyTorch fp32 (eager) | 853.98 | 941.19 | 37 | 1.00x (baseline) |
| 32 | 128 | ONNX Runtime fp32, no graph opts | 914.56 | 1113.44 | 35 | 0.93x [0.88, 0.96] |
| 32 | 128 | ONNX Runtime fp32, fused graph | 722.86 | 821.94 | 44 | 1.18x [1.13, 1.26] |
| 32 | 128 | ONNX Runtime int8 (dynamic quant) | 435.66 | 492.61 | 73 | 1.96x [1.86, 2.10] |
| 1 | 256 | PyTorch fp32 (eager) | 75.28 | 98.48 | 13 | 1.00x (baseline) |
| 1 | 256 | ONNX Runtime fp32, no graph opts | 55.30 | 69.33 | 18 | 1.36x [1.27, 1.61] |
| 1 | 256 | ONNX Runtime fp32, fused graph | 57.65 | 85.86 | 17 | 1.31x [1.10, 1.50] |
| 1 | 256 | ONNX Runtime int8 (dynamic quant) | 30.43 | 44.63 | 33 | 2.47x [2.30, 2.59] |
| 8 | 256 | PyTorch fp32 (eager) | 581.50 | 743.80 | 14 | 1.00x (baseline) |
| 8 | 256 | ONNX Runtime fp32, no graph opts | 482.26 | 583.69 | 17 | 1.21x [1.08, 1.31] |
| 8 | 256 | ONNX Runtime fp32, fused graph | 419.36 | 462.39 | 19 | 1.39x [1.28, 1.53] |
| 8 | 256 | ONNX Runtime int8 (dynamic quant) | 234.94 | 272.87 | 34 | 2.48x [2.27, 2.86] |
| 32 | 256 | PyTorch fp32 (eager) | 2918.81 | 3379.54 | 11 | 1.00x (baseline) |
| 32 | 256 | ONNX Runtime fp32, no graph opts | 1971.14 | 2351.13 | 16 | 1.48x [1.42, 1.60] |
| 32 | 256 | ONNX Runtime fp32, fused graph | 1621.36 | 2033.95 | 20 | 1.80x [1.70, 1.90] |
| 32 | 256 | ONNX Runtime int8 (dynamic quant) | 912.69 | 1021.67 | 35 | 3.20x [3.07, 3.36] |
