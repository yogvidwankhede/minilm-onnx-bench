"""Execution targets for ONNX Runtime: CPU, CoreML (Apple GPU / Neural Engine), CUDA.

Every target keeps CPUExecutionProvider as the fallback, so operators a
provider doesn't support still run. Whether a target actually helps is a
measurement question per model and per machine; tools/target_bench.py
answers it, and the startup self-test guards fidelity on whatever runs.
"""

from __future__ import annotations

import onnxruntime as ort

TARGETS: dict[str, list] = {
    "cpu": ["CPUExecutionProvider"],
    "coreml": [
        (
            "CoreMLExecutionProvider",
            {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "0"},
        ),
        "CPUExecutionProvider",
    ],
    "cuda": [("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
}


class TargetUnavailable(RuntimeError):
    pass


def providers_for(target: str) -> list:
    if target not in TARGETS:
        raise ValueError(f"unknown execution target {target!r}; choose from {sorted(TARGETS)}")
    chain = TARGETS[target]
    primary = chain[0] if isinstance(chain[0], str) else chain[0][0]
    if primary not in ort.get_available_providers():
        raise TargetUnavailable(
            f"{primary} is not available in this onnxruntime build "
            f"(available: {', '.join(ort.get_available_providers())})"
        )
    return chain


def available_targets() -> list[str]:
    out = []
    for t in TARGETS:
        try:
            providers_for(t)
            out.append(t)
        except TargetUnavailable:
            pass
    return out
