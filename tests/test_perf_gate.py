"""The performance gate must fail on regressions, not just pass on good runs."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("perf_gate", Path(__file__).parents[1] / "tools" / "perf_gate.py")
perf_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf_gate)

HW = "github|Linux-x86_64|4cpu|test"
GOOD_FIDELITY = {"pass": True, "failures": []}


def run(speedups: dict, hw=HW, ok=True, fidelity=GOOD_FIDELITY):
    return {
        "hardware": hw,
        "pass": ok,
        "fidelity": fidelity,
        "perf": {k: {"speedup": v, "p50_ms": 1.0} for k, v in speedups.items()},
    }


HISTORY = [run({"b1_s32": s}) for s in (2.4, 2.5, 2.6, 2.5, 2.4)]


def test_first_run_passes_without_history():
    assert perf_gate.evaluate(run({"b1_s32": 2.5}), [], {}, window=5, tolerance=0.2) == []


def test_within_tolerance_passes():
    assert perf_gate.evaluate(run({"b1_s32": 2.1}), HISTORY, {}, 5, 0.2) == []  # 16% below median 2.5


def test_sudden_regression_fails():
    fails = perf_gate.evaluate(run({"b1_s32": 1.9}), HISTORY, {}, 5, 0.2)  # 24% below
    assert len(fails) == 1 and "below the median of the last 5 runs" in fails[0]


def test_only_same_hardware_and_passing_runs_form_the_baseline():
    other_hw = [run({"b1_s32": 10.0}, hw="local|Darwin-arm64|14cpu|M4")] * 5
    failed = [run({"b1_s32": 0.5}, ok=False)] * 5
    fails = perf_gate.evaluate(run({"b1_s32": 2.4}), HISTORY + other_hw + failed, {}, 5, 0.2)
    assert fails == []


def test_floor_catches_slow_drift_the_rolling_median_absorbed():
    drifted = [run({"b1_s32": s}) for s in (1.6, 1.55, 1.5, 1.5, 1.45)]
    fails = perf_gate.evaluate(run({"b1_s32": 1.45}), drifted, {"default": {"b1_s32": 1.8}}, 5, 0.2)
    assert len(fails) == 1 and "committed floor" in fails[0]


def test_fidelity_failure_fails_regardless_of_speed():
    bad = {"pass": False, "failures": ["min_cosine 0.9 < 0.99999"]}
    fails = perf_gate.evaluate(run({"b1_s32": 5.0}, fidelity=bad), HISTORY, {}, 5, 0.2)
    assert fails and fails[0].startswith("fidelity")
