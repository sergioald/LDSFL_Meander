"""Short, warmed, relative performance checks for accepted exact backends.

Run from the repository root. No solver simulation or solver outputs are created.
An optional JSON report can be written with ``--output``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPETITIONS = 5
UNWRAP_CALLS_PER_SAMPLE = 25

# Keep all blocking timing policy in one place. The report adds the measured
# median ratio, sample variability, and any one-time remeasurement.
PERFORMANCE_FLOORS: dict[str, dict[str, Any]] = {
    "vertical_k0123": {
        "metric_name": "vertical_k0123_speedup",
        "reference": "ldsfl.vertical.k0123 (uncached Python reference)",
        "candidate": "ldsfl.vertical_numba._k0123_kernel (warmed raw compiled calculation)",
        "repetitions": REPETITIONS,
        "floor_speedup": 3.0,
        "safety_margin_rationale": (
            "Historical local raw-kernel speedup exceeded 500x; a 3x floor "
            "allows large hosted-runner variation while detecting major regressions."
        ),
    },
    "semiana_recursive_n1000": {
        "metric_name": "semiana_recursive_speedup_n1000",
        "reference": "legacy cached Numba finite-window response",
        "candidate": "recursive cached Numba finite-window response",
        "repetitions": REPETITIONS,
        "floor_speedup": 2.0,
        "safety_margin_rationale": (
            "Historical N=1000 response speedups were above 8x; 2x is a broad "
            "regression floor, not a small-noise target."
        ),
    },
    "semiana_recursive_n2500": {
        "metric_name": "semiana_recursive_speedup_n2500",
        "reference": "legacy cached Numba finite-window response",
        "candidate": "recursive cached Numba finite-window response",
        "repetitions": REPETITIONS,
        "floor_speedup": 2.0,
        "safety_margin_rationale": (
            "Historical N=2500 response speedups were about 25x; 2x leaves "
            "substantial room for runner variation and catches loss of recurrence scaling."
        ),
    },
    "neck_grid_no_hit": {
        "metric_name": "neck_numba_grid_no_hit_speedup",
        "reference": "ldsfl.mathutils._kdtree_first_hit_point_pair",
        "candidate": "ldsfl.neck_numba.spatial_grid_first_hit_point_pair",
        "repetitions": REPETITIONS,
        "floor_speedup": 2.0,
        "safety_margin_rationale": (
            "Validated production replay was about 20x; the 2x detector-only "
            "floor is deliberately conservative."
        ),
    },
    "neck_grid_late_hit": {
        "metric_name": "neck_numba_grid_late_hit_speedup",
        "reference": "ldsfl.mathutils._kdtree_first_hit_point_pair",
        "candidate": "ldsfl.neck_numba.spatial_grid_first_hit_point_pair",
        "repetitions": REPETITIONS,
        "floor_speedup": 2.0,
        "safety_margin_rationale": (
            "Validated production replay was about 20x; the 2x detector-only "
            "floor is deliberately conservative."
        ),
    },
    "geometry_unwrap": {
        "metric_name": "geometry_unwrap_exactness",
        "reference": "ldsfl.mathutils.unwrap_angles_like_matlab",
        "candidate": "ldsfl.geometry_numba.unwrap_angles_like_matlab_numba",
        "repetitions": REPETITIONS,
        "floor_speedup": None,
        "safety_margin_rationale": (
            "The production-sized operation is too short and runner-sensitive "
            "for a blocking timing threshold; exact output and kernel options are gated."
        ),
    },
}


def _timing_stats(samples: list[float]) -> dict[str, Any]:
    median = statistics.median(samples)
    spread = max(samples) - min(samples)
    return {
        "samples_seconds": [float(value) for value in samples],
        "min_seconds": float(min(samples)),
        "median_seconds": float(median),
        "max_seconds": float(max(samples)),
        "range_over_median": float(spread / median) if median > 0.0 else None,
    }


def _measure_attempt(
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    repetitions: int,
    elapsed_selector: Callable[[Any, float], float] | None = None,
    order_offset: int = 0,
) -> dict[str, Any]:
    samples = {"reference": [], "candidate": []}
    for index in range(repetitions):
        names = ("reference", "candidate") if (index + order_offset) % 2 == 0 else ("candidate", "reference")
        calls = {"reference": reference, "candidate": candidate}
        for name in names:
            started = time.perf_counter()
            result = calls[name]()
            wall_seconds = time.perf_counter() - started
            measured = elapsed_selector(result, wall_seconds) if elapsed_selector else wall_seconds
            samples[name].append(float(measured))

    stats = {name: _timing_stats(values) for name, values in samples.items()}
    speedup = stats["reference"]["median_seconds"] / stats["candidate"]["median_seconds"]
    return {
        "reference": stats["reference"],
        "candidate": stats["candidate"],
        "median_speedup": float(speedup),
    }


def _measure_metric(
    metric_key: str,
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    elapsed_selector: Callable[[Any, float], float] | None = None,
) -> dict[str, Any]:
    policy = PERFORMANCE_FLOORS[metric_key]
    repetitions = int(policy["repetitions"])
    floor = policy["floor_speedup"]
    attempts = [
        _measure_attempt(
            reference,
            candidate,
            repetitions=repetitions,
            elapsed_selector=elapsed_selector,
        )
    ]
    # A single repeated median guards a marginal miss against one noisy sample
    # set. A failure persists only when the remeasurement also misses the floor.
    if floor is not None and attempts[0]["median_speedup"] < floor:
        attempts.append(
            _measure_attempt(
                reference,
                candidate,
                repetitions=repetitions,
                elapsed_selector=elapsed_selector,
                order_offset=repetitions,
            )
        )
    selected = attempts[-1]
    passed = floor is None or selected["median_speedup"] >= floor
    variability_warning = any(
        side["range_over_median"] is not None and side["range_over_median"] > 1.0
        for attempt in attempts
        for side in (attempt["reference"], attempt["candidate"])
    )
    return {
        "metric_name": policy["metric_name"],
        "reference_implementation": policy["reference"],
        "candidate_implementation": policy["candidate"],
        "repetitions_per_attempt": repetitions,
        "floor_speedup": floor,
        "safety_margin_rationale": policy["safety_margin_rationale"],
        "attempts": attempts,
        "remeasured": len(attempts) > 1,
        "measured_median_speedup": selected["median_speedup"],
        "headroom_over_floor": (
            float(selected["median_speedup"] / floor) if floor is not None else None
        ),
        "variability_warning_range_over_median_gt_1": variability_warning,
        "passed": bool(passed),
    }


def _commit_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _runtime_metadata() -> dict[str, Any]:
    import llvmlite
    import numba
    import scipy

    return {
        "commit_sha": _commit_sha(),
        "python": sys.version,
        "python_version": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_cpu_count": os.cpu_count(),
    }


def _case_controls() -> dict[str, float | int]:
    from ldsfl.inputs import dimensionless_input_table, read_parameter_table

    beta, ds, theta0, flagbed, rpic_0, mdat = dimensionless_input_table(
        read_parameter_table(ROOT / "Input" / "Parameter.csv"), 1,
    )
    return {
        "case": 1,
        "beta": float(beta),
        "ds": float(ds),
        "theta0": float(theta0),
        "flagbed": int(flagbed),
        "rpic_0": float(rpic_0),
        "Mdat": int(mdat),
    }


def _vertical_inputs() -> tuple[float, dict[str, float | int]]:
    from ldsfl.resistance import resistance_function_flagbed

    controls = _case_controls()
    values = resistance_function_flagbed(
        int(controls["flagbed"]), float(controls["theta0"]),
        float(controls["ds"]), float(controls["rpic_0"]),
    )
    return float(values[1]), controls


def _assert_vertical_parity(reference: tuple, candidate: tuple) -> dict[str, float]:
    maximum_absolute = 0.0
    for index in range(4):
        difference = np.abs(np.asarray(candidate[index]) - np.asarray(reference[index]))
        maximum_absolute = max(maximum_absolute, float(np.max(difference)))
        np.testing.assert_allclose(candidate[index], reference[index], rtol=2.0e-11, atol=5.0e-14)
    for index in range(4, 8):
        difference = np.abs(candidate[index] - reference[index])
        maximum_absolute = max(maximum_absolute, float(np.max(difference)))
        np.testing.assert_allclose(candidate[index], reference[index], rtol=3.0e-11, atol=2.0e-13)
    z0_difference = abs(float(candidate[8]) - float(reference[8]))
    np.testing.assert_allclose(candidate[8], reference[8], rtol=5.0e-15, atol=0.0)
    maximum_absolute = max(maximum_absolute, z0_difference)
    return {"max_absolute_difference": maximum_absolute, "z0_absolute_difference": z0_difference}


def _run_vertical(report: dict[str, Any]) -> None:
    from ldsfl.vertical import k0123
    from ldsfl.vertical_numba import _k0123_kernel

    cf0, controls = _vertical_inputs()
    if _k0123_kernel.targetoptions.get("fastmath", False):
        raise AssertionError("vertical Numba kernel unexpectedly enables fastmath")
    candidate_warm = _k0123_kernel(cf0)
    reference_warm = k0123(cf0)
    parity = _assert_vertical_parity(reference_warm, candidate_warm)
    metric = _measure_metric(
        "vertical_k0123",
        lambda: k0123(cf0),
        lambda: _k0123_kernel(cf0),
    )
    metric["correctness"] = {"passed": True, **parity}
    metric["case"] = controls
    metric["Cf0"] = cf0
    metric["candidate_is_raw_compiled_kernel"] = True
    report["metrics"]["vertical_k0123"] = metric


def _flow_args(controls: dict[str, float | int], n_points: int) -> tuple:
    from ldsfl.resistance import resistance_function_flagbed

    rpic, cf0, ct, cd, phi_t, phi_d, f0 = resistance_function_flagbed(
        int(controls["flagbed"]), float(controls["theta0"]), float(controls["ds"]),
        float(controls["rpic_0"]),
    )
    s = np.linspace(0.0, 10.0, n_points, dtype=np.float64)
    c = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    return (
        c, s, cf0, ct, cd, phi_t, phi_d, float(controls["beta"]), rpic,
        float(controls["theta0"]), f0, int(controls["Mdat"]), 1, n_points,
        np.array([1.0], dtype=np.float64), float(s[1] - s[0]),
    )


def _run_flow_with_strategy(args: tuple, strategy: str | None, observed: list[str] | None = None) -> dict[str, Any]:
    from ldsfl import flowfield as flowfield_module

    original = flowfield_module._run_modes_numba

    def observe_and_select(**kwargs):
        if observed is not None:
            observed.append(str(kwargs["strategy"]))
        if strategy is not None:
            kwargs["strategy"] = strategy
        return original(**kwargs)

    flowfield_module._run_modes_numba = observe_and_select
    try:
        timing: dict[str, float] = {}
        output, resonance_flag = flowfield_module.parall_u_free(
            *args,
            SL=0,
            paral=0,
            backend="numba",
            vertical_backend="numba",
            numba_parallel=False,
            numba_fastmath=False,
            timing=timing,
        )
    finally:
        flowfield_module._run_modes_numba = original
    return {"output": output, "resonance_flag": int(resonance_flag), "timing": timing}


def _assert_response_parity(legacy: dict[str, Any], recursive: dict[str, Any]) -> dict[str, Any]:
    if legacy["resonance_flag"] != recursive["resonance_flag"]:
        raise AssertionError("legacy and recursive SEMIANA resonance flags differ")
    difference = np.abs(recursive["output"] - legacy["output"])
    np.testing.assert_allclose(recursive["output"], legacy["output"], rtol=1.0e-9, atol=1.0e-11)
    return {
        "passed": True,
        "resonance_flag": recursive["resonance_flag"],
        "max_absolute_difference": float(np.max(difference)),
        "rms_difference": float(np.sqrt(np.mean(difference**2))),
    }


def _run_semiana(report: dict[str, Any]) -> None:
    from ldsfl.flowfield_numba import _fill_dwstr_recursive_real_nb, _fill_upstr_recursive_real_nb

    controls = _case_controls()
    kernel_options = []
    for kernel in (_fill_upstr_recursive_real_nb, _fill_dwstr_recursive_real_nb):
        options = kernel.targetoptions
        cache_enabled = kernel.stats.cache_path is not None
        if cache_enabled:
            raise AssertionError(f"recursive SEMIANA kernel unexpectedly enables caching: {kernel.stats.cache_path}")
        if options.get("fastmath", False) or options.get("parallel", False):
            raise AssertionError(f"recursive SEMIANA kernel has unsupported options: {options}")
        kernel_options.append({
            "name": kernel.py_func.__name__,
            "fastmath": bool(options.get("fastmath", False)),
            "parallel": bool(options.get("parallel", False)),
            "cache_enabled": cache_enabled,
        })

    for n_points in (1_000, 2_500):
        args = _flow_args(controls, n_points)
        legacy_warm = _run_flow_with_strategy(args, "legacy")
        recursive_warm = _run_flow_with_strategy(args, "recursive")
        parity = _assert_response_parity(legacy_warm, recursive_warm)
        if n_points == 1_000:
            observed: list[str] = []
            public_warm = _run_flow_with_strategy(args, None, observed)
            public_parity = _assert_response_parity(recursive_warm, public_warm)
            if observed != ["recursive"]:
                raise AssertionError(f"public Numba SL0 route was {observed}, expected recursive")
        else:
            public_parity = None

        def legacy_response(flow_args: tuple = args) -> dict[str, Any]:
            return _run_flow_with_strategy(flow_args, "legacy")

        def recursive_response(flow_args: tuple = args) -> dict[str, Any]:
            return _run_flow_with_strategy(flow_args, "recursive")

        metric_key = f"semiana_recursive_n{n_points}"
        metric = _measure_metric(
            metric_key,
            legacy_response,
            recursive_response,
            elapsed_selector=lambda result, _wall: result["timing"]["semiana_response"],
        )
        metric["correctness"] = parity
        metric["N"] = n_points
        metric["Mdat"] = int(controls["Mdat"])
        metric["public_routing"] = (
            {"selected_strategy": "recursive", "correctness": public_parity}
            if n_points == 1_000 else None
        )
        metric["recursive_kernel_options"] = kernel_options
        report["metrics"][metric_key] = metric


def _make_neck_case(n_points: int, *, hit: bool) -> tuple[np.ndarray, np.ndarray, int, float]:
    x = 0.1 * np.arange(n_points, dtype=np.float64)
    y = 0.25 * np.sin(x / 12.0)
    ss, radius = 30, 1.0
    if hit:
        i = (3 * n_points) // 4
        j = i + ss
        x[j] = x[i] + 0.25
        y[j] = y[i] + 0.05
    return x, y, ss, radius


def _run_neck(report: dict[str, Any]) -> None:
    from ldsfl.mathutils import _kdtree_first_hit_point_pair
    from ldsfl.neck_numba import _spatial_grid_first_hit_kernel, spatial_grid_first_hit_point_pair

    options = _spatial_grid_first_hit_kernel.targetoptions
    cache_enabled = _spatial_grid_first_hit_kernel.stats.cache_path is not None
    if cache_enabled or options.get("fastmath", False) or options.get("parallel", False):
        raise AssertionError(f"neck spatial-grid kernel has unsupported options: {options}")
    report["hard_guardrails"]["neck_kernel_options"] = {
        "cache_enabled": cache_enabled,
        "fastmath": bool(options.get("fastmath", False)),
        "parallel": bool(options.get("parallel", False)),
        "passed": True,
    }

    for label, hit in (("no_hit", False), ("late_hit", True)):
        x, y, ss, radius = _make_neck_case(2_048, hit=hit)
        reference_warm = _kdtree_first_hit_point_pair(x, y, ss, radius)
        candidate_warm = spatial_grid_first_hit_point_pair(x, y, ss, radius)
        if hit and (reference_warm is None or reference_warm[0] < (3 * x.size) // 4 - ss):
            raise AssertionError(f"late-hit geometry did not produce a representative later hit: {reference_warm}")
        if not hit and reference_warm is not None:
            raise AssertionError(f"no-hit geometry unexpectedly returned {reference_warm}")
        if reference_warm != candidate_warm:
            raise AssertionError(
                f"{label} detector mismatch: reference={reference_warm}, candidate={candidate_warm}, hit_case={hit}"
            )
        metric_key = f"neck_grid_{label}"
        metric = _measure_metric(
            metric_key,
            lambda xa=x, ya=y, separation=ss, threshold=radius: _kdtree_first_hit_point_pair(
                xa, ya, separation, threshold,
            ),
            lambda xa=x, ya=y, separation=ss, threshold=radius: spatial_grid_first_hit_point_pair(
                xa, ya, separation, threshold,
            ),
        )
        metric["correctness"] = {
            "passed": True,
            "expected_pair": reference_warm,
            "N": int(x.size),
            "geometry": "smooth uniformized centerline with an isolated later near-approach" if hit else "smooth uniformized no-hit centerline",
        }
        metric["N"] = int(x.size)
        report["metrics"][metric_key] = metric


def _run_geometry_unwrap(report: dict[str, Any]) -> None:
    from ldsfl.geometry_numba import unwrap_angles_like_matlab_numba
    from ldsfl.inputs import read_xy
    from ldsfl.mathutils import matlab_gradient, unwrap_angles_like_matlab

    x, y = read_xy(ROOT / "Input" / "xy.csv")
    theta = np.ascontiguousarray(np.arctan2(matlab_gradient(y), matlab_gradient(x)), dtype=np.float64)
    reference_warm = unwrap_angles_like_matlab(theta)
    candidate_warm = unwrap_angles_like_matlab_numba(theta)
    if not np.array_equal(reference_warm, candidate_warm, equal_nan=True):
        raise AssertionError("Numba geometry unwrap differs from the exact Python reference")
    options = unwrap_angles_like_matlab_numba.targetoptions
    if options.get("fastmath", False) or options.get("parallel", False):
        raise AssertionError(f"geometry unwrap kernel has unsupported options: {options}")
    report["hard_guardrails"]["geometry_unwrap_kernel_options"] = {
        "fastmath": bool(options.get("fastmath", False)),
        "parallel": bool(options.get("parallel", False)),
        "passed": True,
    }

    def reference_batch() -> None:
        for _ in range(UNWRAP_CALLS_PER_SAMPLE):
            unwrap_angles_like_matlab(theta)

    def candidate_batch() -> None:
        for _ in range(UNWRAP_CALLS_PER_SAMPLE):
            unwrap_angles_like_matlab_numba(theta)

    metric = _measure_metric(
        "geometry_unwrap",
        reference_batch,
        candidate_batch,
        elapsed_selector=lambda _result, wall: wall / UNWRAP_CALLS_PER_SAMPLE,
    )
    metric["correctness"] = {
        "passed": True,
        "exact": True,
        "array_length": int(theta.size),
        "calls_per_timed_sample": UNWRAP_CALLS_PER_SAMPLE,
    }
    metric["timing_gated"] = False
    metric["N"] = int(theta.size)
    report["metrics"]["geometry_unwrap"] = metric


def _new_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "accepted exact-backend CI performance guardrails",
        "method": {
            "warmup": "all Numba operations are called and parity-checked before timing",
            "repetitions": REPETITIONS,
            "measurement": "alternating reference/candidate call order; median is the primary statistic",
            "variability": "range divided by median, reported for each implementation and attempt",
            "threadpools": "limited to one thread where threadpoolctl is available",
            "absolute_wall_time_gate": False,
            "whole_solver_benchmark": False,
        },
        "environment": _runtime_metadata(),
        "threshold_policy": PERFORMANCE_FLOORS,
        "hard_guardrails": {},
        "metrics": {},
    }


def _print_report(report: dict[str, Any]) -> None:
    print("\nPerformance guardrail results (warmed, paired medians)")
    print(f"{'Metric':34} {'ref ms':>11} {'candidate ms':>14} {'speedup':>10} {'floor':>8} {'status':>8}")
    for key, metric in report["metrics"].items():
        selected = metric["attempts"][-1]
        reference_ms = selected["reference"]["median_seconds"] * 1_000.0
        candidate_ms = selected["candidate"]["median_seconds"] * 1_000.0
        floor = metric["floor_speedup"]
        floor_text = "correctness" if floor is None else f"{floor:.1f}x"
        status = "PASS" if metric["passed"] else "FAIL"
        print(
            f"{metric['metric_name']:34} {reference_ms:11.4f} {candidate_ms:14.4f} "
            f"{metric['measured_median_speedup']:9.2f}x {floor_text:>8} {status:>8}"
        )
        if metric["remeasured"]:
            print(f"  {key}: remeasured after initial median missed its floor")
        if metric["variability_warning_range_over_median_gt_1"]:
            print(f"  {key}: timing variability warning (range/median > 1)")
    print(f"Total benchmark runtime: {report.get('wall_seconds', 0.0):.2f} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path (the destination directory is created if needed)",
    )
    args = parser.parse_args(argv)

    report = _new_report()
    started = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            _run_vertical(report)
            _run_semiana(report)
            _run_neck(report)
            _run_geometry_unwrap(report)
    except Exception as exc:  # Preserve partial diagnostic data for CI artifacts.
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["passed"] = False
    report["wall_seconds"] = float(time.perf_counter() - started)
    metric_pass = all(metric["passed"] for metric in report["metrics"].values())
    report["passed"] = bool(report.get("passed", True) and metric_pass and "error" not in report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Wrote {args.output}")
    _print_report(report)
    if "error" in report:
        print(f"Guardrail error: {report['error']}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
