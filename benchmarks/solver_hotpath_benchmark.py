"""Cache-audited k0123 and flowfield benchmark.

Run from the repository root with ``python benchmarks/solver_hotpath_benchmark.py``.
The script separates direct Python execution, raw compiled Numba execution
after an untimed JIT warm-up, and exact-key wrapper misses and hits. It also
benchmarks identical and changing Cf0 flow calls plus a short run_case using
both backends.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from shutil import copy2

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldsfl import main as main_module  # noqa: E402
from ldsfl.flowfield import parall_u_free  # noqa: E402
from ldsfl.inputs import dimensionless_input_table, read_parameter_table, read_xy  # noqa: E402
from ldsfl.profile import preprof_3  # noqa: E402
from ldsfl.resistance import resistance_function_flagbed  # noqa: E402
from ldsfl.vertical import (  # noqa: E402
    clear_k0123_reference_cache,
    k0123,
    k0123_reference_cache_info,
)
from ldsfl.vertical_numba import (  # noqa: E402
    _k0123_kernel,
    clear_k0123_cache,
    k0123_cache_info,
    k0123_numba_cached,
)


def _case1():
    beta, ds, theta0, flagbed, rpic_0, mdat = dimensionless_input_table(
        read_parameter_table(ROOT / "Input" / "Parameter.csv"), 1
    )
    x_raw, y_raw = read_xy(ROOT / "Input" / "xy.csv")
    s, _x, _y, _theta, ns, deltas, *_ = preprof_3(x_raw, y_raw, 1.0)
    rpic, cf0, ct, cd, phi_t, phi_d, f0 = resistance_function_flagbed(
        flagbed, theta0, ds, rpic_0
    )
    c = 1e-3 * np.sin(2.0 * np.pi * s / s[-1])
    flow_args = (
        c,
        s,
        cf0,
        ct,
        cd,
        phi_t,
        phi_d,
        beta,
        rpic,
        theta0,
        f0,
        mdat,
        1,
        ns,
        np.array([1.0]),
        deltas,
    )
    return cf0, flow_args


def _cache_snapshot(info) -> dict[str, int | None]:
    return {
        "hits": info.hits,
        "misses": info.misses,
        "currsize": info.currsize,
        "maxsize": info.maxsize,
    }


def _timings(call, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return samples


def _summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "repetitions": len(samples),
        "min_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "max_seconds": max(samples),
    }


def _cache_delta(before, after) -> dict[str, int]:
    return {"hits": after.hits - before.hits, "misses": after.misses - before.misses}


def _clear_cache(backend: str) -> None:
    if backend == "numba":
        clear_k0123_cache()
    else:
        clear_k0123_reference_cache()


def _cache_info(backend: str):
    return k0123_cache_info() if backend == "numba" else k0123_reference_cache_info()


def _flow_call(
    backend: str,
    flow_args: tuple,
    *,
    cf0: float | None = None,
    vertical_backend: str | None = None,
):
    call_args = flow_args if cf0 is None else (*flow_args[:2], cf0, *flow_args[3:])
    timing: dict[str, float] = {}
    start = time.perf_counter()
    parall_u_free(
        *call_args,
        SL=0,
        paral=0,
        backend=backend,
        vertical_backend=vertical_backend,
        timing=timing,
    )
    return time.perf_counter() - start, timing


def _component_medians(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("vertical_coefficients", "modal_coefficients", "semiana_response", "flowfield_total")
    return {key: statistics.median(row[key] for row in rows) for key in keys}


def _run_case(backend: str, cf0: float, flow_args: tuple, steps: int) -> dict:
    # Compile/load selected kernels before timing, then start the exact-key LRU empty.
    _flow_call(backend, flow_args)
    _clear_cache(backend)
    cache_during_flow = {"hits": 0, "misses": 0}
    cf0_calls: list[float] = []
    original_flow = main_module.parall_u_free

    def counted_flow(*args, **kwargs):
        before = _cache_info(backend)
        cf0_calls.append(float(args[2]))
        result = original_flow(*args, **kwargs)
        after = _cache_info(backend)
        delta = _cache_delta(before, after)
        cache_during_flow["hits"] += delta["hits"]
        cache_during_flow["misses"] += delta["misses"]
        return result

    with tempfile.TemporaryDirectory(prefix="solver_hotpath_benchmark_") as temp_dir:
        base_dir = Path(temp_dir)
        (base_dir / "Input").mkdir()
        copy2(ROOT / "Input" / "Parameter.csv", base_dir / "Input" / "Parameter.csv")
        copy2(ROOT / "Input" / "xy.csv", base_dir / "Input" / "xy.csv")
        main_module.parall_u_free = counted_flow
        try:
            start = time.perf_counter()
            result = main_module.run_case(
                base_dir,
                case_i=1,
                max_steps=steps,
                stop_on_steps=True,
                stop_on_time=False,
                stop_on_cutoffs=True,
                flow_backend=backend,
                do_plots=False,
                collect_timing=True,
            )
            wall_seconds = time.perf_counter() - start
        finally:
            main_module.parall_u_free = original_flow

    return {
        "backend": backend,
        "controls": {
            "case_i": 1,
            "max_steps": steps,
            "ER": 1e-8,
            "flow_paral": 0,
            "numba_parallel": False,
            "numba_fastmath": False,
            "do_plots": False,
        },
        "total_wall_seconds": wall_seconds,
        "flowfield_loop_seconds": result["timings"]["flowfield_loop"],
        "flowfield_final_seconds": result["timings"]["flowfield_final"],
        "flowfield_total_seconds": result["timings"]["flowfield_total"],
        "historical_flowfield_seconds": result["timings"]["flowfield"],
        "vertical_coefficients_loop_seconds": result["timings"]["vertical_coefficients"],
        "completed_migration_steps": result["steps"],
        "flowfield_calls_including_final_snapshot": len(cf0_calls),
        "flowfield_cf0_values_exact": cf0_calls,
        "distinct_exact_cf0_count": len(set(cf0_calls)),
        "k0123_cache_delta_during_flowfield_calls": cache_during_flow,
    }


def run_benchmark(repeats: int, flow_repeats: int, run_case_steps: int) -> dict:
    cf0, flow_args = _case1()
    vertical: dict[str, dict] = {}

    clear_k0123_reference_cache()
    clear_k0123_cache()
    reference_before = k0123_reference_cache_info()
    numba_before = k0123_cache_info()
    a_samples = _timings(lambda: k0123(cf0), repeats)
    vertical["python_reference_uncached"] = {
        "function_timed": "ldsfl.vertical.k0123(Cf0), called directly",
        "timing": _summary(a_samples),
        "reference_cache_before": _cache_snapshot(reference_before),
        "reference_cache_after": _cache_snapshot(k0123_reference_cache_info()),
        "numba_cache_before": _cache_snapshot(numba_before),
        "numba_cache_after": _cache_snapshot(k0123_cache_info()),
    }

    warmup_start = time.perf_counter()
    _k0123_kernel(cf0)
    warmup_seconds = time.perf_counter() - warmup_start
    b_before = k0123_cache_info()
    b_samples = _timings(lambda: _k0123_kernel(cf0), repeats)
    b_after = k0123_cache_info()
    vertical["numba_raw_kernel_uncached_after_jit_warmup"] = {
        "function_timed": "ldsfl.vertical_numba._k0123_kernel(Cf0), raw compiled dispatcher bypassing the exact-key LRU",
        "untimed_jit_or_persistent_cache_warmup_seconds": warmup_seconds,
        "compiled_signatures": len(_k0123_kernel.signatures),
        "timing": _summary(b_samples),
        "wrapper_cache_before": _cache_snapshot(b_before),
        "wrapper_cache_after": _cache_snapshot(b_after),
    }

    c_samples: list[float] = []
    c_deltas = {"hits": 0, "misses": 0}
    for _ in range(repeats):
        clear_k0123_cache()  # Intentionally outside the timed region.
        before = k0123_cache_info()
        start = time.perf_counter()
        k0123_numba_cached(cf0)
        c_samples.append(time.perf_counter() - start)
        after = k0123_cache_info()
        delta = _cache_delta(before, after)
        c_deltas["hits"] += delta["hits"]
        c_deltas["misses"] += delta["misses"]
    vertical["numba_public_wrapper_exact_key_miss"] = {
        "function_timed": "ldsfl.vertical_numba.k0123_numba_cached(Cf0), cache cleared outside each timed call",
        "timing": _summary(c_samples),
        "summed_cache_delta_across_calls": c_deltas,
        "cache_info_after_final_miss": _cache_snapshot(k0123_cache_info()),
    }

    clear_k0123_cache()
    k0123_numba_cached(cf0)  # Prime exactly this float key outside the timed region.
    d_before = k0123_cache_info()
    d_samples = _timings(lambda: k0123_numba_cached(cf0), repeats)
    d_after = k0123_cache_info()
    vertical["numba_public_wrapper_exact_key_hit"] = {
        "function_timed": "ldsfl.vertical_numba.k0123_numba_cached(Cf0), same exact float key primed before timing",
        "timing": _summary(d_samples),
        "cache_info_before_timed_calls": _cache_snapshot(d_before),
        "cache_info_after_timed_calls": _cache_snapshot(d_after),
        "timed_cache_delta": _cache_delta(d_before, d_after),
    }

    flowfield: dict[str, dict] = {"repeated_identical_cf0": {}, "distinct_exact_cf0": {}}
    with threadpool_limits(limits=1):
        flow_paths = (
            ("numpy", "numpy"),
            ("numba", "numpy"),
            ("numba", "numba"),
        )
        for backend, vertical_backend in flow_paths:
            path_name = f"{backend}_response_{vertical_backend}_vertical"
            # Warm selected flow backend before measuring, then clear only vertical cache.
            _flow_call(backend, flow_args, vertical_backend=vertical_backend)
            _clear_cache(vertical_backend)
            before = _cache_info(vertical_backend)
            same_samples: list[float] = []
            same_rows: list[dict[str, float]] = []
            for _ in range(flow_repeats):
                elapsed, row = _flow_call(
                    backend, flow_args, vertical_backend=vertical_backend
                )
                same_samples.append(elapsed)
                same_rows.append(row)
            after = _cache_info(vertical_backend)
            flowfield["repeated_identical_cf0"][path_name] = {
                "response_backend": backend,
                "vertical_backend": vertical_backend,
                "timing": _summary(same_samples),
                "median_components_seconds": _component_medians(same_rows),
                "cache_info_before": _cache_snapshot(before),
                "cache_info_after": _cache_snapshot(after),
                "cache_delta": _cache_delta(before, after),
            }

            cf0_values = [float(cf0 * (1.0 + (i + 1) * 1e-9)) for i in range(flow_repeats)]
            if len(set(cf0_values)) != flow_repeats or not all(value > 0.0 for value in cf0_values):
                raise ValueError("Varying-Cf0 benchmark values must be distinct and positive.")
            _clear_cache(vertical_backend)
            before = _cache_info(vertical_backend)
            changing_samples: list[float] = []
            changing_rows: list[dict[str, float]] = []
            for value in cf0_values:
                elapsed, row = _flow_call(
                    backend,
                    flow_args,
                    cf0=value,
                    vertical_backend=vertical_backend,
                )
                changing_samples.append(elapsed)
                changing_rows.append(row)
            after = _cache_info(vertical_backend)
            flowfield["distinct_exact_cf0"][path_name] = {
                "response_backend": backend,
                "vertical_backend": vertical_backend,
                "cf0_values_exact": cf0_values,
                "distinct_exact_values": len(set(cf0_values)),
                "timing": _summary(changing_samples),
                "median_components_seconds": _component_medians(changing_rows),
                "cache_info_before": _cache_snapshot(before),
                "cache_info_after": _cache_snapshot(after),
                "cache_delta": _cache_delta(before, after),
            }

        run_case = {
            backend: _run_case(backend, cf0, flow_args, run_case_steps)
            for backend in ("numpy", "numba")
        }

    return {
        "case": "Case 1 inputs; deterministic sinusoidal curvature for direct flowfield scenarios",
        "cf0_base_exact": cf0,
        "vertical_repetitions": repeats,
        "flowfield_repetitions": flow_repeats,
        "numba_fastmath": False,
        "vertical_measurements": vertical,
        "flowfield_scenarios": flowfield,
        "run_case_same_controls": run_case,
        "timing_note": (
            "A/B/C/D are separate direct-call measurements. B excludes its untimed JIT or persistent-cache warm-up. "
            "C clears the LRU outside each timed call; D times only exact-key hits. Flowfield kernels are warmed "
            "before comparison, while the exact vertical LRU is cleared per scenario. run_case records both its "
            "in-loop solves and the final-snapshot recomputation. Timings are machine- and load-dependent."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30, help="repetitions for each direct k0123 measurement")
    parser.add_argument("--flow-repeats", type=int, default=12, help="repetitions/unique values for flowfield scenarios")
    parser.add_argument("--run-case-steps", type=int, default=5, help="completed migration steps in run_case scenario")
    parser.add_argument("--output", type=Path, help="optional path for the JSON report")
    args = parser.parse_args()
    if min(args.repeats, args.flow_repeats, args.run_case_steps) <= 0:
        parser.error("repetition counts and run-case steps must be positive")

    report = json.dumps(run_benchmark(args.repeats, args.flow_repeats, args.run_case_steps), indent=2)
    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
