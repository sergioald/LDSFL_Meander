"""Audit warmed SEMIANA response scaling without long solver simulations.

Run from the repository root. Results go under the ignored benchmarks/Output
tree. ``legacy`` and ``recursive`` explicitly select the corresponding private
Numba response, independent of the public backend's default strategy.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldsfl import flowfield as flowfield_module  # noqa: E402
from ldsfl.inputs import dimensionless_input_table, read_parameter_table  # noqa: E402
from ldsfl.modes import _precompute_modes  # noqa: E402
from ldsfl.resistance import resistance_function_flagbed  # noqa: E402

TOLL = 1.0e-4
SIZES = (51, 250, 1000, 2500, 5000)
SCENARIOS = (("SL0", 0, 9.0), ("SL1_sub", 1, 6.0), ("SL1_super", 1, 12.0))
FIELDS = ("semiana_response", "flowfield_total", "modal_coefficients", "vertical_coefficients")


def _stats(samples: list[float]) -> dict[str, float]:
    return {
        "min_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "max_seconds": max(samples),
    }


def _case_controls() -> dict[str, float | int]:
    beta, ds, theta0, flagbed, rpic_0, mdat = dimensionless_input_table(
        read_parameter_table(ROOT / "Input" / "Parameter.csv"), 1,
    )
    return {
        "case": 1, "case_beta": float(beta), "ds": float(ds), "theta0": float(theta0),
        "flagbed": int(flagbed), "rpic_0": float(rpic_0), "Mdat": int(mdat),
    }


def _flow_args(controls: dict[str, float | int], n_points: int, beta: float) -> tuple:
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(
        int(controls["flagbed"]), float(controls["theta0"]), float(controls["ds"]),
        float(controls["rpic_0"]),
    )
    s = np.linspace(0.0, 10.0, n_points, dtype=np.float64)
    c = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    return (
        c, s, cf0, ct, cd, phit, phid, beta, rpic, float(controls["theta0"]),
        f0, int(controls["Mdat"]), 1, n_points, np.array([1.0]), float(s[1] - s[0]),
    )


def _jtoll_audit(flow_args: tuple, sl: int) -> dict:
    (_c, s, cf0, ct, cd, phit, phid, beta, rpic, theta0, f0, mdat,
     _nn, n_points, _n, deltas) = flow_args
    modes = _precompute_modes(
        cf0, ct, cd, phit, phid, beta, rpic, theta0, f0, mdat,
        backend="numba",
    )
    eigenvalues = modes[1:5]
    svec = s
    limit_scale = -np.log(TOLL)
    records = []
    for mode in range(mdat):
        sub_resonant = float(eigenvalues[1][mode].real) < 0.0
        for eigen_index, eigen_column in enumerate(eigenvalues, start=1):
            lam = complex(eigen_column[mode])
            if sl == 0:
                direction = "upstream" if lam.real > 0.0 else "downstream" if lam.real < 0.0 else "none"
            elif sub_resonant:
                direction = "upstream" if eigen_index == 1 else "downstream"
            else:
                direction = "downstream" if eigen_index == 4 else "upstream"
            if lam.real == 0.0:
                jtoll = n_points + 1
            else:
                distances = svec - svec[0] if sl == 0 else svec
                jtoll = 1 + int(np.searchsorted(distances, limit_scale / abs(lam.real), side="left"))
            records.append({
                "mode": mode + 1, "eigenvalue": eigen_index, "lambda_real": lam.real,
                "lambda_imag": lam.imag, "direction": direction, "jtoll": jtoll,
                "effective_window": min(jtoll, n_points),
            })
    jtolls = [record["jtoll"] for record in records]
    return {
        "min": min(jtolls), "median": statistics.median(jtolls), "max": max(jtolls),
        "per_eigenvalue_mode": records,
    }


def _measure(flow_args: tuple, sl: int, strategy: str, repeats: int) -> dict:
    original = flowfield_module._run_modes_numba

    def selected_response(**kwargs):
        kwargs["strategy"] = strategy
        return original(**kwargs)

    flowfield_module._run_modes_numba = selected_response
    try:
        def call() -> tuple[dict[str, float], int]:
            timing: dict[str, float] = {}
            _u, flag = flowfield_module.parall_u_free(
                *flow_args, SL=sl, paral=0, backend="numba", vertical_backend="numba",
                numba_parallel=False, numba_fastmath=False, timing=timing,
            )
            return timing, flag

        call()  # Compile and warm all selected kernels outside timed samples.
        samples = [call() for _ in range(repeats)]
    finally:
        flowfield_module._run_modes_numba = original
    return {
        "resonance_flag": samples[0][1],
        "timings": {field: _stats([sample[0][field] for sample in samples]) for field in FIELDS},
    }


def _flow_output(flow_args: tuple, sl: int, backend: str, strategy: str) -> tuple[np.ndarray, int]:
    original = flowfield_module._run_modes_numba

    def select_strategy(**kwargs):
        kwargs["strategy"] = strategy
        return original(**kwargs)

    if backend == "numba":
        flowfield_module._run_modes_numba = select_strategy
    try:
        return flowfield_module.parall_u_free(
            *flow_args, SL=sl, paral=0, backend=backend, vertical_backend="numba",
            numba_parallel=False, numba_fastmath=False,
        )
    finally:
        flowfield_module._run_modes_numba = original


def _errors(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.abs(candidate - reference)
    return {
        "max_absolute": float(np.max(difference)),
        "max_relative_with_1e-12_denominator_floor": float(np.max(difference / np.maximum(np.abs(reference), 1.0e-12))),
        "rms_absolute": float(np.sqrt(np.mean(difference**2))),
    }


def _parity(flow_args: tuple, sl: int, n_points: int, numpy_max_n: int) -> dict:
    legacy, legacy_flag = _flow_output(flow_args, sl, "numba", "legacy")
    recursive, recursive_flag = _flow_output(flow_args, sl, "numba", "recursive")
    result = {
        "legacy_flag": legacy_flag, "recursive_flag": recursive_flag,
        "legacy_vs_recursive": _errors(recursive, legacy),
    }
    if n_points <= numpy_max_n:
        reference, reference_flag = _flow_output(flow_args, sl, "numpy", "legacy")
        result.update({
            "numpy_flag": reference_flag,
            "numpy_vs_legacy": _errors(legacy, reference),
            "numpy_vs_recursive": _errors(recursive, reference),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("legacy", "recursive", "both"), default="legacy")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--parity", action="store_true", help="also compare legacy, recursive and NumPy outputs")
    parser.add_argument("--numpy-parity-max-n", type=int, default=1000)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if not sizes or any(value < 3 for value in sizes):
        parser.error("--sizes must contain values of at least 3")

    controls = _case_controls()
    strategies = ("legacy", "recursive") if args.strategy == "both" else (args.strategy,)
    rows = []
    with threadpool_limits(limits=1):
        for scenario, sl, beta in SCENARIOS:
            for n_points in sizes:
                flow_args = _flow_args(controls, n_points, beta)
                jtoll = _jtoll_audit(flow_args, sl)
                for strategy in strategies:
                    measured = _measure(flow_args, sl, strategy, args.repetitions)
                    row = {
                        "scenario": scenario, "N": n_points, "Mdat": controls["Mdat"],
                        "SL": sl, "beta": beta, "deltas": flow_args[-1], "backend": "numba",
                        "vertical_backend": "numba", "strategy": strategy,
                        "resonance_flag": measured["resonance_flag"], "jtoll": jtoll,
                        "timings": measured["timings"],
                    }
                    if args.parity and strategy == "recursive":
                        row["parity"] = _parity(flow_args, sl, n_points, args.numpy_parity_max_n)
                    rows.append(row)
                    response_ms = 1.0e3 * row["timings"]["semiana_response"]["median_seconds"]
                    print(f"{scenario:10} N={n_points:5} {strategy:9} response median={response_ms:10.3f} ms jtoll={jtoll['min']}/{jtoll['median']}/{jtoll['max']}", flush=True)

    result = {
        "method": "one untimed warm-up per scenario and N; then steady-state timings",
        "strategy": args.strategy, "repetitions": args.repetitions,
        "controls": controls, "TOLL": TOLL, "numba_parallel": False,
        "numba_fastmath": False, "paral": 0, "synthetic_s_range": [0.0, 10.0],
        "curvature": "1e-3 * sin(2*pi*s/10)", "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "rows": rows,
    }
    output = args.output or (
        ROOT / "benchmarks" / "Output" / "semiana_hotpath" /
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.strategy}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
