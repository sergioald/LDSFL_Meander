"""Profile geometry costs and run the requested evolving Numba baseline.

Run the inexpensive synthetic size sweep with::

    python benchmarks/geometry_hotpath_benchmark.py --mode microbench

Run the single 5,000-step case-1 Numba baseline plus the sweep with::

    python benchmarks/geometry_hotpath_benchmark.py --mode all --steps 5000

Generated artifacts are written beneath the ignored ``benchmarks/Output`` tree.
The synthetic geometries are smooth monotone-x meanders; their increasing point
counts isolate geometry scaling and are not presented as model trajectories.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import math
import pstats
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import long_term_backend_parity as parity  # noqa: E402

from ldsfl import mathutils as mathutils_module  # noqa: E402
from ldsfl.geometry import geometry4  # noqa: E402
from ldsfl.mathutils import (  # noqa: E402
    find_neck_cutoff_kdtree_with_refine,
    matlab_gradient,
    matlab_spline,
    unwrap_angles_like_matlab,
)

DEFAULT_OUTPUT = ROOT / "benchmarks" / "Output" / "geometry_hotpath"
SIZES = (250, 1_000, 2_500, 5_000)
GEOMETRY_TIMING_KEYS = (
    "geometry_arclength",
    "geometry_initial_uniformization",
    "geometry_neck",
    "geometry_smoothing",
    "geometry_resample",
    "geometry_curvature",
    "geometry_diagnostics",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2), encoding="utf-8")


def _arclength(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)
    return np.concatenate(([0.0], np.cumsum(np.sqrt(dx**2 + dy**2))))


def _synthetic_centerline(size: int) -> tuple[np.ndarray, np.ndarray]:
    x = 0.9 * np.arange(size, dtype=np.float64)
    y = 1.1 * np.sin(x / 35.0) + 0.25 * np.sin(x / 8.0)
    return x, y


def _initial_uniformization(x: np.ndarray, y: np.ndarray, dsliminicial: float = 1.0):
    """Benchmark-only copy of geometry4's initial uniformization sequence."""
    sa = _arclength(x, y)
    nnew = int(1 + round(sa[-1] / dsliminicial))
    if nnew > 4 * x.size:
        raise RuntimeError("synthetic input exceeded geometry4's length guard")
    query = np.linspace(sa[0], sa[-1], x.size)
    x = matlab_spline(sa, x, query)
    y = matlab_spline(sa, y, query)
    sa = _arclength(x, y)
    return x, y, sa


def _curvature_reconstruction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sa = _arclength(x, y)
    deltas = float(sa[-1] / (sa.size - 1))
    dx = matlab_gradient(x)
    dy = matlab_gradient(y)
    theta = unwrap_angles_like_matlab(np.arctan2(dy, dx))
    theta = -1.0 * theta
    return matlab_gradient(theta) / deltas


def _geometry_case(x: np.ndarray, y: np.ndarray, *, cutoff: bool) -> Callable[[], dict[str, Any]]:
    def call() -> dict[str, Any]:
        timing: dict[str, Any] = {}
        result = geometry4(
            x.copy(),
            y.copy(),
            jt=3 if cutoff else 1,
            dsliminicial=1.0,
            id_files="geometry-hotpath-benchmark",
            Ntstep=3,
            cut_cnt=0,
            beta=9.0,
            base_out=DEFAULT_OUTPUT,
            neck_cutoff_interval=3,
            smoothing_enabled=True,
            timing=timing,
            do_plots=False,
        )
        return {
            "timing": timing,
            "Ns": int(result[5]),
            "cut_cnt": int(result[-1]),
        }

    return call


def _benchmark_callable(
    *,
    size: int,
    name: str,
    call_factory: Callable[[], Callable[[], Any]],
    repetitions: int,
) -> dict[str, Any]:
    warm = call_factory()
    warm_result = warm()
    elapsed_samples = []
    component_samples: dict[str, list[float]] = {}
    count_samples: dict[str, list[int]] = {}
    for _ in range(repetitions):
        call = call_factory()
        started = time.perf_counter()
        result = call()
        elapsed_samples.append(time.perf_counter() - started)
        if isinstance(result, dict):
            for key, value in (result.get("timing") or {}).items():
                if key in GEOMETRY_TIMING_KEYS:
                    component_samples.setdefault(key, []).append(float(value))
                elif key in (
                    "geometry_calls", "geometry_smoothing_calls", "geometry_resamples",
                    "geometry_neck_searches", "geometry_cutoff_events",
                ):
                    count_samples.setdefault(key, []).append(int(value))
    row: dict[str, Any] = {
        "N_input": size,
        "workload": name,
        "repetitions": repetitions,
        "wall_seconds_min": min(elapsed_samples),
        "wall_seconds_median": statistics.median(elapsed_samples),
        "wall_seconds_max": max(elapsed_samples),
    }
    if isinstance(warm_result, dict):
        row["output_Ns"] = warm_result.get("Ns")
        row["cut_cnt"] = warm_result.get("cut_cnt")
    for key, values in component_samples.items():
        row[f"{key}_median_seconds"] = statistics.median(values)
    for key, values in count_samples.items():
        row[key] = values[-1]
    return row


def _profile_geometry(size: int, output_dir: Path) -> dict[str, Any]:
    x, y = _synthetic_centerline(size)
    call = _geometry_case(x, y, cutoff=True)
    call()  # Warm SciPy and Python paths outside the profile.
    tree_profile = {
        "construction_seconds": 0.0,
        "query_ball_point_seconds": 0.0,
        "query_ball_point_calls": 0,
    }
    original_tree = mathutils_module.cKDTree

    class TimedKDTree:
        def __init__(self, *args, **kwargs):
            started = time.perf_counter()
            self._tree = original_tree(*args, **kwargs)
            tree_profile["construction_seconds"] += time.perf_counter() - started

        def query_ball_point(self, *args, **kwargs):
            started = time.perf_counter()
            result = self._tree.query_ball_point(*args, **kwargs)
            tree_profile["query_ball_point_seconds"] += time.perf_counter() - started
            tree_profile["query_ball_point_calls"] += 1
            return result

    profile = cProfile.Profile()
    mathutils_module.cKDTree = TimedKDTree
    try:
        profile.runcall(call)
    finally:
        mathutils_module.cKDTree = original_tree
    stats = pstats.Stats(profile).strip_dirs().sort_stats("cumulative")
    stream = io.StringIO()
    stats.stream = stream
    stats.print_stats(50)
    (output_dir / "cprofile_geometry4.txt").write_text(stream.getvalue(), encoding="utf-8")

    focused = []
    watch = (
        "geometry4", "matlab_spline", "CubicSpline", "_recompute_sa", "find_neck_cutoff",
        "_kdtree_first_hit_point_pair", "query_ball_point", "smooth_xy_via_theta", "butter",
        "sosfiltfilt", "unwrap_angles_like_matlab", "matlab_gradient", "gradient", "dxdy2",
    )
    for (filename, line, function), (primitive_calls, calls, own, cumulative, _callers) in stats.stats.items():
        if any(term in function for term in watch):
            focused.append({
                "function": f"{filename}:{line}({function})",
                "primitive_calls": primitive_calls,
                "calls": calls,
                "self_seconds": own,
                "cumulative_seconds": cumulative,
            })
    focused.sort(key=lambda item: item["cumulative_seconds"], reverse=True)
    _write_json(output_dir / "cprofile_geometry4.json", focused)
    return {
        "profiled_N": size,
        "profiled_workload": "geometry4 with cutoff search enabled and no expected cutoff",
        "kdtree_native_timing": tree_profile,
        "top_focused_functions": focused[:20],
        "text_path": "cprofile_geometry4.txt",
    }


def _run_scaling(repetitions: int, output_dir: Path) -> dict[str, Any]:
    rows = []
    for size in SIZES:
        x, y = _synthetic_centerline(size)
        tasks = (
            ("geometry4_non_cutoff_step", lambda x=x, y=y: _geometry_case(x, y, cutoff=False)),
            ("geometry4_cutoff_search_no_hit", lambda x=x, y=y: _geometry_case(x, y, cutoff=True)),
            (
                "cutoff_detector_direct_no_hit",
                lambda x=x, y=y: lambda: find_neck_cutoff_kdtree_with_refine(x, y, 18, 9.0),
            ),
            (
                "initial_uniformization_and_xy_splines",
                lambda x=x, y=y: lambda: _initial_uniformization(x, y),
            ),
            (
                "curvature_and_tangent_reconstruction",
                lambda x=x, y=y: lambda: _curvature_reconstruction(x, y),
            ),
        )
        for name, factory in tasks:
            row = _benchmark_callable(
                size=size,
                name=name,
                call_factory=factory,
                repetitions=repetitions,
            )
            if name == "geometry4_cutoff_search_no_hit" and row.get("cut_cnt") != 0:
                raise RuntimeError(f"synthetic cutoff workload unexpectedly cut points at N={size}")
            rows.append(row)
        print(f"[geometry sweep] completed N={size}", flush=True)

    with (output_dir / "geometry_scaling.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output_dir / "geometry_scaling.json", rows)
    profile = _profile_geometry(5_000, output_dir)
    _write_json(output_dir / "cprofile_geometry4_summary.json", profile)
    return {"rows": rows, "profile": profile}


def _run_baseline(case: int, steps: int, output_dir: Path) -> dict[str, Any]:
    try:
        import numba  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"Numba is required for the baseline: {exc}") from exc
    checkpoints = {0, 1, 10, 100, 500, 1_000, 2_500, 5_000}
    checkpoints = {step for step in checkpoints if step <= steps}
    run = parity._run_one(
        label="numba_production_default_baseline",
        case=case,
        steps=steps,
        checkpoints=checkpoints,
        flow_backend="numba",
        vertical_backend="numba",
        progress_every=1_000,
    )
    serialized = parity._serial_run(run)
    _write_json(output_dir / "baseline_run.json", serialized)
    checkpoint_rows = []
    for _step, snapshot in sorted(run["_checkpoints"].items()):
        checkpoint_rows.append({key: value for key, value in snapshot.items() if key != "_arrays"})
    if checkpoint_rows:
        with (output_dir / "baseline_checkpoints.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(checkpoint_rows[0]))
            writer.writeheader()
            writer.writerows(checkpoint_rows)
    print(
        f"[baseline] completed {run['completed_steps']:,}/{steps:,} steps; "
        f"Ns={run['final_Ns']}; cutoffs={run['number_of_cutoffs']}; "
        f"wall={run['total_wall_seconds']:.3f}s",
        flush=True,
    )
    return serialized


def _report(output_dir: Path, baseline: dict[str, Any] | None, scaling: dict[str, Any] | None) -> None:
    lines = [
        "# Geometry and migration hot-path profile",
        "",
        "Machine timings are indicative and will vary across hardware and software versions.",
        "The point-count sweep uses synthetic monotone-x meanders; it measures algorithmic costs and is not a solver trajectory.",
        "",
    ]
    if baseline is not None:
        counts = baseline["geometry_counts"]
        timings = baseline["solver_component_timings"]
        accounting = baseline["geometry_timing_accounting"]
        wall = float(baseline["total_wall_seconds"])
        geometry = float(timings.get("geometry") or 0.0)
        lines.extend([
            "## Evolving 5,000-step Numba baseline",
            "",
            f"Completed {baseline['completed_steps']:,} steps; final Ns={baseline['final_Ns']}; "
            f"sinuosity={baseline['final_sinuosity']:.9g}; cutoffs={baseline['number_of_cutoffs']}; "
            f"wall={wall:.3f}s.",
            "",
            "| Component | Seconds | % wall | % geometry |",
            "|---|---:|---:|---:|",
        ])
        display = (
            ("flowfield_loop", "flowfield_loop"),
            ("flowfield_final", "flowfield_final"),
            ("flowfield_total", "flowfield_total"),
            ("vertical coefficients (all flow calls)", "vertical_coefficient_seconds_all_flow_calls"),
            ("modal coefficients (all flow calls)", "modal_coefficient_seconds_all_flow_calls"),
            ("SEMIANA response (all flow calls)", "semiana_response_seconds_all_flow_calls"),
            ("move", "move"),
            ("dxdy2 (subset of move)", "dxdy2"),
            ("coordinate migration (move minus dxdy2)", "coordinate_migration"),
            ("geometry", "geometry"),
            *tuple((key, key) for key in GEOMETRY_TIMING_KEYS),
            ("other geometry", "other_geometry"),
            ("update", "update"),
            ("saving", "saving"),
            ("other wall", "other_wall"),
        )
        timing_values = {
            **timings,
            "vertical_coefficient_seconds_all_flow_calls": baseline["vertical_coefficient_seconds_all_flow_calls"],
            "modal_coefficient_seconds_all_flow_calls": baseline["modal_coefficient_seconds_all_flow_calls"],
            "semiana_response_seconds_all_flow_calls": baseline["semiana_response_seconds_all_flow_calls"],
            "other_geometry": accounting["other_geometry_seconds"],
            "other_wall": accounting["other_wall_seconds"],
        }
        geometry_keys = set(GEOMETRY_TIMING_KEYS) | {"geometry", "other_geometry"}
        for label, key in display:
            raw = timing_values.get(key)
            if raw is None:
                continue
            seconds = float(raw)
            wall_pct = 100.0 * seconds / wall if wall else 0.0
            geom_pct = 100.0 * seconds / geometry if key in geometry_keys and geometry else None
            geom_text = f"{geom_pct:.2f}%" if geom_pct is not None else "n/a"
            lines.append(f"| {label} | {seconds:.6g} | {wall_pct:.2f}% | {geom_text} |")
        lines.extend([
            "",
            "Geometry component buckets are nonoverlapping. `geometry_initial_uniformization` owns its arclength work; `geometry_arclength` includes only standalone post-cut arclength recomputations. Arclength work inside conditional regridding is charged to `geometry_resample`. `geometry_smoothing` is removed from the resample bucket. Legacy `neck` and `smoothing` remain subset aliases.",
            "",
            f"Counts: {counts['geometry_calls']} geometry calls, {counts['neck_detector_calls']} neck searches, "
            f"{counts['cutoff_events']} cutoff events, {counts['smoothing_calls']} smoothing calls, "
            f"{counts['spacing_resamples']} spacing resamples.",
            "",
            "Baseline data: `baseline_run.json`; checkpoint scalars: `baseline_checkpoints.csv`.",
            "",
        ])
    if scaling is not None:
        lines.extend([
            "## Synthetic geometry size sweep",
            "",
            "Each workload is warmed once and then measured over repeated calls. Values below are median milliseconds; full min/median/max results are in `geometry_scaling.csv` and `.json`.",
            "",
            "| N | Workload | Median (ms) |",
            "|---:|---|---:|",
        ])
        for row in scaling["rows"]:
            lines.append(
                f"| {row['N_input']} | {row['workload']} | {1000.0 * row['wall_seconds_median']:.6g} |"
            )
        lines.extend([
            "",
            "The direct detector and enabled-no-hit geometry workloads assert that no synthetic cutoff occurred. The initial uniformization and curvature workloads reproduce those operations for measurement only.",
            "",
            "## cProfile",
            "",
            "A warmed N=5,000 `geometry4` call with cutoff search enabled and no hit was profiled. The cumulative-time listing is in `cprofile_geometry4.txt`; focused function rows are in `cprofile_geometry4.json`.",
        ])
        tree_profile = scaling["profile"]["kdtree_native_timing"]
        lines.append(
            f"The benchmark-only KD-tree wrapper measured construction at "
            f"{tree_profile['construction_seconds'] * 1000.0:.3f} ms and "
            f"{tree_profile['query_ball_point_calls']:,} `query_ball_point` calls at "
            f"{tree_profile['query_ball_point_seconds'] * 1000.0:.3f} ms cumulative."
        )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("microbench", "baseline", "all"), default="microbench")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.steps <= 0 or args.case <= 0 or args.repetitions <= 0:
        raise SystemExit("--steps, --case, and --repetitions must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{time.time_ns() % 1_000_000:06d}"
    output_dir = args.results_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline = None
    scaling = None
    if args.mode in ("baseline", "all"):
        baseline = _run_baseline(args.case, args.steps, output_dir)
    if args.mode in ("microbench", "all"):
        scaling = _run_scaling(args.repetitions, output_dir)
    _report(output_dir, baseline, scaling)
    print(f"[geometry profile] results: {output_dir}", flush=True)
    if baseline is not None and baseline["run_error"] is not None:
        return 1
    if baseline is not None and baseline["completed_steps"] != args.steps:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
