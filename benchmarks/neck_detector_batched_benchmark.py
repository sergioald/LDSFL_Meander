"""Benchmark and deterministic differential audit for batched neck queries.

Run from the repository root with::

    python benchmarks/neck_detector_batched_benchmark.py --repetitions 5

All generated artifacts are placed in the ignored ``benchmarks/Output`` tree.
The test geometries are synthetic and are used only to compare the detector
implementations; they are not presented as model trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldsfl import mathutils  # noqa: E402

SIZES = (250, 1_000, 2_500, 5_000)
BATCH_SIZES = (16, 32, 64, 128, 256)
WORK_SEPARATION = 18
DEFAULT_OUTPUT = ROOT / "benchmarks" / "Output" / "neck_detector_batched"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _case_geometry(size: int, case: str) -> tuple[np.ndarray, np.ndarray, int, float]:
    ss = WORK_SEPARATION
    end = size - ss - 1
    radius = 1.0
    if case == "dense":
        x = 0.004 * np.arange(size, dtype=np.float64)
        y = 0.006 * np.sin(np.arange(size, dtype=np.float64) / 13.0)
        return x, y, ss, radius

    x = 10.0 * np.arange(size, dtype=np.float64)
    y = np.zeros(size, dtype=np.float64)
    if case == "no_hit":
        return x, y, ss, radius

    if case == "early_hit":
        i = 0
    elif case == "middle_hit":
        i = end // 2
    elif case == "late_hit":
        i = end - ss
    else:
        raise ValueError(f"unknown benchmark case: {case}")
    j = i + ss
    x[j] = x[i] + 0.25
    return x, y, ss, radius


def _measure_structure(
    fn,
    x: np.ndarray,
    y: np.ndarray,
    ss: int,
    radius: float,
    batch_size: int | None,
) -> dict[str, int]:
    counts = {
        "query_batches": 0,
        "effective_query_points": 0,
        "neighbor_references_total": 0,
        "max_neighbor_references_in_batch": 0,
    }
    base_tree = mathutils.cKDTree

    class CountingTree:
        def __init__(self, *args, **kwargs):
            self._tree = base_tree(*args, **kwargs)

        def query_ball_point(self, points, *args, **kwargs):
            counts["query_batches"] += 1
            point_array = np.asarray(points)
            counts["effective_query_points"] += (
                int(point_array.shape[0]) if point_array.ndim > 1 else 1
            )
            result = self._tree.query_ball_point(points, *args, **kwargs)
            if point_array.ndim > 1:
                neighbor_references = sum(len(items) for items in result)
            else:
                neighbor_references = len(result)
            counts["neighbor_references_total"] += neighbor_references
            counts["max_neighbor_references_in_batch"] = max(
                counts["max_neighbor_references_in_batch"], neighbor_references
            )
            return result

        def __getattr__(self, name):
            return getattr(self._tree, name)

    mathutils.cKDTree = CountingTree
    try:
        if batch_size is None:
            result = fn(x, y, ss, radius, workers=1)
        else:
            result = fn(x, y, ss, radius, workers=1, batch_size=batch_size)
    finally:
        mathutils.cKDTree = base_tree
    counts["result_is_hit"] = int(result is not None)
    return counts


def _time_detector_suite(x, y, ss, radius, repetitions: int) -> dict[int | None, dict[str, Any]]:
    implementations = [
        (None, mathutils._kdtree_first_hit_point_pair_scalar),
        *(
            (batch_size, mathutils._kdtree_first_hit_point_pair_batched)
            for batch_size in BATCH_SIZES
        ),
    ]

    def call(batch_size, fn):
        if batch_size is None:
            return fn(x, y, ss, radius, workers=1)
        return fn(x, y, ss, radius, workers=1, batch_size=batch_size)

    expected = {
        batch_size: call(batch_size, fn)
        for batch_size, fn in implementations
    }  # Warm every implementation before measuring.
    samples = {batch_size: [] for batch_size, _fn in implementations}
    for repetition in range(repetitions):
        ordered = implementations if repetition % 2 == 0 else list(reversed(implementations))
        for batch_size, fn in ordered:
            started = time.perf_counter()
            result = call(batch_size, fn)
            samples[batch_size].append(time.perf_counter() - started)
            if result != expected[batch_size]:
                raise AssertionError(
                    f"detector result changed at batch_size={batch_size}: "
                    f"{expected[batch_size]} != {result}"
                )

    measured = {}
    for batch_size, fn in implementations:
        values = samples[batch_size]
        measured[batch_size] = {
            "min_seconds": min(values),
            "median_seconds": statistics.median(values),
            "max_seconds": max(values),
            "result": expected[batch_size],
            **_measure_structure(fn, x, y, ss, radius, batch_size),
        }
    return measured


def _run_batch_sweep(repetitions: int, output_dir: Path) -> dict[str, Any]:
    rows = []
    cases = ("no_hit", "early_hit", "middle_hit", "late_hit", "dense")
    for size in SIZES:
        for case in cases:
            x, y, ss, radius = _case_geometry(size, case)
            measured = _time_detector_suite(x, y, ss, radius, repetitions)
            scalar = measured[None]
            for batch_size in BATCH_SIZES:
                batched = measured[batch_size]
                if batched["result"] != scalar["result"]:
                    raise AssertionError(
                        f"result mismatch N={size} case={case} batch_size={batch_size}: "
                        f"{scalar['result']} != {batched['result']}"
                    )
                rows.append(
                    {
                        "N": size,
                        "case": case,
                        "batch_size": batch_size,
                        "repetitions": repetitions,
                        "scalar_min_seconds": scalar["min_seconds"],
                        "scalar_median_seconds": scalar["median_seconds"],
                        "scalar_max_seconds": scalar["max_seconds"],
                        "batched_min_seconds": batched["min_seconds"],
                        "batched_median_seconds": batched["median_seconds"],
                        "batched_max_seconds": batched["max_seconds"],
                        "speedup": scalar["median_seconds"] / batched["median_seconds"],
                        "scalar_query_batches": scalar["query_batches"],
                        "scalar_effective_query_points": scalar["effective_query_points"],
                        "batched_query_batches": batched["query_batches"],
                        "batched_effective_query_points": batched["effective_query_points"],
                        "batched_total_neighbor_references": batched["neighbor_references_total"],
                        "batched_max_neighbor_references_in_batch": batched[
                            "max_neighbor_references_in_batch"
                        ],
                        "result": batched["result"],
                    }
                )
        print(f"[detector sweep] completed N={size}", flush=True)

    output_path = output_dir / "batch_size_sweep.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": rows, "csv": output_path.name}


def _random_geometry(rng: np.random.Generator, case_index: int):
    size = int(rng.integers(24, 251))
    ss = int(rng.integers(1, min(24, size - 2)))
    mode = case_index % 3
    if mode == 0:
        spacing = float(rng.uniform(2.0, 12.0))
        x = spacing * np.arange(size, dtype=np.float64)
        y = np.zeros(size, dtype=np.float64)
        radius = float(rng.uniform(0.1, min(1.5, spacing * 0.4)))
    elif mode == 1:
        if size - 2 * ss - 1 < 0:
            ss = (size - 2) // 2
        x = 5.0 * np.arange(size, dtype=np.float64)
        y = rng.normal(0.0, 0.02, size=size)
        radius = float(rng.uniform(0.05, 1.5))
        end = size - ss - 1
        i = int(rng.integers(0, end - ss + 1))
        j = i + ss
        x[j] = x[i] + float(rng.uniform(0.0, radius * 0.9))
    else:
        increments = rng.uniform(0.08, 1.8, size=size - 1)
        t = np.linspace(0.0, float(rng.uniform(3.0, 30.0)) * np.pi, size)
        amplitude = float(rng.uniform(0.0, 15.0))
        y = amplitude * np.sin(t) + rng.normal(0.0, 0.05, size=size)
        radius = float(rng.uniform(0.1, 4.0))
        # Compress a valid arc-length-index interval in x and smoothly align
        # its endpoints in y to construct an ordered near-self approach.
        if size > 60 and case_index % 2 == 0:
            end = size - ss - 1
            max_start = max(0, min(size // 3, end - ss))
            start = int(rng.integers(0, max_start + 1))
            later = int(rng.integers(start + ss, end + 1))
            gap = min(radius * 0.5, 1.0)
            increments[start:later] = gap / (later - start)
            y[start : later + 1] -= np.linspace(
                0.0, y[later] - y[start], later - start + 1
            )
        x = np.concatenate(([0.0], np.cumsum(increments)))
    return x, y, ss, radius


def _run_randomized_audit(count: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    hits = 0
    no_hits = 0
    mismatches = []
    batch_choices = (1, 3, 16, 64, 128, 256)
    for idx in range(count):
        x, y, ss, radius = _random_geometry(rng, idx)
        batch_size = int(rng.choice(batch_choices))
        workers = int(rng.choice((1, 2)))
        scalar = mathutils._kdtree_first_hit_point_pair_scalar(x, y, ss, radius, workers=workers)
        batched = mathutils._kdtree_first_hit_point_pair_batched(
            x, y, ss, radius, workers=workers, batch_size=batch_size
        )
        if scalar is None:
            no_hits += 1
        else:
            hits += 1
        if scalar != batched:
            mismatches.append(
                {
                    "case": idx,
                    "N": int(x.size),
                    "ss": ss,
                    "radius": radius,
                    "batch_size": batch_size,
                    "workers": workers,
                    "scalar": scalar,
                    "batched": batched,
                }
            )
    return {
        "seed": seed,
        "comparisons": count,
        "scalar_hits": hits,
        "no_hits": no_hits,
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:10],
    }


def _run_paired_confirmation(repetitions: int, output_dir: Path) -> dict[str, Any]:
    rows = []
    for size in (2_500, 5_000):
        for case in ("no_hit", "middle_hit", "late_hit"):
            x, y, ss, radius = _case_geometry(size, case)
            scalar_fn = mathutils._kdtree_first_hit_point_pair_scalar
            batched_fn = mathutils._kdtree_first_hit_point_pair_batched
            scalar_result = scalar_fn(x, y, ss, radius, workers=1)
            batched_result = batched_fn(x, y, ss, radius, workers=1, batch_size=32)
            if scalar_result != batched_result:
                raise AssertionError(
                    f"paired confirmation mismatch N={size} case={case}: "
                    f"{scalar_result} != {batched_result}"
                )
            samples = {"scalar": [], "batched": []}
            paired_speedups = []
            for repetition in range(repetitions):
                order = ("scalar", "batched") if repetition % 2 == 0 else ("batched", "scalar")
                current = {}
                for name in order:
                    started = time.perf_counter()
                    if name == "scalar":
                        result = scalar_fn(x, y, ss, radius, workers=1)
                    else:
                        result = batched_fn(x, y, ss, radius, workers=1, batch_size=32)
                    elapsed = time.perf_counter() - started
                    if result != scalar_result:
                        raise AssertionError(f"paired result changed for N={size}, {case}, {name}")
                    samples[name].append(elapsed)
                    current[name] = elapsed
                paired_speedups.append(current["scalar"] / current["batched"])
            rows.append(
                {
                    "N": size,
                    "case": case,
                    "batch_size": 32,
                    "repetitions": repetitions,
                    "scalar_min_seconds": min(samples["scalar"]),
                    "scalar_median_seconds": statistics.median(samples["scalar"]),
                    "scalar_max_seconds": max(samples["scalar"]),
                    "batched_min_seconds": min(samples["batched"]),
                    "batched_median_seconds": statistics.median(samples["batched"]),
                    "batched_max_seconds": max(samples["batched"]),
                    "paired_speedup_median": statistics.median(paired_speedups),
                    "result": scalar_result,
                }
            )
        print(f"[paired detector confirmation] completed N={size}", flush=True)
    output_path = output_dir / "paired_confirmation.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": rows, "csv": output_path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--random-cases", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.repetitions < 1 or args.random_cases < 1:
        parser.error("--repetitions and --random-cases must be positive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    sweep = _run_batch_sweep(args.repetitions, output_dir)
    confirmation = _run_paired_confirmation(args.repetitions, output_dir)
    audit = _run_randomized_audit(args.random_cases, args.seed)
    summary = {
        "created_utc": stamp,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "machine": platform.platform(),
        "batch_sizes": BATCH_SIZES,
        "sizes": SIZES,
        "cases_per_size": ("no_hit", "early_hit", "middle_hit", "late_hit", "dense"),
        "repetitions": args.repetitions,
        "randomized_audit": audit,
        "batch_sweep": sweep,
        "paired_confirmation": confirmation,
    }
    _write_json(output_dir / "summary.json", summary)
    print(f"Artifacts: {output_dir}", flush=True)
    print(
        f"Random audit: {audit['comparisons']} comparisons, {audit['scalar_hits']} hits, "
        f"{audit['no_hits']} no-hits, {audit['mismatches']} mismatches",
        flush=True,
    )
    return int(audit["mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
