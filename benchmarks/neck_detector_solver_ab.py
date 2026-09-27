"""Exact solver A/B validation for scalar versus batched neck detection.

The runner performs a paired 1,000-step case-1 run with the same Numba
flowfield controls, first forcing the private scalar detector and then using
the production bounded-batched detector. It automatically continues to 5,000
steps only if the 1,000-step runs both complete and are exactly identical.
Generated numerical data are written under ignored ``benchmarks/Output``.

This runner intentionally has no option for 10,000-step or longer studies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import long_term_backend_parity as parity  # noqa: E402

from ldsfl import geometry as geometry_module  # noqa: E402
from ldsfl import main as main_module  # noqa: E402
from ldsfl import mathutils  # noqa: E402

DEFAULT_OUTPUT = ROOT / "benchmarks" / "Output" / "neck_detector_solver_ab"
STAGES = {
    1_000: (0, 1, 10, 100, 500, 1_000),
    5_000: (0, 1, 10, 100, 500, 1_000, 2_500, 5_000),
}
HISTORY_FIELDS = (
    "step", "jt", "dt", "dt_cum", "Ns", "cut_cnt", "sinuosity", "beta",
    "theta0", "ds", "Cf0",
)
CHECKPOINT_SCALARS = (
    "jt", "dt", "dt_cum", "Ns", "cut_cnt", "sinuosity", "beta", "theta0", "ds",
    "Cf0", "flow_flag", "resonance_state", "resonance_decay_rate",
)
CHECKPOINT_ARRAYS = ("x", "y", "s", "c", "U")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(parity._json_value(value), indent=2), encoding="utf-8")


def _fmt(value: Any) -> str:
    return f"{float(value):.6g}" if value is not None else "n/a"


def _run_one_with_pair_capture(*, label: str, steps: int, checkpoints: set[int], force_scalar: bool):
    pair_events: list[dict[str, int]] = []
    current_step = {"jt": None}
    original_geometry4 = main_module.geometry4
    original_find = geometry_module.find_neck_cutoff_kdtree_with_refine
    original_dispatch = mathutils._kdtree_first_hit_point_pair

    def contextual_geometry4(*args, **kwargs):
        current_step["jt"] = int(args[2] if len(args) > 2 else kwargs["jt"])
        return original_geometry4(*args, **kwargs)

    def recording_find(*args, **kwargs):
        pair = original_find(*args, **kwargs)
        if pair is not None:
            pair_events.append(
                {
                    "step": int(current_step["jt"]),
                    "i0": int(pair[0]),
                    "j0": int(pair[1]),
                }
            )
        return pair

    def scalar_dispatch(x, y, ss, dslim3, *, workers: int = 1):
        return mathutils._kdtree_first_hit_point_pair_scalar(
            x, y, ss, dslim3, workers=workers
        )

    main_module.geometry4 = contextual_geometry4
    geometry_module.find_neck_cutoff_kdtree_with_refine = recording_find
    if force_scalar:
        mathutils._kdtree_first_hit_point_pair = scalar_dispatch
    try:
        run = parity._run_one(
            label=label,
            case=1,
            steps=steps,
            checkpoints=checkpoints,
            flow_backend="numba",
            vertical_backend="numba",
            progress_every=250,
        )
    finally:
        main_module.geometry4 = original_geometry4
        geometry_module.find_neck_cutoff_kdtree_with_refine = original_find
        mathutils._kdtree_first_hit_point_pair = original_dispatch

    run["_detector_pair_events"] = pair_events
    return run


def _array_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    aa = np.asarray(a)
    bb = np.asarray(b)
    return aa.shape == bb.shape and bool(np.array_equal(aa, bb, equal_nan=True))


def _checkpoint_comparison(reference, candidate, checkpoints):
    rows = []
    exact = True
    snapshots = {}
    for step in checkpoints:
        ref = reference["_checkpoints"].get(step)
        cand = candidate["_checkpoints"].get(step)
        row: dict[str, Any] = {"step": int(step), "both_present": ref is not None and cand is not None}
        snapshot_detail: dict[str, Any] = {"step": int(step), "both_present": row["both_present"]}
        if ref is None or cand is None:
            row["exact"] = False
            exact = False
            rows.append(row)
            snapshots[str(step)] = snapshot_detail
            continue

        step_exact = True
        for field in CHECKPOINT_SCALARS:
            ref_value = ref.get(field)
            cand_value = cand.get(field)
            equal = ref_value == cand_value
            row[f"{field}_reference"] = ref_value
            row[f"{field}_batched"] = cand_value
            row[f"{field}_exact"] = bool(equal)
            snapshot_detail[f"{field}_reference"] = ref_value
            snapshot_detail[f"{field}_batched"] = cand_value
            if not equal:
                step_exact = False

        arrays = {}
        for field in CHECKPOINT_ARRAYS:
            ref_array = ref["_arrays"].get(field)
            cand_array = cand["_arrays"].get(field)
            equal = _array_equal(ref_array, cand_array)
            arrays[field] = {
                "both_present": ref_array is not None and cand_array is not None,
                "exact": equal,
            }
            if ref_array is not None and cand_array is not None:
                ref_values = np.asarray(ref_array, dtype=np.float64)
                cand_values = np.asarray(cand_array, dtype=np.float64)
                if ref_values.shape == cand_values.shape:
                    delta = np.abs(ref_values - cand_values)
                    arrays[field]["max_abs_difference"] = float(np.max(delta)) if delta.size else 0.0
                    arrays[field]["rms_difference"] = (
                        float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
                    )
                else:
                    arrays[field]["shape_reference"] = ref_values.shape
                    arrays[field]["shape_batched"] = cand_values.shape
            if not equal:
                step_exact = False
        row["array_comparisons"] = arrays
        row["exact"] = bool(step_exact)
        snapshot_detail["arrays"] = arrays
        snapshot_detail["reference_arrays"] = {
            field: ref["_arrays"].get(field) for field in CHECKPOINT_ARRAYS
        }
        snapshot_detail["batched_arrays"] = {
            field: cand["_arrays"].get(field) for field in CHECKPOINT_ARRAYS
        }
        rows.append(row)
        snapshots[str(step)] = snapshot_detail
        exact = exact and step_exact
    return rows, snapshots, exact


def _history_comparison(reference, candidate):
    ref_rows = reference["_scalar_history"]
    cand_rows = candidate["_scalar_history"]
    rows = []
    exact = len(ref_rows) == len(cand_rows)
    for index in range(max(len(ref_rows), len(cand_rows))):
        ref = ref_rows[index] if index < len(ref_rows) else None
        cand = cand_rows[index] if index < len(cand_rows) else None
        row: dict[str, Any] = {"history_index": index, "both_present": ref is not None and cand is not None}
        row_exact = ref is not None and cand is not None
        if row_exact:
            for field in HISTORY_FIELDS:
                row[f"{field}_reference"] = ref[field]
                row[f"{field}_batched"] = cand[field]
                equal = ref[field] == cand[field]
                row[f"{field}_exact"] = bool(equal)
                row_exact = row_exact and equal
        else:
            exact = False
        row["exact"] = bool(row_exact)
        rows.append(row)
        exact = exact and row_exact
    return rows, bool(exact)


def _cutoff_comparison(reference, candidate):
    ref_events = reference["_cutoff_events"]
    cand_events = candidate["_cutoff_events"]
    pair_ref = reference["_detector_pair_events"]
    pair_cand = candidate["_detector_pair_events"]
    rows = []
    exact_geometry = len(ref_events) == len(cand_events)
    exact_pairs = len(pair_ref) == len(pair_cand)
    for index in range(max(len(ref_events), len(cand_events), len(pair_ref), len(pair_cand))):
        ref = ref_events[index] if index < len(ref_events) else None
        cand = cand_events[index] if index < len(cand_events) else None
        ref_pair = pair_ref[index] if index < len(pair_ref) else None
        cand_pair = pair_cand[index] if index < len(pair_cand) else None
        geometry_fields = ("step", "cut_cnt", "Ns_before", "Ns_after", "Ns_after_geometry")
        geometry_match = ref is not None and cand is not None and all(
            ref.get(field) == cand.get(field) for field in geometry_fields
        )
        pair_match = ref_pair == cand_pair
        if ref is not None or cand is not None:
            exact_geometry = exact_geometry and bool(geometry_match)
        if ref_pair is not None or cand_pair is not None:
            exact_pairs = exact_pairs and bool(pair_match)
        rows.append(
            {
                "event_index": index + 1,
                "reference_geometry_event": ref,
                "batched_geometry_event": cand,
                "reference_selected_pair": ref_pair,
                "batched_selected_pair": cand_pair,
                "geometry_cutoff_exact": bool(geometry_match),
                "selected_pair_exact": bool(pair_match),
                "points_removed_reference": (
                    int(ref["Ns_before"] - ref["Ns_after"]) if ref is not None else None
                ),
                "points_removed_batched": (
                    int(cand["Ns_before"] - cand["Ns_after"]) if cand is not None else None
                ),
            }
        )
    ref_steps = [event["step"] for event in ref_events]
    cand_steps = [event["step"] for event in cand_events]
    ref_removed = [event["Ns_before"] - event["Ns_after"] for event in ref_events]
    cand_removed = [event["Ns_before"] - event["Ns_after"] for event in cand_events]
    details = {
        "scalar_cutoff_count": len(ref_events),
        "batched_cutoff_count": len(cand_events),
        "cutoff_counts_match": len(ref_events) == len(cand_events),
        "cutoff_step_sequences_match": ref_steps == cand_steps,
        "points_removed_per_cutoff_match": ref_removed == cand_removed,
        "selected_pair_count_scalar": len(pair_ref),
        "selected_pair_count_batched": len(pair_cand),
        "selected_pair_sequence_matches": pair_ref == pair_cand,
        "scalar_selected_pairs": pair_ref,
        "batched_selected_pairs": pair_cand,
        "cutoff_event_steps_scalar": ref_steps,
        "cutoff_event_steps_batched": cand_steps,
        "points_removed_scalar": ref_removed,
        "points_removed_batched": cand_removed,
    }
    return rows, details, bool(exact_geometry and exact_pairs)


def _timing_summary(run):
    components = run["solver_component_timings"]
    return {
        "total_wall_seconds": run["total_wall_seconds"],
        "geometry_seconds": components.get("geometry"),
        "neck_detector_seconds": components.get("geometry_neck"),
        "flowfield_total_seconds": components.get("flowfield_total"),
        "flowfield_loop_seconds": components.get("flowfield_loop"),
        "flowfield_final_seconds": components.get("flowfield_final"),
        "move_seconds": components.get("move"),
        "update_seconds": components.get("update"),
        "saving_seconds": components.get("saving"),
        "geometry_calls": run["geometry_counts"]["geometry_calls"],
        "neck_searches": run["geometry_counts"]["neck_detector_calls"],
        "cutoff_events": run["geometry_counts"]["cutoff_events"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: parity._json_value(value) for key, value in row.items()})


def _run_stage(steps: int, output_root: Path) -> tuple[dict[str, Any], bool, Path]:
    checkpoints = set(STAGES[steps])
    print(f"[neck A/B] starting {steps:,}-step scalar run", flush=True)
    scalar = _run_one_with_pair_capture(
        label=f"scalar_detector_{steps}",
        steps=steps,
        checkpoints=checkpoints,
        force_scalar=True,
    )
    print(
        f"[neck A/B] scalar {steps:,}: completed={scalar['completed_steps']}, "
        f"cutoffs={scalar['number_of_cutoffs']}, wall={scalar['total_wall_seconds']:.3f}s, "
        f"error={scalar['run_error']}",
        flush=True,
    )

    print(f"[neck A/B] starting {steps:,}-step batched run", flush=True)
    batched = _run_one_with_pair_capture(
        label=f"batched_detector_{steps}",
        steps=steps,
        checkpoints=checkpoints,
        force_scalar=False,
    )
    print(
        f"[neck A/B] batched {steps:,}: completed={batched['completed_steps']}, "
        f"cutoffs={batched['number_of_cutoffs']}, wall={batched['total_wall_seconds']:.3f}s, "
        f"error={batched['run_error']}",
        flush=True,
    )

    checkpoint_rows, checkpoint_states, checkpoint_exact = _checkpoint_comparison(
        scalar, batched, STAGES[steps]
    )
    history_rows, history_exact = _history_comparison(scalar, batched)
    cutoff_rows, cutoff_details, cutoff_exact = _cutoff_comparison(scalar, batched)
    complete = all(
        run["run_error"] is None and run["completed_steps"] == steps for run in (scalar, batched)
    )
    exact = bool(complete and checkpoint_exact and history_exact and cutoff_exact)
    scalar_timing = _timing_summary(scalar)
    batched_timing = _timing_summary(batched)
    speedups = {}
    for key, a_key, b_key in (
        ("whole_solver", "total_wall_seconds", "total_wall_seconds"),
        ("geometry", "geometry_seconds", "geometry_seconds"),
        ("neck_detector", "neck_detector_seconds", "neck_detector_seconds"),
    ):
        denominator = batched_timing[b_key]
        speedups[key] = (
            float(scalar_timing[a_key] / denominator)
            if denominator and scalar_timing[a_key] is not None
            else None
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_%f")
    output_dir = output_root / stamp / f"{steps}_steps"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "requested_steps": steps,
        "checkpoints": list(STAGES[steps]),
        "exact_parity": exact,
        "both_runs_completed": complete,
        "checkpoint_state_exact": checkpoint_exact,
        "full_scalar_history_exact": history_exact,
        "cutoff_and_pair_sequences_exact": cutoff_exact,
        "runs": {
            "scalar": parity._serial_run(scalar),
            "batched": parity._serial_run(batched),
        },
        "timings": {"scalar": scalar_timing, "batched": batched_timing},
        "speedups": speedups,
        "cutoff_comparison": cutoff_details,
        "controls": {
            "case": 1,
            "flow_backend": "numba",
            "vertical_backend": "numba",
            "flow_paral": 0,
            "numba_parallel": False,
            "numba_fastmath": False,
            "do_plots": False,
            "geometry_controls": "unchanged run_case defaults",
            "detector_A": "private scalar reference forced by benchmark-only monkeypatch",
            "detector_B": "private production dispatcher, bounded batch size 32",
        },
        "timing_note": "Wall time includes run_case, final flowfield recomputation, and temporary output finalization. Both detector wrappers use the same pair-capture instrumentation.",
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "checkpoint_states.json", checkpoint_states)
    _write_json(output_dir / "cutoff_and_pair_events.json", cutoff_rows)
    _write_csv(output_dir / "checkpoint_comparison.csv", checkpoint_rows)
    _write_csv(output_dir / "scalar_history_comparison.csv", history_rows)
    _write_csv(output_dir / "cutoff_and_pair_comparison.csv", cutoff_rows)
    report = [
        f"# Scalar versus batched neck detector: {steps:,}-step validation",
        "",
        f"Exact parity: **{exact}**. Both runs completed: **{complete}**.",
        "",
        "| Run | Wall s | Geometry s | Neck detector s | Flowfield total s | Move s | Update s | Saving s | Cutoffs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, timing in (("Scalar", scalar_timing), ("Batched", batched_timing)):
        report.append(
            f"| {name} | {_fmt(timing['total_wall_seconds'])} | {_fmt(timing['geometry_seconds'])} | "
            f"{_fmt(timing['neck_detector_seconds'])} | {_fmt(timing['flowfield_total_seconds'])} | "
            f"{_fmt(timing['move_seconds'])} | {_fmt(timing['update_seconds'])} | "
            f"{_fmt(timing['saving_seconds'])} | {timing['cutoff_events']} |"
        )
    report.extend(
        [
            "",
            f"Cutoff steps match: **{cutoff_details['cutoff_step_sequences_match']}**. "
            f"Selected pair sequence matches: **{cutoff_details['selected_pair_sequence_matches']}**. "
            f"Points removed per cutoff match: **{cutoff_details['points_removed_per_cutoff_match']}**.",
            "",
            f"Speedups (scalar/batched): whole solver {_fmt(speedups['whole_solver'])}x; "
            f"geometry {_fmt(speedups['geometry'])}x; neck detector {_fmt(speedups['neck_detector'])}x.",
            "",
            "The JSON and CSV files preserve checkpoint arrays, every scalar-history row, all selected detector pairs, and every geometry cutoff event.",
            "",
            "All wall timings are indicative and machine-specific.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[neck A/B] saved {steps:,}-step diagnostics to {output_dir}", flush=True)
    return summary, exact, output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for steps in (1_000, 5_000):
        summary, exact, output_dir = _run_stage(steps, args.output_root)
        reports.append({"steps": steps, "exact_parity": exact, "output_dir": str(output_dir)})
        if not exact:
            print(
                f"[neck A/B] stopping after {steps:,} steps because the exact-parity gate failed",
                flush=True,
            )
            break
    _write_json(args.output_root / "latest_validation.json", reports)
    return 0 if reports[-1]["exact_parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
