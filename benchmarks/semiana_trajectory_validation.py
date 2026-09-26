"""Short solver trajectory A/B validation for internal SEMIANA strategies.

Only the reviewed 1,000-, 5,000-, and 10,000-step validations are accepted.
Generated state arrays and reports are written beneath the ignored
benchmarks/Output directory.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import long_term_backend_parity as parity  # noqa: E402

from ldsfl import flowfield as flowfield_module  # noqa: E402

CHECKPOINTS = {
    1_000: (0, 1, 10, 100, 250, 500, 750, 1_000),
    5_000: (0, 1, 10, 100, 500, 1_000, 2_500, 5_000),
    10_000: (0, 1, 10, 100, 500, 1_000, 2_500, 5_000, 7_500, 10_000),
}
CHECKPOINT_FIELDS = (
    "step", "jt", "dt", "dt_cum", "Ns", "cut_cnt", "sinuosity", "Cf0",
    "beta", "theta0", "ds", "flow_flag", "resonance_state", "resonance_decay_rate",
    "max_abs_curvature", "mean_U", "max_U", "max_abs_U",
)


def _run_strategy(strategy: str, *, steps: int, checkpoints: set[int], progress_every: int) -> dict[str, Any]:
    original = flowfield_module._run_modes_numba

    if strategy not in ("public_default", "legacy", "recursive"):
        raise ValueError(f"Unknown SEMIANA strategy: {strategy!r}")

    def explicit_response(**kwargs):
        kwargs["strategy"] = strategy
        return original(**kwargs)

    if strategy in ("legacy", "recursive"):
        flowfield_module._run_modes_numba = explicit_response
    try:
        return parity._run_one(
            label=f"{strategy}_semiana",
            case=1,
            steps=steps,
            checkpoints=checkpoints,
            flow_backend="numba",
            vertical_backend="numba",
            progress_every=progress_every,
        )
    finally:
        flowfield_module._run_modes_numba = original


def _checkpoint_metadata(run: dict[str, Any], checkpoints: tuple[int, ...]) -> dict[str, Any]:
    output = {}
    for step in checkpoints:
        snapshot = run["_checkpoints"].get(step)
        if snapshot is None:
            output[str(step)] = None
            continue
        record = {key: snapshot.get(key) for key in CHECKPOINT_FIELDS}
        record["array_shapes"] = {
            name: None if array is None else list(np.asarray(array).shape)
            for name, array in snapshot["_arrays"].items()
        }
        output[str(step)] = record
    return output


def _checkpoint_archive(output_dir: Path, runs: list[dict[str, Any]], checkpoints: tuple[int, ...]) -> None:
    arrays = {}
    for run in runs:
        label = run["label"]
        for step in checkpoints:
            snapshot = run["_checkpoints"].get(step)
            if snapshot is None:
                continue
            for field, values in snapshot["_arrays"].items():
                if values is not None:
                    arrays[f"{label}_step_{step}_{field}"] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(output_dir / "checkpoint_arrays.npz", **arrays)


def _point_counts_match(rows: list[dict[str, Any]]) -> bool:
    return all(
        not row["both_present"] or int(row["Ns_reference"]) == int(row["Ns_candidate"])
        for row in rows
    )


def _first_geometry_difference(details: dict[str, Any]) -> dict[str, Any] | None:
    metric_names = (
        "max_abs_x_difference", "max_abs_y_difference", "max_abs_s_difference",
        "max_abs_curvature_difference", "max_abs_U_difference",
    )
    for step, detail in sorted(details.items(), key=lambda item: int(item[0])):
        for name in metric_names:
            if detail.get(name, 0.0) > 0.0:
                return {"step": int(step), "metric": name, "absolute_difference": detail[name]}
    return None


def _max_checkpoint_metric(details: dict[str, Any], name: str) -> float | None:
    values = [detail[name] for detail in details.values() if name in detail]
    return max(values) if values else None


def _save_validation(steps: int, checkpoints: tuple[int, ...], runs: list[dict[str, Any]], results_root: Path) -> Path:
    legacy, recursive = runs
    checkpoint_rows, checkpoint_details = parity._checkpoint_comparison(legacy, recursive, list(checkpoints))
    scalar_rows, trajectory, _shared_steps = parity._scalar_comparison(legacy, recursive)
    cutoff_rows, cutoff_details = parity._cutoff_comparison(legacy, recursive)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{time.time_ns() % 1_000_000:06d}"
    output_dir = results_root / run_id / f"steps_{steps}"
    output_dir.mkdir(parents=True, exist_ok=False)

    _checkpoint_archive(output_dir, runs, checkpoints)
    (output_dir / "checkpoint_states.json").write_text(
        json.dumps({run["label"]: _checkpoint_metadata(run, checkpoints) for run in runs}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cutoff_events.json").write_text(
        json.dumps({run["label"]: run["_cutoff_events"] for run in runs}, indent=2),
        encoding="utf-8",
    )
    parity._write_csv(output_dir / "checkpoint_comparison.csv", checkpoint_rows)
    parity._write_csv(output_dir / "scalar_history_comparison.csv", scalar_rows)
    parity._write_csv(output_dir / "cutoff_comparison.csv", cutoff_rows)

    scalar_crossings = trajectory
    max_geometry = {
        key: _max_checkpoint_metric(checkpoint_details, key)
        for key in (
            "max_abs_x_difference", "rms_x_difference", "max_abs_y_difference", "rms_y_difference",
            "max_abs_s_difference", "rms_s_difference", "max_abs_curvature_difference",
            "rms_curvature_difference", "max_abs_U_difference", "rms_U_difference",
        )
    }
    timings = {}
    for run in runs:
        total_flow = run["flowfield_total_seconds"]
        timings[run["label"]] = {
            "total_wall_seconds": run["total_wall_seconds"],
            "flowfield_loop_seconds": run["flowfield_loop_seconds"],
            "flowfield_final_seconds": run["flowfield_final_seconds"],
            "flowfield_total_seconds": total_flow,
            "semiana_response_seconds_all_flow_calls": run["semiana_response_seconds_all_flow_calls"],
            "vertical_coefficient_seconds_all_flow_calls": run["vertical_coefficient_seconds_all_flow_calls"],
            "modal_coefficient_seconds_all_flow_calls": run["modal_coefficient_seconds_all_flow_calls"],
            "move_seconds": run["solver_component_timings"].get("move"),
            "geometry_seconds": run["solver_component_timings"].get("geometry"),
            "update_seconds": run["solver_component_timings"].get("update"),
            "migration_remainder_seconds": sum(
                float(run["solver_component_timings"].get(key) or 0.0)
                for key in ("move", "geometry", "update")
            ),
            "wall_minus_flowfield_seconds": None if total_flow is None else run["total_wall_seconds"] - total_flow,
        }

    completed = [run["completed_steps"] == steps and run["run_error"] is None for run in runs]
    summary = {
        "comparison": "numba_vertical_legacy_vs_recursive_semiana",
        "requested_steps": steps,
        "checkpoint_steps": list(checkpoints),
        "comparison_definitions": {
            "legacy": {"flow_backend": "numba", "vertical_backend": "numba", "semiana_strategy": "legacy"},
            "recursive": {"flow_backend": "numba", "vertical_backend": "numba", "semiana_strategy": "recursive"},
        },
        "same_initial_input_hashes": runs[0]["input_sha256"] == runs[1]["input_sha256"],
        "runs_completed_requested_steps": completed,
        "runs": [parity._serial_run(run) for run in runs],
        "timings": timings,
        "cutoff_comparison": cutoff_details,
        "cutoff_step_sequences": {
            run["label"]: [event["step"] for event in run["_cutoff_events"]]
            for run in runs
        },
        "point_count_equality_at_checkpoints": _point_counts_match(checkpoint_rows),
        "checkpoint_comparisons": checkpoint_details,
        "max_compatible_checkpoint_geometry_errors": max_geometry,
        "first_nonzero_checkpoint_geometry_difference": _first_geometry_difference(checkpoint_details),
        "scalar_trajectory_divergence": scalar_crossings,
        "final_sinuosity_difference": abs(legacy["final_sinuosity"] - recursive["final_sinuosity"]),
        "final_point_counts_match": legacy["final_Ns"] == recursive["final_Ns"],
        "final_resonance_flags_match": (
            legacy["_checkpoints"].get(steps, {}).get("flow_flag")
            == recursive["_checkpoints"].get(steps, {}).get("flow_flag")
        ),
        "instrumentation_errors": {run["label"]: run["capture_errors"] for run in runs},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "controls": {
            "flow_backend": "numba", "vertical_backend": "numba", "flow_paral": 0,
            "numba_parallel": False, "numba_fastmath": False, "TOLL": 1.0e-4,
            "run_case_controls": "same run_case defaults for both strategies",
        },
        "criteria_for_next_requested_stage": {
            "both_completed": all(completed),
            "no_solver_errors": all(run["run_error"] is None for run in runs),
            "cutoff_steps_match": cutoff_details["cutoff_steps_match"],
            "cutoff_counts_match": cutoff_details["cutoff_counts_match"],
            "no_capture_errors": all(not run["capture_errors"] for run in runs),
            "point_count_equality_at_checkpoints": _point_counts_match(checkpoint_rows),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(parity._json_value(summary), indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return output_dir


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# SEMIANA solver trajectory A/B validation",
        "",
        f"Requested and completed target: **{summary['requested_steps']:,} steps**. Both runs use Numba vertical coefficients; only the internal SEMIANA strategy differs.",
        "",
        "| Strategy | Wall (s) | Flow loop (s) | Final flow (s) | Flow total (s) | SEMIANA all calls (s) | Vertical coefficients (s) | Migration remainder (s) | Steps | Cutoffs | Ns | Sinuosity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        t = summary["timings"][run["label"]]
        lines.append(
            "| {label} | {wall:.6g} | {loop} | {final} | {total} | {semiana:.6g} | {vertical:.6g} | {remainder:.6g} | {steps} | {cutoffs} | {ns} | {sinuosity:.12g} |".format(
                label=run["label"], wall=t["total_wall_seconds"],
                loop="n/a" if t["flowfield_loop_seconds"] is None else f"{t['flowfield_loop_seconds']:.6g}",
                final="n/a" if t["flowfield_final_seconds"] is None else f"{t['flowfield_final_seconds']:.6g}",
                total="n/a" if t["flowfield_total_seconds"] is None else f"{t['flowfield_total_seconds']:.6g}",
                semiana=t["semiana_response_seconds_all_flow_calls"],
                vertical=t["vertical_coefficient_seconds_all_flow_calls"],
                remainder=t["migration_remainder_seconds"], steps=run["completed_steps"],
                cutoffs=run["number_of_cutoffs"], ns=run["final_Ns"], sinuosity=run["final_sinuosity"],
            )
        )
    lines.extend(["", "## Cutoff comparison", ""])
    lines.append(f"`{json.dumps(summary['cutoff_comparison'], sort_keys=True)}`")
    lines.append("")
    lines.append(f"Complete event steps: `{json.dumps(summary['cutoff_step_sequences'], sort_keys=True)}`")
    lines.extend(["", "## Checkpoint parity", ""])
    lines.append("Per-checkpoint scalars and geometry errors are in `checkpoint_comparison.csv`; both complete states (x, y, s, curvature, U) are preserved in `checkpoint_arrays.npz` and `checkpoint_states.json`.")
    lines.append("")
    lines.append(f"Final sinuosity difference: `{summary['final_sinuosity_difference']:.12g}`. Point counts match at all checkpoints: `{summary['point_count_equality_at_checkpoints']}`; final resonance flags match: `{summary['final_resonance_flags_match']}`.")
    lines.append("")
    lines.append(f"Next-stage structural criteria: `{json.dumps(summary['criteria_for_next_requested_stage'], sort_keys=True)}`. Timings are indicative and machine-specific.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=tuple(CHECKPOINTS), default=1_000)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--results-dir", type=Path,
        default=ROOT / "benchmarks" / "Output" / "semiana_trajectory_validation",
    )
    args = parser.parse_args()
    checkpoints = CHECKPOINTS[args.steps]
    runs = []
    for strategy in ("legacy", "recursive"):
        print(f"Starting {strategy} SEMIANA run for {args.steps:,} steps", flush=True)
        run = _run_strategy(
            strategy, steps=args.steps, checkpoints=set(checkpoints), progress_every=args.progress_every,
        )
        runs.append(run)
        if run["run_error"] is None:
            print(f"[{strategy}] completed {run['completed_steps']:,} steps in {run['total_wall_seconds']:.3f}s", flush=True)
        else:
            print(f"[{strategy}] failed: {run['run_error']['type']}: {run['run_error']['message']}", flush=True)

    output = _save_validation(args.steps, checkpoints, runs, args.results_dir)
    print(f"Wrote A/B validation artifacts: {output}", flush=True)
    successful = all(run["completed_steps"] == args.steps and run["run_error"] is None for run in runs)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
