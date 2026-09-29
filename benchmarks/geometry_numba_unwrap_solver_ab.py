"""Public-path production confirmation for the exact Numba geometry unwrap.

Run with the Numba-enabled environment::

    python benchmarks/geometry_numba_unwrap_solver_ab.py

The runner warms the production kernel, compares the public Python and Numba
controls for 1,000 steps, then runs one 5,000-step pair if the preflight is
bitwise exact. It does not patch the geometry unwrap binding.
"""

from __future__ import annotations

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

from ldsfl import mathutils as mathutils_module  # noqa: E402
from ldsfl.mathutils import matlab_gradient  # noqa: E402

DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "Output" / "geometry_numba_unwrap_solver_ab"
PREFLIGHT_CHECKPOINTS = (0, 1, 10, 100, 1_000)
CONFIRMATION_CHECKPOINTS = (0, 1, 10, 100, 500, 1_000, 2_500, 5_000)


def _make_candidate():
    from ldsfl.geometry_numba import unwrap_angles_like_matlab_numba

    return unwrap_angles_like_matlab_numba


def _same_exact(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        if left is None or right is None:
            return left is right
        return np.asarray(float(left), dtype=np.float64).tobytes() == np.asarray(
            float(right), dtype=np.float64
        ).tobytes()
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same_exact(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_same_exact(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _exact_comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for field in (
        "completed_steps", "final_Ns", "final_cut_cnt", "number_of_cutoffs", "final_sinuosity",
        "input_sha256", "errors",
    ):
        if not _same_exact(reference.get(field), candidate.get(field)):
            mismatches.append(f"run.{field}")
    if reference.get("run_error") is not None or candidate.get("run_error") is not None:
        mismatches.append("run_error")

    checkpoint_steps = sorted(set(reference["_checkpoints"]) | set(candidate["_checkpoints"]))
    checkpoint_mismatches: list[str] = []
    array_mismatches: list[str] = []
    for step in checkpoint_steps:
        left = reference["_checkpoints"].get(step)
        right = candidate["_checkpoints"].get(step)
        if left is None or right is None:
            checkpoint_mismatches.append(f"step={step}:presence")
            continue
        left_scalars = {key: value for key, value in left.items() if key != "_arrays"}
        right_scalars = {key: value for key, value in right.items() if key != "_arrays"}
        if not _same_exact(left_scalars, right_scalars):
            checkpoint_mismatches.append(f"step={step}:scalars")
        if not _same_exact(left.get("_arrays", {}), right.get("_arrays", {})):
            array_mismatches.append(f"step={step}:arrays")

    history_equal = _same_exact(reference["_scalar_history"], candidate["_scalar_history"])
    cutoff_equal = _same_exact(reference["_cutoff_events"], candidate["_cutoff_events"])
    if checkpoint_mismatches:
        mismatches.extend(f"checkpoint:{value}" for value in checkpoint_mismatches[:20])
    if array_mismatches:
        mismatches.extend(f"checkpoint:{value}" for value in array_mismatches[:20])
    if not history_equal:
        mismatches.append("scalar_history")
    if not cutoff_equal:
        mismatches.append("cutoff_events")
    return {
        "exact": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:50],
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_scalar_mismatch_count": len(checkpoint_mismatches),
        "checkpoint_array_mismatch_count": len(array_mismatches),
        "scalar_history_exact": history_equal,
        "cutoff_events_exact": cutoff_equal,
        "reference_cutoff_steps": [event["step"] for event in reference["_cutoff_events"]],
        "candidate_cutoff_steps": [event["step"] for event in candidate["_cutoff_events"]],
        "reference_cutoff_pairs": [
            [event.get("selected_pair_i0"), event.get("selected_pair_j0"), event.get("points_removed")]
            for event in reference["_cutoff_events"]
        ],
        "candidate_cutoff_pairs": [
            [event.get("selected_pair_i0"), event.get("selected_pair_j0"), event.get("points_removed")]
            for event in candidate["_cutoff_events"]
        ],
    }


def _warm_candidate(kernel, numba_module) -> dict[str, Any]:
    initial, _ = parity._initial_case(1)
    theta_raw = np.arctan2(matlab_gradient(initial["y"]), matlab_gradient(initial["x"]))
    if theta_raw.dtype != np.dtype(np.float64) or theta_raw.ndim != 1 or not theta_raw.flags.c_contiguous:
        raise TypeError("representative unwrap input must be a 1-D C-contiguous float64 array")
    signatures_before = [str(signature) for signature in kernel.signatures]
    started = time.perf_counter()
    accelerated = kernel(theta_raw)
    warmup_seconds = time.perf_counter() - started
    reference = mathutils_module.unwrap_angles_like_matlab(theta_raw)
    if not _same_exact(reference, accelerated):
        raise RuntimeError("production Numba warm-up output is not bitwise equal to the Python reference")
    live_type = numba_module.typeof(theta_raw)
    matching_signature = any(signature[0] == live_type for signature in kernel.signatures)
    if not matching_signature:
        raise RuntimeError(f"warm-up signature {live_type} does not match {kernel.signatures}")
    return {
        "warmup_seconds_excluded_from_solver_timings": float(warmup_seconds),
        "candidate_name": kernel.py_func.__name__,
        "representative_array": {
            "dtype": str(theta_raw.dtype),
            "ndim": int(theta_raw.ndim),
            "shape": list(theta_raw.shape),
            "C_contiguous": bool(theta_raw.flags.c_contiguous),
            "signature_type": str(live_type),
        },
        "compiled_signatures_before": signatures_before,
        "compiled_signatures_after": [str(signature) for signature in kernel.signatures],
        "nopython_signatures_after": [str(signature) for signature in kernel.nopython_signatures],
        "matching_signature_verified": bool(matching_signature),
        "warmup_output_bitwise_equal": True,
        "numba_target_options": {str(key): str(value) for key, value in kernel.targetoptions.items()},
        "persistent_numba_cache_enabled": bool(kernel.targetoptions.get("cache", False)),
        "compiled_in_this_process": not bool(signatures_before) and bool(kernel.signatures),
    }


def _run_variant(
    *,
    label: str,
    steps: int,
    checkpoints: set[int],
    progress_every: int,
    backend: str,
) -> dict[str, Any]:
    run = parity._run_one(
        label=label,
        case=1,
        steps=steps,
        checkpoints=checkpoints,
        flow_backend="numba",
        vertical_backend="numba",
        geometry_unwrap_backend=backend,
        progress_every=progress_every,
    )
    run["geometry_unwrap_public_path"] = {
        "selected_backend": backend,
        "run_case_option": run.get("geometry_unwrap_backend"),
        "run_config_option": run.get("run_config_options", {}).get("geometry_unwrap_backend"),
        "private_geometry_binding_patched": False,
    }
    live_inputs = []
    for _step, checkpoint in sorted(run["_checkpoints"].items()):
        arrays = checkpoint.get("_arrays", {})
        if arrays.get("x") is None or arrays.get("y") is None:
            continue
        live_inputs.append(
            np.arctan2(
                matlab_gradient(np.asarray(arrays["y"], dtype=np.float64)),
                matlab_gradient(np.asarray(arrays["x"], dtype=np.float64)),
            )
        )
    run["_unwrap_inputs"] = live_inputs
    return run


def _save_run(run: dict[str, Any], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    serial = parity._serial_run(run)
    serial["scalar_history"] = run["_scalar_history"]
    serial["cutoff_events"] = run["_cutoff_events"]
    (path / "run.json").write_text(json.dumps(parity._json_value(serial), indent=2), encoding="utf-8")
    scalar_checkpoints: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for step, checkpoint in sorted(run["_checkpoints"].items()):
        key = f"step_{step}"
        scalar_checkpoints[key] = {name: value for name, value in checkpoint.items() if name != "_arrays"}
        for name, array in checkpoint.get("_arrays", {}).items():
            if array is not None:
                arrays[f"{key}_{name}"] = np.asarray(array)
    (path / "checkpoint_scalars.json").write_text(
        json.dumps(parity._json_value(scalar_checkpoints), indent=2), encoding="utf-8"
    )
    np.savez_compressed(path / "checkpoint_arrays.npz", **arrays)
    if run.get("_unwrap_inputs"):
        np.savez_compressed(
            path / "checkpoint_unwrap_inputs.npz",
            **{f"theta_{index}": values for index, values in enumerate(run["_unwrap_inputs"])},
        )


def _paired_helper_sanity(kernel, reference_run: dict[str, Any]) -> dict[str, Any]:
    arrays = reference_run.get("_unwrap_inputs", [])
    if not arrays:
        return {"available": False, "reason": "no checkpoint geometries captured"}
    ref_seconds = 0.0
    numba_seconds = 0.0
    mismatches = 0
    signatures_before = [str(signature) for signature in kernel.signatures]
    for index, theta in enumerate(arrays):
        outputs = {}
        order = ("reference", "numba") if index % 2 == 0 else ("numba", "reference")
        for helper in order:
            start = time.perf_counter()
            output = (
                mathutils_module.unwrap_angles_like_matlab(theta)
                if helper == "reference"
                else kernel(theta)
            )
            elapsed = time.perf_counter() - start
            outputs[helper] = output
            if helper == "reference":
                ref_seconds += elapsed
            else:
                numba_seconds += elapsed
        if not _same_exact(outputs["reference"], outputs["numba"]):
            mismatches += 1
    signatures_after = [str(signature) for signature in kernel.signatures]
    return {
        "available": True,
        "input_angle_array_count": len(arrays),
        "reference_total_calls": len(arrays),
        "numba_total_calls": len(arrays),
        "reference_total_seconds": float(ref_seconds),
        "numba_total_seconds": float(numba_seconds),
        "speedup_reference_over_numba": ref_seconds / numba_seconds if numba_seconds else None,
        "bitwise_mismatching_arrays": mismatches,
        "helper_outputs_bitwise_equal": mismatches == 0,
        "alternating_order": True,
        "jit_signatures_before": signatures_before,
        "jit_signatures_after": signatures_after,
        "no_compilation_during_helper_timing": signatures_before == signatures_after,
        "input_source": "checkpoint centreline states from the production-path solver run",
        "supporting_evidence_only": True,
    }


def _timing(run: dict[str, Any], field: str) -> float:
    if field in {"total_wall_seconds", "flowfield_total_seconds"}:
        return float(run.get(field, 0.0) or 0.0)
    return float(run.get("solver_component_timings", {}).get(field, 0.0) or 0.0)


def _pair_result(reference: dict[str, Any], candidate: dict[str, Any], exact: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"exact_parity": exact["exact"]}
    for metric, field in (
        ("whole_solver", "total_wall_seconds"),
        ("geometry", "geometry"),
        ("curvature", "geometry_curvature"),
    ):
        ref = _timing(reference, field)
        cand = _timing(candidate, field)
        result[f"reference_{metric}_seconds"] = ref
        result[f"candidate_{metric}_seconds"] = cand
        result[f"{metric}_speedup"] = ref / cand if cand else None
    return result


def _environment() -> dict[str, str]:
    import llvmlite
    import numba
    import scipy

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
    }


def _write_outputs(output: Path, summary: dict[str, Any]) -> None:
    (output / "summary.json").write_text(
        json.dumps(parity._json_value(summary), indent=2), encoding="utf-8"
    )
    lines = [
        "# Production-path exact Numba curvature-unwrap A/B",
        "",
        f"Verdict: **{summary.get('verdict', 'IN_PROGRESS')}**",
        "",
        "The official `geometry_unwrap_backend` control was used through `run_case`; the runner did not patch the geometry unwrap binding. Flow and vertical backends were Numba for both variants, with identical serial/non-fastmath controls.",
        "",
        f"JIT warm-up: {summary['jit_warmup']['warmup_seconds_excluded_from_solver_timings']:.6f} s outside solver timings; live signature verified: {summary['jit_warmup']['matching_signature_verified']}.",
        "",
        "| Run | Unwrap backend | Steps | Wall (s) | Geometry (s) | Curvature (s) | Flowfield (s) | Final Ns | Cutoffs | Exactness |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for run in summary.get("runs", []):
        exact = run.get("exactness", {}).get("exact")
        lines.append(
            f"| {run['label']} | {run.get('geometry_unwrap_backend', 'n/a')} | {run['completed_steps']} | "
            f"{run['total_wall_seconds']:.4f} | {(_timing(run, 'geometry') or 0.0):.4f} | "
            f"{(_timing(run, 'geometry_curvature') or 0.0):.4f} | "
            f"{(_timing(run, 'flowfield_total_seconds') or 0.0):.4f} | "
            f"{run['final_Ns']} | {run['number_of_cutoffs']} | "
            f"{'yes' if exact else 'no' if exact is False else 'pending'} |"
        )
    for stage in ("preflight", "production_confirmation"):
        result = summary.get(stage)
        if result is None:
            continue
        lines.extend(["", f"## {stage.replace('_', ' ').title()}", ""])
        exact = result.get("exactness", {})
        lines.append(
            f"Exact parity: {exact.get('exact')}; mismatch count: {exact.get('mismatch_count')}; "
            f"both completed: {result.get('both_completed')}."
        )
        if result.get("pair"):
            pair = result["pair"]
            lines.append(
                f"Python / Numba wall: {pair['reference_whole_solver_seconds']:.4f} / "
                f"{pair['candidate_whole_solver_seconds']:.4f} s "
                f"({pair['whole_solver_speedup']:.4f}x); geometry: "
                f"{pair['reference_geometry_seconds']:.4f} / {pair['candidate_geometry_seconds']:.4f} s "
                f"({pair['geometry_speedup']:.4f}x); curvature: "
                f"{pair['reference_curvature_seconds']:.4f} / {pair['candidate_curvature_seconds']:.4f} s "
                f"({pair['curvature_speedup']:.4f}x)."
            )
    helper = summary.get("paired_helper_sanity", {})
    if helper.get("available"):
        lines.extend([
            "",
            f"Checkpoint helper sanity: {helper['reference_total_calls']} paired calls per path; "
            f"{helper['reference_total_seconds']:.6f} s Python vs {helper['numba_total_seconds']:.6f} s Numba "
            f"({helper['speedup_reference_over_numba']:.4f}x); bitwise equal={helper['helper_outputs_bitwise_equal']}.",
        ])
    lines.extend(["", "All timings are machine-specific and indicative. The five-pair benchmark remains the primary performance evidence.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse

    import numba

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{time.time_ns() % 1_000_000:06d}"
    output = args.results_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    kernel = _make_candidate()
    summary: dict[str, Any] = {
        "run_id": run_id,
        "case": 1,
        "environment": _environment(),
        "method": "official run_case geometry_unwrap_backend; no unwrap-binding monkeypatch",
        "solver_controls": {
            "flow_backend": "numba",
            "vertical_backend": "numba",
            "numba_parallel": False,
            "numba_fastmath": False,
            "flow_parallel": 0,
            "geometry_unwrap_backend": ["python", "numba"],
            "plots": False,
        },
        "runs": [],
        "verdict": "IN_PROGRESS",
    }
    summary["jit_warmup"] = _warm_candidate(kernel, numba)
    _write_outputs(output, summary)

    preflight_runs = []
    for label, backend in (("preflight_python", "python"), ("preflight_numba", "numba")):
        run = _run_variant(
            label=label,
            steps=1_000,
            checkpoints=set(PREFLIGHT_CHECKPOINTS),
            progress_every=args.progress_every,
            backend=backend,
        )
        preflight_runs.append(run)
        _save_run(run, output / "preflight_1000" / label)
        summary["runs"].append(parity._serial_run(run))
        print(
            f"[{label}] {run['completed_steps']}/1,000 steps, error={run['run_error']}, "
            f"wall={run['total_wall_seconds']:.3f}s, Ns={run['final_Ns']}, "
            f"cutoffs={run['number_of_cutoffs']}, geometry backend={run['geometry_unwrap_backend']}",
            flush=True,
        )
    preflight_exactness = _exact_comparison(*preflight_runs)
    for run in summary["runs"][-2:]:
        run["exactness"] = preflight_exactness
    preflight_completed = all(
        run["run_error"] is None and run["completed_steps"] == 1_000 for run in preflight_runs
    )
    summary["preflight"] = {
        "requested_steps": 1_000,
        "both_completed": preflight_completed,
        "exactness": preflight_exactness,
        "expected_trajectory": {
            "python_Ns": preflight_runs[0]["final_Ns"],
            "numba_Ns": preflight_runs[1]["final_Ns"],
            "python_cutoffs": preflight_runs[0]["number_of_cutoffs"],
            "numba_cutoffs": preflight_runs[1]["number_of_cutoffs"],
            "python_sinuosity": preflight_runs[0]["final_sinuosity"],
            "numba_sinuosity": preflight_runs[1]["final_sinuosity"],
        },
    }
    summary["paired_helper_sanity"] = _paired_helper_sanity(kernel, preflight_runs[0])
    (output / "paired_helper_sanity.json").write_text(
        json.dumps(parity._json_value(summary["paired_helper_sanity"]), indent=2), encoding="utf-8"
    )
    _write_outputs(output, summary)
    if not (preflight_completed and preflight_exactness["exact"]):
        summary["verdict"] = "REJECT_PRODUCTION_UNWRAP_PREFLIGHT"
        summary["stop_reason"] = "The public-path 1,000-step comparison did not complete exactly."
        _write_outputs(output, summary)
        return 1

    confirmation_runs = []
    for label, backend in (("confirmation_python", "python"), ("confirmation_numba", "numba")):
        run = _run_variant(
            label=label,
            steps=5_000,
            checkpoints=set(CONFIRMATION_CHECKPOINTS),
            progress_every=args.progress_every,
            backend=backend,
        )
        confirmation_runs.append(run)
        _save_run(run, output / "confirmation_5000" / label)
        summary["runs"].append(parity._serial_run(run))
        print(
            f"[{label}] {run['completed_steps']}/5,000 steps, error={run['run_error']}, "
            f"wall={run['total_wall_seconds']:.3f}s, Ns={run['final_Ns']}, "
            f"cutoffs={run['number_of_cutoffs']}, geometry backend={run['geometry_unwrap_backend']}",
            flush=True,
        )
    confirmation_exactness = _exact_comparison(*confirmation_runs)
    for run in summary["runs"][-2:]:
        run["exactness"] = confirmation_exactness
    confirmation_completed = all(
        run["run_error"] is None and run["completed_steps"] == 5_000 for run in confirmation_runs
    )
    pair = _pair_result(confirmation_runs[0], confirmation_runs[1], confirmation_exactness)
    summary["production_confirmation"] = {
        "requested_steps": 5_000,
        "both_completed": confirmation_completed,
        "exactness": confirmation_exactness,
        "pair": pair,
        "python_final_Ns": confirmation_runs[0]["final_Ns"],
        "numba_final_Ns": confirmation_runs[1]["final_Ns"],
        "python_final_cutoffs": confirmation_runs[0]["number_of_cutoffs"],
        "numba_final_cutoffs": confirmation_runs[1]["number_of_cutoffs"],
        "python_final_sinuosity": confirmation_runs[0]["final_sinuosity"],
        "numba_final_sinuosity": confirmation_runs[1]["final_sinuosity"],
    }
    summary["verdict"] = (
        "PRODUCTION_PUBLIC_PATHS_EXACT"
        if confirmation_completed and confirmation_exactness["exact"]
        else "REJECT_PRODUCTION_UNWRAP"
    )
    if summary["verdict"] != "PRODUCTION_PUBLIC_PATHS_EXACT":
        summary["stop_reason"] = "The public-path 5,000-step comparison did not complete exactly."
    _write_outputs(output, summary)
    print(f"{summary['verdict']}; evidence={output}", flush=True)
    return 0 if summary["verdict"] == "PRODUCTION_PUBLIC_PATHS_EXACT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
