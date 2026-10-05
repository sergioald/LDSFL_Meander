"""Public-path A/B validation for the optional Numba grid neck detector.

This runner observes solver calls and saved cut output, but always routes neck
search through ``run_case(neck_detector_backend=...)`` without replacing the
detector. Results are placed in a timestamped ignored ``benchmarks/Output``
directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldsfl import geometry as geometry_module  # noqa: E402
from ldsfl import main as main_module  # noqa: E402
from ldsfl.main import run_case  # noqa: E402

COMMON_CONTROLS = {
    "Nprint": 10000,
    "Ntstep": 100000,
    "Max_Cut": 0,
    "stop_on_steps": True,
    "stop_on_cutoffs": False,
    "flow_backend": "numba",
    "geometry_unwrap_backend": "numba",
    "flow_paral": 0,
    "flow_workers": 0,
    "numba_parallel": False,
    "numba_fastmath": False,
    "collect_timing": True,
    "do_plots": False,
    "cstab": 0.01,
}
EXPECTED_TRAJECTORIES = {
    1000: {"steps": 1000, "Ns": 1187, "cutoffs": 0, "sinuosity": 1.1917102476657917},
    5000: {"steps": 5000, "Ns": 2359, "cutoffs": 44, "sinuosity": 1.7245401220504077},
}


def _checkpoint_steps(steps: int) -> tuple[int, ...]:
    if steps == 1000:
        return (0, 1, 10, 100, 500, 1000)
    return (0, 1, 10, 100, 500, 1000, 2500, 5000)


def _copy_inputs(destination: Path) -> None:
    shutil.copytree(REPO_ROOT / "Input", destination / "Input")


def _case_history(base_dir: Path, result: dict) -> np.ndarray:
    paths = list((base_dir / "Output" / result["id_files"] / "files").glob("var_*.csv"))
    if not paths:
        raise RuntimeError("The public run did not write variable history")
    latest = max(paths, key=lambda path: path.stat().st_mtime_ns)
    return np.genfromtxt(latest, delimiter=",", names=True, dtype=np.float64)


def _final_point_count(base_dir: Path, result: dict) -> int:
    paths = list((base_dir / "Output" / result["id_files"] / "xyu").glob("xyu_*.csv"))
    if not paths:
        raise RuntimeError("The public run did not write a final centreline snapshot")
    latest = max(paths, key=lambda path: path.stat().st_mtime_ns)
    with latest.open(encoding="utf-8") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def _run_public_case(base_dir: Path, steps: int, neck_backend: str) -> dict:
    _copy_inputs(base_dir)
    wanted = set(_checkpoint_steps(steps))
    observation = {
        "current_step": 0,
        "flow_calls": 0,
        "events": [],
        "checkpoints": {},
        "flags": {},
    }

    original_preprof = main_module.preprof_3
    original_flow = main_module.parall_u_free
    original_geometry4 = main_module.geometry4
    original_save_cut = geometry_module.save_xy_cut

    def capture_initial(*args, **kwargs):
        values = original_preprof(*args, **kwargs)
        s, _x, _y, theta, point_count, deltas, wave, valley, sinuosity = values
        observation["checkpoints"][0] = {
            "s": np.asarray(s).copy(),
            "x": np.asarray(_x).copy(),
            "y": np.asarray(_y).copy(),
            "theta": np.asarray(theta).copy(),
            "curvature": main_module.initial_curvature(theta, deltas).copy(),
            "U": np.zeros_like(_x, dtype=np.float64),
            "Ns": int(point_count),
            "sinuosity": float(sinuosity),
            "cut_cnt": 0,
            "wave": float(wave),
            "valley": float(valley),
        }
        return values

    def capture_flow(*args, **kwargs):
        U, flag = original_flow(*args, **kwargs)
        observation["flow_calls"] += 1
        step = observation["flow_calls"]
        if step in wanted:
            observation["flags"][step] = flag.item() if isinstance(flag, np.generic) else flag
            observation["checkpoints"].setdefault(step, {})["U"] = np.asarray(U).copy()
        return U, flag

    def capture_cut_file(*args, **kwargs):
        # Observe the normal geometry output call; detector routing stays public.
        xa = np.asarray(args[1])
        point_count = int(xa.size)
        i1, aa, jt, cut_count, ss = map(int, (args[3], args[4], args[6], args[8], args[9]))
        i0 = i1 - 1
        j0 = i0 + ss + aa - 1
        start = i1 - 1
        end_mat = ss + aa + i1 + 1
        end_inclusive = min(end_mat - 1, point_count - 1)
        removed = max(0, end_inclusive - start + 1)
        observation["events"].append(
            {
                "step": int(observation["current_step"] or jt),
                "cut_cnt": cut_count,
                "Ns_before": point_count,
                "Ns_after": point_count - removed,
                "i0": i0,
                "j0": j0,
                "points_removed": removed,
            }
        )
        return original_save_cut(*args, **kwargs)

    def capture_geometry(x, y, jt, *args, **kwargs):
        observation["current_step"] = int(jt)
        values = original_geometry4(x, y, jt, *args, **kwargs)
        if int(jt) in wanted:
            curvature, s, x_new, y_new, theta, Ns, deltas, wave, valley, sinuosity, cut_count = values
            observation["checkpoints"].setdefault(int(jt), {}).update(
                {
                    "curvature": np.asarray(curvature).copy(),
                    "s": np.asarray(s).copy(),
                    "x": np.asarray(x_new).copy(),
                    "y": np.asarray(y_new).copy(),
                    "theta": np.asarray(theta).copy(),
                    "Ns": int(Ns),
                    "deltas": float(deltas),
                    "wave": float(wave),
                    "valley": float(valley),
                    "sinuosity": float(sinuosity),
                    "cut_cnt": int(cut_count),
                }
            )
            for event in observation["events"]:
                if event["step"] == int(jt) and event.get("Ns_after") is not None:
                    event["Ns_after_geometry"] = int(Ns)
        return values

    main_module.preprof_3 = capture_initial
    main_module.parall_u_free = capture_flow
    main_module.geometry4 = capture_geometry
    geometry_module.save_xy_cut = capture_cut_file
    started = perf_counter()
    try:
        result = run_case(
            base_dir,
            1,
            max_steps=steps,
            **COMMON_CONTROLS,
            neck_detector_backend=neck_backend,
        )
    finally:
        main_module.preprof_3 = original_preprof
        main_module.parall_u_free = original_flow
        main_module.geometry4 = original_geometry4
        geometry_module.save_xy_cut = original_save_cut
    wall_seconds = perf_counter() - started
    history = _case_history(base_dir, result)
    status_path = base_dir / "Output" / result["id_files"] / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("simulation_completed") or not status.get("output_complete"):
        raise RuntimeError(f"Public solver run did not complete cleanly: {status}")
    return {
        "label": neck_backend,
        "base_dir": str(base_dir),
        "result": result,
        "wall_seconds": float(wall_seconds),
        "history": history,
        "events": observation["events"],
        "checkpoints": observation["checkpoints"],
        "flags": observation["flags"],
        "point_count": _final_point_count(base_dir, result),
        "id_files": result["id_files"],
    }


def _scalar_checkpoint(run: dict, step: int) -> dict:
    rows = run["history"][run["history"]["state_step"] == float(step)]
    if rows.size != 1:
        raise RuntimeError(f"Expected one scalar row at step {step}, got {rows.size}")
    row = rows[0]
    fields = ("Cf0", "beta", "theta0", "ds", "dt", "dt_cum", "cut_cnt", "sinuo")
    return {field: float(row[field]) for field in fields}


def _array_difference(left, right) -> dict:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return {"compatible_grid": False, "shape_reference": left.shape, "shape_candidate": right.shape}
    delta = left - right
    return {
        "compatible_grid": True,
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rms": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
        "exact": bool(np.array_equal(left, right, equal_nan=True)),
    }


def _cutoff_steps(run: dict) -> list[int]:
    return [int(event["step"]) for event in run["events"]]


def _compare(reference: dict, candidate: dict, steps: int) -> dict:
    checkpoints = {}
    checkpoint_exact = True
    for step in _checkpoint_steps(steps):
        ref = reference["checkpoints"].get(step)
        cand = candidate["checkpoints"].get(step)
        if ref is None or cand is None:
            checkpoints[str(step)] = {"exact": False, "reason": "missing checkpoint"}
            checkpoint_exact = False
            continue
        arrays = {
            field: _array_difference(ref[field], cand[field])
            for field in ("x", "y", "s", "curvature", "theta", "U")
        }
        scalar_ref = _scalar_checkpoint(reference, step)
        scalar_cand = _scalar_checkpoint(candidate, step)
        exact = (
            all(item.get("exact", False) for item in arrays.values())
            and ref["Ns"] == cand["Ns"]
            and ref["cut_cnt"] == cand["cut_cnt"]
            and ref["sinuosity"] == cand["sinuosity"]
            and scalar_ref == scalar_cand
            and reference["flags"].get(step) == candidate["flags"].get(step)
        )
        checkpoint_exact = checkpoint_exact and exact
        checkpoints[str(step)] = {
            "exact": bool(exact),
            "Ns_reference": int(ref["Ns"]),
            "Ns_candidate": int(cand["Ns"]),
            "array_differences": arrays,
            "scalar_reference": scalar_ref,
            "scalar_candidate": scalar_cand,
            "resonance_flag_reference": reference["flags"].get(step),
            "resonance_flag_candidate": candidate["flags"].get(step),
        }

    scalar_fields = reference["history"].dtype.names or ()
    scalar_history_exact = scalar_fields == (candidate["history"].dtype.names or ()) and all(
        np.array_equal(reference["history"][name], candidate["history"][name], equal_nan=True)
        for name in scalar_fields
    )
    events_exact = reference["events"] == candidate["events"]
    flags_exact = reference["flags"] == candidate["flags"]
    return {
        "exact": bool(checkpoint_exact and scalar_history_exact and events_exact and flags_exact),
        "checkpoint_exact": bool(checkpoint_exact),
        "scalar_history_exact": bool(scalar_history_exact),
        "cutoff_events_exact": bool(events_exact),
        "cutoff_steps_reference": _cutoff_steps(reference),
        "cutoff_steps_candidate": _cutoff_steps(candidate),
        "selected_pairs_exact": [
            (event["i0"], event["j0"]) for event in reference["events"]
        ]
        == [(event["i0"], event["j0"]) for event in candidate["events"]],
        "points_removed_exact": [event["points_removed"] for event in reference["events"]]
        == [event["points_removed"] for event in candidate["events"]],
        "resonance_flags_exact": bool(flags_exact),
        "reference_final_Ns": int(reference["point_count"]),
        "candidate_final_Ns": int(candidate["point_count"]),
        "reference_final_sinuosity": float(reference["result"]["sinuo_final"]),
        "candidate_final_sinuosity": float(candidate["result"]["sinuo_final"]),
        "reference_cutoffs": int(reference["result"]["cut_cnt"]),
        "candidate_cutoffs": int(candidate["result"]["cut_cnt"]),
        "checkpoints": checkpoints,
    }


def _run_summary(run: dict) -> dict:
    result = run["result"]
    return {
        "neck_detector_backend": run["label"],
        "wall_seconds": run["wall_seconds"],
        "timings_seconds": result["timings"],
        "steps": int(result["steps"]),
        "Ns": int(run["point_count"]),
        "cutoffs": int(result["cut_cnt"]),
        "sinuosity": float(result["sinuo_final"]),
        "resonance": result["resonance"],
        "run_dir": str(Path(run["base_dir"]) / "Output" / run["id_files"]),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(args) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = REPO_ROOT / "benchmarks" / "Output" / "neck_spatial_grid_numba_public" / f"{args.steps}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    order = ("numba_grid", "kdtree") if args.steps == 5000 else ("kdtree", "numba_grid")
    runs = {}
    for backend in order:
        print(f"Running {args.steps}-step public {backend} leg...", flush=True)
        runs[backend] = _run_public_case(output_dir / backend, args.steps, backend)

    comparison = _compare(runs["kdtree"], runs["numba_grid"], args.steps)
    expected = EXPECTED_TRAJECTORIES[args.steps]
    expected_trajectory_matches = all(
        _run_summary(run)[key] == expected[value_key]
        for run in runs.values()
        for key, value_key in (
            ("steps", "steps"),
            ("Ns", "Ns"),
            ("cutoffs", "cutoffs"),
            ("sinuosity", "sinuosity"),
        )
    )
    output = {
        "study": "public production neck detector backend validation",
        "steps_requested": int(args.steps),
        "execution_order": list(order),
        "controls": {**COMMON_CONTROLS, "max_steps": int(args.steps)},
        "reference": _run_summary(runs["kdtree"]),
        "candidate": _run_summary(runs["numba_grid"]),
        "comparison": comparison,
        "expected_trajectory": expected,
        "expected_trajectory_matches": bool(expected_trajectory_matches),
        "run_directories": {backend: _run_summary(run)["run_dir"] for backend, run in runs.items()},
    }

    if args.steps == 5000:
        profile_dir = output_dir / "candidate_profile"
        print("Running one fresh public 5,000-step candidate profile...", flush=True)
        profile = _run_public_case(profile_dir, args.steps, "numba_grid")
        timings = profile["result"]["timings"]
        profile_shares = {
            key: float(timings[key]) / profile["wall_seconds"]
            for key in (
                "flowfield_total",
                "semiana_response",
                "modal_coefficients",
                "geometry",
                "geometry_neck",
                "geometry_neck_detector",
                "geometry_initial_uniformization",
                "geometry_curvature",
            )
        }
        output["post_integration_profile"] = {
            "run": _run_summary(profile),
            "wall_shares": profile_shares,
            "largest_reported_component": max(profile_shares, key=profile_shares.get),
        }

    _write_json(output_dir / "summary.json", output)
    print(json.dumps({
        "steps": args.steps,
        "execution_order": list(order),
        "reference": output["reference"],
        "candidate": output["candidate"],
        "comparison": {
            key: value
            for key, value in comparison.items()
            if key != "checkpoints"
        },
        "expected_trajectory_matches": output["expected_trajectory_matches"],
        "post_integration_profile": output.get("post_integration_profile"),
        "results_directory": str(output_dir),
    }, indent=2, sort_keys=True))
    print(f"\nResults: {output_dir}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=(1000, 5000), required=True)
    args = parser.parse_args()
    try:
        output = _run(args)
    except Exception as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    return 0 if output["comparison"]["exact"] and output["expected_trajectory_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
