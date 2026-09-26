"""Long-run numerical and performance comparison for solver backends.

Run from the repository root, for example::

    python benchmarks/long_term_backend_parity.py --steps 10000 --case 1 \
        --comparison vertical-only

Generated results are written to the ignored ``benchmarks/Output`` tree.
The runner observes ``run_case`` by temporarily wrapping its module-level
functions; it does not edit solver code or change solver equations/settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldsfl import geometry as geometry_module  # noqa: E402
from ldsfl import main as main_module  # noqa: E402
from ldsfl.flowfield import parall_u_free as raw_parall_u_free  # noqa: E402
from ldsfl.inputs import dimensionless_input_table, read_parameter_table, read_xy  # noqa: E402
from ldsfl.profile import preprof_3  # noqa: E402
from ldsfl.resistance import resistance_function_flagbed  # noqa: E402
from ldsfl.resonance import fundamental_decay_rate  # noqa: E402
from ldsfl.vertical import (  # noqa: E402
    clear_k0123_reference_cache,
    k0123_reference_cache_info,
)
from ldsfl.vertical_numba import (  # noqa: E402
    clear_k0123_cache,
    k0123_cache_info,
)

DEFAULT_CHECKPOINTS = (0, 1, 10, 100, 1_000, 5_000, 10_000, 25_000, 50_000, 75_000, 100_000)
SCALAR_FIELDS = ("sinuosity", "beta", "theta0", "ds", "dt_cum")
THRESHOLDS = (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "Output" / "long_term_results"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sinuosity_xy(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    chord = float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
    return length / chord if chord > 0.0 else float("inf")


def _initial_case(case: int) -> tuple[dict[str, Any], dict[str, str]]:
    input_dir = ROOT / "Input"
    params = read_parameter_table(input_dir / "Parameter.csv")
    beta, ds, theta0, flagbed, rpic_0, mdat = dimensionless_input_table(params, case)
    x_raw, y_raw = read_xy(input_dir / "xy.csv")
    s, x, y, theta, ns, deltas, wave_l, valle_l, sinuosity = preprof_3(x_raw, y_raw, 1.0)
    curvature = main_module.initial_curvature(theta, deltas)
    rpic, cf0, ct, cd, phi_t, phi_d, f0 = resistance_function_flagbed(flagbed, theta0, ds, rpic_0)
    state = {
        "step": 0,
        "jt": 1,
        "dt": 0.0,
        "dt_cum": 0.0,
        "Ns": int(ns),
        "cut_cnt": 0,
        "sinuosity": float(sinuosity),
        "beta": float(beta),
        "theta0": float(theta0),
        "ds": float(ds),
        "Cf0": float(cf0),
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "s": np.asarray(s, dtype=np.float64),
        "c": np.asarray(curvature, dtype=np.float64),
        "th": np.asarray(theta, dtype=np.float64),
        "deltas": float(deltas),
        "flow_flag": None,
        "parameters": {
            "flagbed": int(flagbed),
            "rpic_0": float(rpic_0),
            "Mdat": int(mdat),
            "rpic": float(rpic),
            "CT": float(ct),
            "CD": float(cd),
            "phiT": float(phi_t),
            "phiD": float(phi_d),
            "F0": float(f0),
        },
    }
    hashes = {
        "Parameter.csv": _sha256(input_dir / "Parameter.csv"),
        "xy.csv": _sha256(input_dir / "xy.csv"),
    }
    return state, hashes


def _cache_functions(backend: str):
    if backend == "numba":
        return k0123_cache_info, clear_k0123_cache
    return k0123_reference_cache_info, clear_k0123_reference_cache


def _checkpoint_record(state: dict[str, Any]) -> dict[str, Any]:
    record = {
        key: state[key]
        for key in (
            "step", "jt", "dt", "dt_cum", "Ns", "cut_cnt", "sinuosity", "beta",
            "theta0", "ds", "Cf0", "deltas", "flow_flag",
        )
    }
    record["max_abs_curvature"] = float(np.max(np.abs(state["c"]))) if state["c"].size else 0.0
    record["mean_U"] = None
    record["max_U"] = None
    record["max_abs_U"] = None
    record["resonance_decay_rate"] = None
    record["resonance_state"] = None
    record["_arrays"] = {
        key: np.asarray(state[key], dtype=np.float64).copy()
        for key in ("x", "y", "s", "c")
    }
    record["_arrays"]["U"] = None
    return record


class RunRecorder:
    """Collect per-step data while ``run_case`` executes, using wrappers only."""

    def __init__(self, initial: dict[str, Any], checkpoints: set[int], progress_every: int, label: str):
        self.state = initial
        self.checkpoints = checkpoints
        self.progress_every = max(0, int(progress_every))
        self.label = label
        self.completed = 0
        self.dt_cum = 0.0
        self.last_dt = 0.0
        self.pending_geometry: dict[str, Any] | None = None
        self.latest_resistance = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_final = {"hits": 0, "misses": 0, "currsize": 0, "maxsize": 16}
        self.vertical_coefficients_seconds = 0.0
        self.flow_calls = 0
        self.checkpoint_states: dict[int, dict[str, Any]] = {}
        self.scalar_history: list[dict[str, Any]] = []
        self.cutoff_events: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._remember_initial()

    def _remember_initial(self) -> None:
        self.scalar_history.append(self._scalar_row(self.state))
        if 0 in self.checkpoints:
            self.checkpoint_states[0] = _checkpoint_record(self.state)

    @staticmethod
    def _scalar_row(state: dict[str, Any]) -> dict[str, Any]:
        return {
            key: state[key]
            for key in (
                "step", "jt", "dt", "dt_cum", "Ns", "cut_cnt", "sinuosity", "beta",
                "theta0", "ds", "Cf0",
            )
        }

    def wrap_flow(self, original, vertical_backend: str):
        cache_info, _ = _cache_functions(vertical_backend)

        def wrapped(*args, **kwargs):
            # run_case does not expose vertical_backend; inject it only at this
            # flowfield boundary to isolate the vertical path without changing
            # the solver's controls or implementation.
            kwargs["vertical_backend"] = vertical_backend
            before = cache_info()
            timing = kwargs.get("timing")
            if timing is None:
                timing = {}
                kwargs["timing"] = timing
            result = original(*args, **kwargs)
            after = cache_info()
            self.cache_hits += int(after.hits - before.hits)
            self.cache_misses += int(after.misses - before.misses)
            self.cache_final = {
                "hits": int(after.hits),
                "misses": int(after.misses),
                "currsize": int(after.currsize),
                "maxsize": int(after.maxsize),
            }
            self.vertical_coefficients_seconds += float(timing.get("vertical_coefficients", 0.0))
            self.flow_calls += 1
            try:
                velocity = np.asarray(result[0], dtype=np.float64)
                flow_flag = result[1]
                step = int(self.completed)
                self.state["flow_flag"] = _json_value(flow_flag)
                snapshot = self.checkpoint_states.get(step)
                if snapshot is not None:
                    snapshot["flow_flag"] = _json_value(flow_flag)
                    snapshot["mean_U"] = float(np.mean(velocity)) if velocity.size else float("nan")
                    snapshot["max_U"] = float(np.max(velocity)) if velocity.size else float("nan")
                    snapshot["max_abs_U"] = float(np.max(np.abs(velocity))) if velocity.size else 0.0
                    snapshot["_arrays"]["U"] = velocity.copy()
            except Exception as exc:  # diagnostic capture should not stop the solver
                self.errors.append(f"flow snapshot at step {self.completed}: {exc}")
            return result

        return wrapped

    def wrap_dxdy(self, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            self.last_dt = float(result[2])
            self.dt_cum += self.last_dt
            return result

        return wrapped

    def wrap_geometry(self, original):
        def wrapped(*args, **kwargs):
            jt = int(args[2] if len(args) > 2 else kwargs["jt"])
            before = self.state
            result = original(*args, **kwargs)
            c, s, x, y, th, ns, deltas, wave_l, valle_l, sinuosity, cut_cnt = result
            self.pending_geometry = {
                "jt": jt,
                "c": np.asarray(c, dtype=np.float64),
                "s": np.asarray(s, dtype=np.float64),
                "x": np.asarray(x, dtype=np.float64),
                "y": np.asarray(y, dtype=np.float64),
                "th": np.asarray(th, dtype=np.float64),
                "Ns": int(ns),
                "deltas": float(deltas),
                "wave_l": float(wave_l),
                "valle_l": float(valle_l),
                "sinuosity": float(sinuosity),
                "cut_cnt": int(cut_cnt),
                "Ns_before": int(before["Ns"]),
                "sinuosity_before": float(before["sinuosity"]),
            }
            for event in self.cutoff_events:
                if event["step"] == jt and event.get("Ns_after_geometry") is None:
                    event["Ns_after_geometry"] = int(ns)
                    event["sinuosity_after_geometry"] = float(sinuosity)
                    event["cut_cnt_after_geometry"] = int(cut_cnt)
            return result

        return wrapped

    def wrap_resistance(self, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            self.latest_resistance = tuple(float(value) for value in result)
            return result

        return wrapped

    def wrap_update(self, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            geometry = self.pending_geometry
            if geometry is None:
                return result
            beta, theta0, ds = (float(value) for value in result)
            cf0 = self.latest_resistance[1] if self.latest_resistance is not None else self.state["Cf0"]
            self.completed += 1
            state = {
                **geometry,
                "step": self.completed,
                "jt": int(geometry["jt"] + 1),
                "dt": self.last_dt,
                "dt_cum": self.dt_cum,
                "beta": beta,
                "theta0": theta0,
                "ds": ds,
                "Cf0": float(cf0),
                "flow_flag": self.state.get("flow_flag"),
                "parameters": self.state["parameters"],
            }
            self.state = state
            self.scalar_history.append(self._scalar_row(state))
            if self.completed in self.checkpoints:
                self.checkpoint_states[self.completed] = _checkpoint_record(state)
            self.pending_geometry = None
            if self.progress_every and (self.completed % self.progress_every == 0):
                print(
                    f"[{self.label}] completed {self.completed:,} steps; "
                    f"Ns={state['Ns']}, cutoffs={state['cut_cnt']}, sinuosity={state['sinuosity']:.9g}",
                    flush=True,
                )
            return result

        return wrapped

    def wrap_save_cut(self, original):
        def wrapped(*args, **kwargs):
            xa = np.asarray(args[1] if len(args) > 1 else kwargs["xa"], dtype=np.float64)
            ya = np.asarray(args[2] if len(args) > 2 else kwargs["ya"], dtype=np.float64)
            i1 = int(args[3] if len(args) > 3 else kwargs["i1"])
            aa = int(args[4] if len(args) > 4 else kwargs["AA"])
            jt = int(args[6] if len(args) > 6 else kwargs["jt"])
            cut_cnt = int(args[8] if len(args) > 8 else kwargs["cut_cnt"])
            ss = int(args[9] if len(args) > 9 else kwargs["ss"])
            start = i1 - 1
            end_inclusive = min(ss + aa + i1, xa.size - 1)
            mask = np.ones(xa.size, dtype=bool)
            if end_inclusive >= start:
                mask[start : end_inclusive + 1] = False
            x_after = xa[mask]
            y_after = ya[mask]
            event = {
                "step": jt,
                "dt_cum": float(self.dt_cum),
                "cut_cnt": cut_cnt,
                "Ns_before": int(xa.size),
                "Ns_after": int(x_after.size),
                "Ns_after_geometry": None,
                "sinuosity_before": _sinuosity_xy(xa, ya),
                "sinuosity_after_cut": _sinuosity_xy(x_after, y_after),
                "sinuosity_after_geometry": None,
                "cut_cnt_after_geometry": None,
            }
            self.cutoff_events.append(event)
            return original(*args, **kwargs)

        return wrapped


def _warm_flow(state: dict[str, Any], backend: str, vertical_backend: str) -> float:
    p = state["parameters"]
    start = time.perf_counter()
    raw_parall_u_free(
        state["c"], state["s"], state["Cf0"], p["CT"], p["CD"], p["phiT"], p["phiD"],
        state["beta"], p["rpic"], state["theta0"], p["F0"], p["Mdat"], 1, state["Ns"],
        np.array([1.0], dtype=np.float64), state["deltas"], SL=0, paral=0, n_workers=None,
        backend=backend, vertical_backend=vertical_backend,
        numba_parallel=False, numba_fastmath=False,
    )
    return time.perf_counter() - start


def _run_one(
    *,
    label: str,
    case: int,
    steps: int,
    checkpoints: set[int],
    flow_backend: str,
    vertical_backend: str,
    progress_every: int,
) -> dict[str, Any]:
    initial, input_hashes = _initial_case(case)
    warm_seconds = _warm_flow(initial, flow_backend, vertical_backend)
    _, clear_cache = _cache_functions(vertical_backend)
    clear_cache()
    recorder = RunRecorder(initial, checkpoints, progress_every, label)

    originals = {
        "parall_u_free": main_module.parall_u_free,
        "dxdy2": main_module.dxdy2,
        "geometry4": main_module.geometry4,
        "resistance_function_flagbed": main_module.resistance_function_flagbed,
        "update_parameters": main_module.update_parameters,
        "save_xy_cut": geometry_module.save_xy_cut,
    }
    main_module.parall_u_free = recorder.wrap_flow(originals["parall_u_free"], vertical_backend)
    main_module.dxdy2 = recorder.wrap_dxdy(originals["dxdy2"])
    main_module.geometry4 = recorder.wrap_geometry(originals["geometry4"])
    main_module.resistance_function_flagbed = recorder.wrap_resistance(originals["resistance_function_flagbed"])
    main_module.update_parameters = recorder.wrap_update(originals["update_parameters"])
    geometry_module.save_xy_cut = recorder.wrap_save_cut(originals["save_xy_cut"])

    run_result = None
    run_error = None
    wall_start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"ldsfl-long-parity-{label}-") as temp_name:
            project_dir = Path(temp_name)
            (project_dir / "Input").mkdir()
            copy2(ROOT / "Input" / "Parameter.csv", project_dir / "Input" / "Parameter.csv")
            copy2(ROOT / "Input" / "xy.csv", project_dir / "Input" / "xy.csv")
            run_result = main_module.run_case(
                project_dir,
                case,
                Nprint=steps + 2,
                flow_backend=flow_backend,
                numba_parallel=False,
                numba_fastmath=False,
                max_steps=steps,
                stop_on_steps=True,
                do_plots=False,
                collect_timing=True,
            )
    except Exception as exc:
        run_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        wall_seconds = time.perf_counter() - wall_start
        main_module.parall_u_free = originals["parall_u_free"]
        main_module.dxdy2 = originals["dxdy2"]
        main_module.geometry4 = originals["geometry4"]
        main_module.resistance_function_flagbed = originals["resistance_function_flagbed"]
        main_module.update_parameters = originals["update_parameters"]
        geometry_module.save_xy_cut = originals["save_xy_cut"]

    timings = (run_result or {}).get("timings") or {}
    if recorder.completed in recorder.checkpoints and recorder.completed not in recorder.checkpoint_states:
        recorder.checkpoint_states[recorder.completed] = _checkpoint_record(recorder.state)
    for _step, snapshot in recorder.checkpoint_states.items():
        try:
            decay = fundamental_decay_rate(
                float(snapshot["beta"]),
                float(snapshot["theta0"]),
                float(snapshot["ds"]),
                initial["parameters"]["rpic_0"],
                initial["parameters"]["flagbed"],
                initial["parameters"]["Mdat"],
            )
            snapshot["resonance_decay_rate"] = float(decay)
            snapshot["resonance_state"] = (
                "sub-resonant" if decay < 0.0 else "super-resonant" if decay > 0.0 else "resonant"
            )
        except Exception as exc:
            snapshot["resonance_error"] = str(exc)

    run_steps = int((run_result or {}).get("steps", recorder.completed))
    return {
        "label": label,
        "flow_backend": flow_backend,
        "vertical_backend": vertical_backend,
        "input_sha256": input_hashes,
        "warmup_flowfield_seconds": float(warm_seconds),
        "total_wall_seconds": float(wall_seconds),
        "flowfield_loop_seconds": timings.get("flowfield_loop"),
        "flowfield_final_seconds": timings.get("flowfield_final"),
        "flowfield_total_seconds": timings.get("flowfield_total"),
        "flowfield_legacy_loop_seconds": timings.get("flowfield"),
        "vertical_coefficient_seconds_all_flow_calls": float(recorder.vertical_coefficients_seconds),
        "k0123_cache_hits": int(recorder.cache_hits),
        "k0123_cache_misses": int(recorder.cache_misses),
        "k0123_cache_final": recorder.cache_final,
        "flowfield_calls": recorder.flow_calls,
        "completed_steps": int(run_steps),
        "recorded_completed_steps": int(recorder.completed),
        "number_of_cutoffs": len(recorder.cutoff_events),
        "final_cut_cnt": int((run_result or {}).get("cut_cnt", recorder.state["cut_cnt"])),
        "final_Ns": int(recorder.state["Ns"]),
        "final_sinuosity": float(recorder.state["sinuosity"]),
        "stop_reason": (run_result or {}).get("stop_reason"),
        "run_error": run_error,
        "capture_errors": recorder.errors,
        "_checkpoints": recorder.checkpoint_states,
        "_scalar_history": recorder.scalar_history,
        "_cutoff_events": recorder.cutoff_events,
    }


def _array_metrics(a: np.ndarray, b: np.ndarray, name: str) -> dict[str, float] | None:
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return None
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    absolute = np.abs(delta)
    return {
        f"max_abs_{name}_difference": float(np.max(absolute)),
        f"rms_{name}_difference": float(np.sqrt(np.mean(delta * delta))),
    }


def _checkpoint_comparison(reference: dict[str, Any], candidate: dict[str, Any], checkpoints: list[int]):
    rows = []
    details = {}
    ref_snaps = reference["_checkpoints"]
    cand_snaps = candidate["_checkpoints"]
    for step in checkpoints:
        ref = ref_snaps.get(step)
        cand = cand_snaps.get(step)
        row: dict[str, Any] = {"step": step, "both_present": ref is not None and cand is not None}
        detail: dict[str, Any] = {"step": step, "both_present": ref is not None and cand is not None}
        for side, snapshot in (("reference", ref), ("candidate", cand)):
            for field in (
                "jt", "dt", "dt_cum", "Ns", "cut_cnt", "sinuosity", "beta", "theta0", "ds", "Cf0",
                "deltas", "max_abs_curvature", "mean_U", "max_U", "max_abs_U", "flow_flag",
                "resonance_state", "resonance_decay_rate",
            ):
                row[f"{field}_{side}"] = snapshot.get(field) if snapshot else None
        compatible = False
        reason = "one or both runs did not record this checkpoint"
        metrics = {}
        if ref is not None and cand is not None:
            ra, ca = ref["_arrays"], cand["_arrays"]
            compatible = (
                int(ref["Ns"]) == int(cand["Ns"])
                and ra["x"].shape == ca["x"].shape
                and ra["y"].shape == ca["y"].shape
                and ra["s"].shape == ca["s"].shape
                and ra["c"].shape == ca["c"].shape
            )
            reason = (
                "same topology size; compared corresponding vertex indices without interpolation"
                if compatible
                else "incompatible topology/grid sizes; pointwise geometry metrics omitted"
            )
            if compatible:
                for field in ("x", "y", "c", "s"):
                    metric = _array_metrics(ra[field], ca[field], "curvature" if field == "c" else field)
                    if metric:
                        metrics.update(metric)
                ru, cu = ra.get("U"), ca.get("U")
                umetric = _array_metrics(ru, cu, "U")
                if umetric:
                    metrics.update(umetric)
                else:
                    reason += "; velocity arrays unavailable or differently sized"
        row["grid_compatible"] = compatible
        row["compatibility_reason"] = reason
        row.update(metrics)
        detail.update({"grid_compatible": compatible, "compatibility_reason": reason, **metrics})
        details[str(step)] = detail
        rows.append(row)
    return rows, details


def _scalar_comparison(reference: dict[str, Any], candidate: dict[str, Any]):
    ref_by_step = {int(row["step"]): row for row in reference["_scalar_history"]}
    cand_by_step = {int(row["step"]): row for row in candidate["_scalar_history"]}
    rows = []
    threshold_summary: dict[str, Any] = {}
    shared_steps = sorted(set(ref_by_step) & set(cand_by_step))
    for step in sorted(set(ref_by_step) | set(cand_by_step)):
        ref = ref_by_step.get(step)
        cand = cand_by_step.get(step)
        row = {"step": step, "both_present": ref is not None and cand is not None}
        for field in ("jt", "Ns", "cut_cnt", *SCALAR_FIELDS, "Cf0", "dt"):
            row[f"{field}_reference"] = ref.get(field) if ref else None
            row[f"{field}_candidate"] = cand.get(field) if cand else None
            if ref is not None and cand is not None and field in (*SCALAR_FIELDS, "Cf0", "dt"):
                left, right = float(ref[field]), float(cand[field])
                row[f"abs_{field}_difference"] = abs(left - right)
        rows.append(row)

    for field in SCALAR_FIELDS:
        crossings = {f"{threshold:.0e}": None for threshold in THRESHOLDS}
        first_nonzero = None
        max_difference = 0.0
        for step in shared_steps:
            left, right = float(ref_by_step[step][field]), float(cand_by_step[step][field])
            difference = abs(left - right)
            max_difference = max(max_difference, difference)
            if first_nonzero is None and difference > 0.0:
                first_nonzero = {"step": step, "abs_difference": difference}
            for threshold in THRESHOLDS:
                key = f"{threshold:.0e}"
                if crossings[key] is None and difference > threshold:
                    crossings[key] = {"step": step, "abs_difference": difference}
        threshold_summary[field] = {
            "first_measurable_difference": first_nonzero,
            "first_step_exceeding_threshold": crossings,
            "max_abs_difference_over_shared_history": max_difference,
        }
    first_cut_mismatch = next(
        (
            step for step in shared_steps
            if int(ref_by_step[step]["cut_cnt"]) != int(cand_by_step[step]["cut_cnt"])
        ),
        None,
    )
    first_ns_mismatch = next(
        (
            step for step in shared_steps
            if int(ref_by_step[step]["Ns"]) != int(cand_by_step[step]["Ns"])
        ),
        None,
    )
    threshold_summary["cut_cnt"] = {"first_step_different": first_cut_mismatch}
    threshold_summary["Ns"] = {"first_step_different": first_ns_mismatch}
    return rows, threshold_summary, shared_steps


def _cutoff_comparison(reference: dict[str, Any], candidate: dict[str, Any]):
    ref_events = reference["_cutoff_events"]
    cand_events = candidate["_cutoff_events"]
    rows = []
    first_difference = None
    for index in range(max(len(ref_events), len(cand_events))):
        ref = ref_events[index] if index < len(ref_events) else None
        cand = cand_events[index] if index < len(cand_events) else None
        step_match = ref is not None and cand is not None and ref["step"] == cand["step"]
        cutcnt_match = ref is not None and cand is not None and ref["cut_cnt"] == cand["cut_cnt"]
        if first_difference is None and not (step_match and cutcnt_match):
            first_difference = {
                "event_index": index + 1,
                "reference": ref,
                "candidate": cand,
            }
        row: dict[str, Any] = {
            "event_index": index + 1,
            "both_present": ref is not None and cand is not None,
            "event_step_matches": step_match,
            "cut_cnt_matches": cutcnt_match,
        }
        for side, event in (("reference", ref), ("candidate", cand)):
            for field in (
                "step", "dt_cum", "cut_cnt", "Ns_before", "Ns_after", "Ns_after_geometry",
                "sinuosity_before", "sinuosity_after_cut", "sinuosity_after_geometry", "cut_cnt_after_geometry",
            ):
                row[f"{field}_{side}"] = event.get(field) if event else None
        rows.append(row)
    count_match = len(ref_events) == len(cand_events)
    sequence_match = count_match and [e["cut_cnt"] for e in ref_events] == [e["cut_cnt"] for e in cand_events]
    step_match = count_match and [e["step"] for e in ref_events] == [e["step"] for e in cand_events]
    return rows, {
        "reference_cutoff_events": len(ref_events),
        "candidate_cutoff_events": len(cand_events),
        "cutoff_counts_match": count_match,
        "cutoff_event_sequence_matches": sequence_match,
        "cutoff_steps_match": step_match,
        "first_different_cutoff_event": first_difference,
    }


def _morphology_comparison(reference: dict[str, Any], candidate: dict[str, Any], shared_steps: list[int]):
    if not shared_steps:
        return {"assessment": "not assessable: no shared completed steps"}
    ref_by_step = {int(row["step"]): row for row in reference["_scalar_history"]}
    cand_by_step = {int(row["step"]): row for row in candidate["_scalar_history"]}
    start_step = shared_steps[max(0, int(0.75 * (len(shared_steps) - 1)))]
    suffix = [step for step in shared_steps if step >= start_step]
    ref_values = np.asarray([ref_by_step[step]["sinuosity"] for step in suffix], dtype=np.float64)
    cand_values = np.asarray([cand_by_step[step]["sinuosity"] for step in suffix], dtype=np.float64)
    ref_events = [e for e in reference["_cutoff_events"] if e["step"] >= start_step]
    cand_events = [e for e in candidate["_cutoff_events"] if e["step"] >= start_step]

    def distribution(values: np.ndarray):
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p50": float(np.quantile(values, 0.5)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(np.max(values)),
        }

    ref_stats = distribution(ref_values)
    cand_stats = distribution(cand_values)
    rel_mean = abs(ref_stats["mean"] - cand_stats["mean"]) / max(abs(ref_stats["mean"]), 1.0e-300)
    rel_std = abs(ref_stats["std"] - cand_stats["std"]) / max(abs(ref_stats["std"]), 1.0e-300)
    rel_median = abs(ref_stats["p50"] - cand_stats["p50"]) / max(abs(ref_stats["p50"]), 1.0e-300)
    ref_rate = len(ref_events) / max(1, len(suffix)) * 1000.0
    cand_rate = len(cand_events) / max(1, len(suffix)) * 1000.0
    relative_cutoff_rate_difference = abs(ref_rate - cand_rate) / max(ref_rate, 1.0 / max(1, len(suffix)) * 1000.0)
    # This deliberately transparent descriptive heuristic is not a scientific
    # acceptance criterion. Exact distributions and rates are reported beside it.
    comparable = rel_mean <= 0.02 and rel_std <= 0.10 and rel_median <= 0.02 and relative_cutoff_rate_difference <= 0.10
    return {
        "window": "last 25% of shared scalar history",
        "start_step": start_step,
        "end_step": shared_steps[-1],
        "reference_sinuosity_distribution": ref_stats,
        "candidate_sinuosity_distribution": cand_stats,
        "relative_mean_difference": rel_mean,
        "relative_std_difference": rel_std,
        "relative_median_difference": rel_median,
        "reference_cutoffs_per_1000_steps": ref_rate,
        "candidate_cutoffs_per_1000_steps": cand_rate,
        "relative_cutoff_rate_difference": relative_cutoff_rate_difference,
        "descriptive_heuristic": {
            "criteria": "mean and median sinuosity within 2%, sinuosity standard deviation within 10%, and cutoff rate within 10% over the last quarter",
            "assessment": "comparable under this descriptive heuristic" if comparable else "not comparable under this descriptive heuristic",
            "scientific_acceptance_claim": False,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(value) for key, value in row.items()})


def _serial_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if not key.startswith("_")}


def _summary_report(summary: dict[str, Any]) -> str:
    reference = summary["runs"][0]
    candidate = summary["runs"][1]
    lines = [
        "# Long-term backend parity report",
        "",
        f"Comparison: **{summary['comparison']}**; case {summary['case']}; requested {summary['requested_steps']:,} steps.",
        "",
        "Both runs use the same input files and solver controls. JIT warm-up is untimed and reported separately. The per-run wall time includes the `run_case` call, final flowfield recomputation, and temporary output finalization. No production solver code was changed by this validation runner.",
        "",
        "| Run | Flow backend | Vertical backend | Completed steps | Wall (s) | Loop flowfield (s) | Final flowfield (s) | Total flowfield (s) | Vertical coefficients (s) | k0123 hits | k0123 misses | Cutoffs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in (reference, candidate):
        lines.append(
            "| {label} | {flow_backend} | {vertical_backend} | {completed_steps} | {total_wall_seconds:.6g} | {loop} | {final} | {total} | {vertical:.6g} | {hits} | {misses} | {cutoffs} |".format(
                label=run["label"], flow_backend=run["flow_backend"], vertical_backend=run["vertical_backend"],
                completed_steps=run["completed_steps"], total_wall_seconds=run["total_wall_seconds"],
                loop=f"{run['flowfield_loop_seconds']:.6g}" if run["flowfield_loop_seconds"] is not None else "n/a",
                final=f"{run['flowfield_final_seconds']:.6g}" if run["flowfield_final_seconds"] is not None else "n/a",
                total=f"{run['flowfield_total_seconds']:.6g}" if run["flowfield_total_seconds"] is not None else "n/a",
                vertical=run["vertical_coefficient_seconds_all_flow_calls"], hits=run["k0123_cache_hits"],
                misses=run["k0123_cache_misses"], cutoffs=run["number_of_cutoffs"],
            )
        )
    lines.extend(["", "## Parity findings", ""])
    lines.append(f"Cutoff comparison: `{json.dumps(summary['cutoff_comparison'], sort_keys=True)}`")
    lines.append("")
    lines.append(f"First threshold crossings and topology divergence are in `summary.json`; full checkpoint and scalar data are in the CSV files. The last-quarter sinuosity/cutoff comparison is `{summary['statistical_morphology'].get('descriptive_heuristic', {}).get('assessment', 'not assessable')}` under the explicitly descriptive heuristic. This heuristic is not a scientific acceptance criterion.")
    lines.append("")
    lines.append("All timings are machine-specific and indicative. Checkpoint geometry metrics compare corresponding point indices only when both runs have matching point counts; no interpolation is performed.")
    return "\n".join(lines) + "\n"


def _save_comparison(
    *,
    comparison: str,
    case: int,
    steps: int,
    checkpoints: list[int],
    runs: list[dict[str, Any]],
    results_root: Path,
) -> Path:
    reference, candidate = runs
    checkpoint_rows, checkpoint_details = _checkpoint_comparison(reference, candidate, checkpoints)
    scalar_rows, trajectory, shared_steps = _scalar_comparison(reference, candidate)
    cutoff_rows, cutoff_details = _cutoff_comparison(reference, candidate)
    morphology = _morphology_comparison(reference, candidate, shared_steps)
    checkpoint_incompatibility = [
        int(row["step"]) for row in checkpoint_rows if row["both_present"] and not row["grid_compatible"]
    ]
    summary = {
        "comparison": comparison,
        "case": int(case),
        "requested_steps": int(steps),
        "checkpoint_steps": checkpoints,
        "comparison_definitions": {
            "vertical-only": {
                "reference": {"backend": "numpy", "vertical_backend": "numpy"},
                "candidate": {"backend": "numpy", "vertical_backend": "numba"},
            },
            "full": {
                "reference": {"backend": "numpy", "vertical_backend": "numpy"},
                "candidate": {"backend": "numba", "vertical_backend": "numba"},
            },
        }[comparison],
        "common_input_sha256": reference["input_sha256"],
        "runs": [_serial_run(run) for run in runs],
        "cutoff_comparison": cutoff_details,
        "scalar_trajectory_divergence": trajectory,
        "first_checkpoint_with_incompatible_topology": checkpoint_incompatibility[0] if checkpoint_incompatibility else None,
        "checkpoint_comparisons": checkpoint_details,
        "statistical_morphology": morphology,
        "run_completed_requested_steps": [run["completed_steps"] == steps and run["run_error"] is None for run in runs],
        "runner_instrumentation_errors": [run["capture_errors"] for run in runs],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "controls": {
            "dsliminicial": 1.0,
            "ER": 1.0e-8,
            "cstab": 0.01,
            "geometry_smoothing_enabled": True,
            "geometry_smoothing_factor": 8.0,
            "neck_cutoff_interval": 3,
            "numba_parallel": False,
            "numba_fastmath": False,
            "flow_paral": 0,
            "periodic_boundary": False,
            "do_plots": False,
            "collect_timing": True,
            "periodic_output_snapshots_suppressed_by_Nprint": True,
        },
    }
    for optional, name in (("numba", "numba"),):
        try:
            module = __import__(optional)
            summary["environment"][name] = module.__version__
        except Exception:
            summary["environment"][name] = None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{time.time_ns() % 1_000_000:06d}"
    output_dir = results_root / run_id / comparison
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(_json_value(summary), indent=2), encoding="utf-8")
    _write_csv(output_dir / "checkpoint_comparison.csv", checkpoint_rows)
    _write_csv(output_dir / "scalar_history_comparison.csv", scalar_rows)
    _write_csv(output_dir / "cutoff_comparison.csv", cutoff_rows)
    (output_dir / "report.md").write_text(_summary_report(summary), encoding="utf-8")
    return output_dir


def _parse_checkpoints(value: str | None, steps: int) -> list[int]:
    if value is None:
        return sorted({step for step in DEFAULT_CHECKPOINTS if step <= steps})
    try:
        requested = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint steps must be comma-separated integers") from exc
    if any(step < 0 for step in requested):
        raise argparse.ArgumentTypeError("checkpoint steps must be non-negative")
    requested.add(0)
    return sorted(step for step in requested if step <= steps)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000, help="maximum migration steps per run (default: 10000)")
    parser.add_argument("--case", type=int, default=1, help="case ID from Input/Parameter.csv (default: 1)")
    parser.add_argument(
        "--comparison", choices=("vertical-only", "full", "both"), default="vertical-only",
        help="compare only the vertical optimization, the complete Numba backend, or run both sequentially",
    )
    parser.add_argument(
        "--checkpoint-steps", default=None,
        help="comma-separated checkpoint steps; defaults to 0,1,10,100,1000,5000,10000,25000,50000,75000,100000 filtered by --steps",
    )
    parser.add_argument("--progress-every", type=int, default=1_000, help="print progress every N steps; 0 disables progress")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="generated results directory")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.case <= 0:
        parser.error("--case must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    try:
        checkpoints = _parse_checkpoints(args.checkpoint_steps, args.steps)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    try:
        import numba  # noqa: F401
    except Exception as exc:
        if args.comparison in ("vertical-only", "full", "both"):
            parser.error(f"Numba is required for the requested comparison but could not be imported: {exc}")

    comparisons = ("vertical-only", "full") if args.comparison == "both" else (args.comparison,)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for comparison in comparisons:
        if comparison == "vertical-only":
            settings = (
                ("reference_numpy_numpy", "numpy", "numpy"),
                ("vertical_numba_only", "numpy", "numba"),
            )
        else:
            settings = (
                ("reference_numpy_numpy", "numpy", "numpy"),
                ("full_numba", "numba", "numba"),
            )
        runs = []
        for label, flow_backend, vertical_backend in settings:
            print(
                f"Starting {comparison} run {label}: flow backend={flow_backend}, "
                f"vertical backend={vertical_backend}, max steps={args.steps:,}.",
                flush=True,
            )
            run = _run_one(
                label=label,
                case=args.case,
                steps=args.steps,
                checkpoints=set(checkpoints),
                flow_backend=flow_backend,
                vertical_backend=vertical_backend,
                progress_every=args.progress_every,
            )
            runs.append(run)
            if run["run_error"]:
                print(f"[{label}] stopped with {run['run_error']['type']}: {run['run_error']['message']}", flush=True)
            else:
                print(
                    f"[{label}] finished {run['completed_steps']:,} steps in {run['total_wall_seconds']:.3f}s.",
                    flush=True,
                )
        output_dir = _save_comparison(
            comparison=comparison,
            case=args.case,
            steps=args.steps,
            checkpoints=checkpoints,
            runs=runs,
            results_root=args.results_dir,
        )
        generated.append(output_dir)
        print(f"Wrote results: {output_dir}", flush=True)
        if not all(run["completed_steps"] == args.steps and run["run_error"] is None for run in runs):
            print(f"{comparison} did not complete the requested step count for both runs; stopping subsequent comparisons.", flush=True)
            break
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
