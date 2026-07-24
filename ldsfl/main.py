from __future__ import annotations

import os
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

try:  # optional
    import numba as _numba
except Exception:  # pragma: no cover
    _numba = None

from .evolution import dxdy2, update_parameters
from .flowfield import parall_u_free
from .flowfield_periodic import parall_u_periodic
from .geometry import geometry4
from .inputs import dimensionless_input_table, read_parameter_table, read_xy
from .outputs import ensure_dirs, plot_it, save_sinuosity_history, save_variables, save_xystcu
from .profile import preprof_3
from .resistance import resistance_function_flagbed
from .resonance import resonance_report
from .stability import sinuosity_equivalence_stability


def initial_curvature(th: np.ndarray, deltas: float = 1.0) -> np.ndarray:
    """Curvature for the first flow-field call, matching geometry4's convention.

    ``preprof_3`` already returns the negated tangent angle
    (``theta = -unwrap(arctan2(dy, dx))``), and ``geometry4`` computes
    curvature as ``gradient(theta_negated) / deltas`` on every subsequent step.
    Negating again here flips the first-step curvature sign.

    ``deltas`` is the centreline arclength spacing. ``np.gradient`` uses unit
    spacing, so it returns dtheta/d(index); dividing by the spacing gives the
    physical curvature dtheta/ds. Passing it keeps the first step consistent
    with every later step and makes the result independent of the sampling
    resolution of ``Input/xy.csv``. The default of 1.0 preserves the previous
    behaviour for external callers that do not supply a spacing.
    """
    return np.gradient(np.asarray(th, dtype=np.float64), edge_order=1) / float(deltas)


def make_id_files(case_i: int, beta: float, ds: float, theta0: float, flagbed: int, rpic_0: float) -> str:
    parts = [
        str(int(case_i)),
        format(beta, ".15g"),
        format(ds, ".15g"),
        format(theta0, ".15g"),
        str(int(flagbed)),
        format(rpic_0, ".15g"),
    ]
    return ("_".join(parts)).replace(".", "")


def _criterion_names(
    *,
    steps: int,
    dt_cum: float,
    cut_cnt: int,
    max_steps: int | None,
    max_sim_time: float | None,
    max_cutoffs: int | None,
    stop_on_steps: bool,
    stop_on_time: bool,
    stop_on_cutoffs: bool,
    stop_on_sinuosity_stability: bool = False,
    sinuosity_stability_reached: bool = False,
) -> list[str]:
    reached: list[str] = []
    if stop_on_steps and max_steps is not None and max_steps > 0 and steps >= max_steps:
        reached.append("max_steps")
    if stop_on_time and max_sim_time is not None and max_sim_time > 0.0 and dt_cum >= max_sim_time:
        reached.append("max_sim_time")
    if stop_on_cutoffs and max_cutoffs is not None and max_cutoffs > 0 and cut_cnt >= max_cutoffs:
        reached.append("max_cutoffs")
    if stop_on_sinuosity_stability and bool(sinuosity_stability_reached):
        reached.append("sinuosity_stability")
    return reached


def _should_stop(
    reached: list[str],
    *,
    stop_mode: str,
    stop_on_steps: bool,
    stop_on_time: bool,
    stop_on_cutoffs: bool,
    stop_on_sinuosity_stability: bool = False,
) -> bool:
    enabled_count = (
        int(stop_on_steps)
        + int(stop_on_time)
        + int(stop_on_cutoffs)
        + int(stop_on_sinuosity_stability)
    )
    if enabled_count <= 0:
        return False
    if enabled_count == 1:
        return len(reached) == 1
    if str(stop_mode).lower() == "all":
        return len(reached) == enabled_count
    return len(reached) > 0


def _final_snapshot_velocity(
    U: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    th: np.ndarray,
    c: np.ndarray,
    Cf0: float,
    CT: float,
    CD: float,
    phiT: float,
    phiD: float,
    beta: float,
    rpic: float,
    theta0: float,
    F0: float,
    Mdat: int,
    n1: np.ndarray,
    deltas: float,
    flow_bc: str,
    flow_paral: int,
    n_workers: int,
    flow_backend: str,
    numba_parallel: bool,
    numba_fastmath: bool,
) -> np.ndarray:
    """Recompute a final-snapshot velocity field for the final geometry.

    The solver computes ``U`` before moving and resampling the planform. Regular
    in-loop snapshots are saved before that move, so their geometry and velocity
    arrays are naturally aligned. The final snapshot is saved after the loop,
    using the latest geometry produced by the most recent move/resampling step.

    That final geometry can be newer than the last velocity field even when the
    number of centreline points happens to be unchanged. Recompute the final flow
    field unconditionally so the saved final ``x/y/s/th/c`` and ``U`` all refer
    to the same centreline state.
    """
    geometry_lengths = {
        "x": int(np.asarray(x).shape[0]),
        "y": int(np.asarray(y).shape[0]),
        "s": int(np.asarray(s).shape[0]),
        "th": int(np.asarray(th).shape[0]),
        "c": int(np.asarray(c).shape[0]),
    }
    expected = geometry_lengths["x"]
    inconsistent = {name: length for name, length in geometry_lengths.items() if length != expected}
    if inconsistent:
        raise ValueError(f"Final geometry arrays have inconsistent lengths: {geometry_lengths}")

    Ns_final = int(expected)
    if str(flow_bc).lower().startswith("per"):
        final_U, _ = parall_u_periodic(
            c,
            s,
            Cf0,
            CT,
            CD,
            phiT,
            phiD,
            beta,
            rpic,
            theta0,
            F0,
            Mdat,
            1,
            Ns_final,
            n1,
            deltas,
            paral=int(flow_paral),
            n_workers=n_workers,
        )
    else:
        final_U, _ = parall_u_free(
            c,
            s,
            Cf0,
            CT,
            CD,
            phiT,
            phiD,
            beta,
            rpic,
            theta0,
            F0,
            Mdat,
            1,
            Ns_final,
            n1,
            deltas,
            SL=0,
            paral=int(flow_paral),
            n_workers=n_workers,
            backend=str(flow_backend),
            numba_parallel=bool(numba_parallel),
            numba_fastmath=bool(numba_fastmath),
        )

    final_U = np.asarray(final_U, dtype=np.float64)
    if final_U.shape[0] != expected:
        raise ValueError(
            f"Recomputed final velocity length {final_U.shape[0]} does not match final geometry length {expected}."
        )
    return final_U


def _sinuosity_stability_metrics(
    step_hist,
    sinuo_hist,
    *,
    window: int = 100,
    rel_tol: float = 5.0e-3,
) -> dict:
    """Assess whether sinuosity is stable or quasi-stable.

    The test uses the last ``window`` stored values. Two metrics are reported:

    * relative span: (max - min) / final sinuosity over the window;
    * relative trend per step: absolute least-squares slope divided by final sinuosity.

    Suggested interpretation:

    * stable: relative span < rel_tol and relative trend per step < rel_tol / window;
    * quasi-stable: relative span < 2 rel_tol and relative trend per step < 2 rel_tol / window.
    """
    steps = np.asarray(step_hist, dtype=np.float64)
    vals = np.asarray(sinuo_hist, dtype=np.float64)
    if vals.size == 0:
        return {
            "state": "not available",
            "stable": False,
            "quasi_stable": False,
            "window_requested": int(window),
            "window_used": 0,
            "rel_span": np.nan,
            "rel_last_change": np.nan,
            "rel_trend_per_step": np.nan,
            "sinuo_final": np.nan,
        }
    if vals.size == 1:
        return {
            "state": "not enough history",
            "stable": False,
            "quasi_stable": False,
            "window_requested": int(window),
            "window_used": 1,
            "rel_span": 0.0,
            "rel_last_change": 0.0,
            "rel_trend_per_step": 0.0,
            "sinuo_final": float(vals[-1]),
        }

    w = max(2, min(int(window), vals.size))
    tail = vals[-w:]
    tail_steps = steps[-w:] if steps.size >= vals.size else np.arange(vals.size - w, vals.size, dtype=np.float64)
    ref = max(abs(float(tail[-1])), 1.0e-12)

    rel_span = float((tail.max() - tail.min()) / ref)
    rel_last_change = float(abs(tail[-1] - tail[-2]) / ref)

    x = tail_steps - tail_steps[0]
    if np.allclose(x, x[0]):
        slope = 0.0
    else:
        slope = float(np.polyfit(x, tail, 1)[0])
    rel_trend_per_step = float(abs(slope) / ref)

    trend_tol = float(rel_tol) / max(float(w), 1.0)
    stable = (rel_span < rel_tol) and (rel_trend_per_step < trend_tol)
    quasi_stable = stable or ((rel_span < 2.0 * rel_tol) and (rel_trend_per_step < 2.0 * trend_tol))
    if stable:
        state = "stable"
    elif quasi_stable:
        state = "quasi-stable"
    else:
        state = "not stable"

    return {
        "state": state,
        "stable": bool(stable),
        "quasi_stable": bool(quasi_stable),
        "window_requested": int(window),
        "window_used": int(w),
        "rel_span": rel_span,
        "rel_last_change": rel_last_change,
        "rel_trend_per_step": rel_trend_per_step,
        "sinuo_final": float(vals[-1]),
    }

def _combined_sinuosity_stability_metrics(
    step_hist,
    sinuo_hist,
    *,
    window: int = 100,
    rel_tol: float = 5.0e-3,
    equivalence_transient_step: float | None = 40_000.0,
    equivalence_drift_tolerance: float = 0.02,
    equivalence_confidence: float = 0.90,
    equivalence_min_points: int = 10,
    equivalence_hac_lags: int = 50,
    equivalence_method: str = "increment",
) -> dict:
    """Return moving-window and equivalence-style sinuosity diagnostics."""
    stability = _sinuosity_stability_metrics(step_hist, sinuo_hist, window=window, rel_tol=rel_tol)
    stability["equivalence"] = sinuosity_equivalence_stability(
        step_hist,
        sinuo_hist,
        transient_step=equivalence_transient_step,
        drift_tolerance=equivalence_drift_tolerance,
        confidence=equivalence_confidence,
        min_points=equivalence_min_points,
        hac_lags=equivalence_hac_lags,
        method=equivalence_method,
    )
    return stability

def run_case(
    base_dir: Path,
    case_i: int,
    Nprint: int = 10000,
    Ntstep: int = 1_000_000,
    Max_Cut: int = 100_000,
    dsliminicial: float = 1.0,
    ER: float = 1.0e-8,
    flow_bc="free",
    flow_paral=0,
    flow_workers=0,
    flow_backend: str = "numpy",
    numba_parallel: bool = False,
    numba_fastmath: bool = False,
    max_steps: int | None = None,
    max_sim_time: float | None = None,
    stop_on_steps: bool = True,
    stop_on_time: bool = False,
    stop_on_cutoffs: bool = True,
    stop_on_sinuosity_stability: bool = False,
    stop_mode: str = "first",
    cstab: float = 0.01,
    geometry_smoothing_enabled: bool = True,
    geometry_smoothing_factor: float = 8.0,
    neck_cutoff_interval: int = 3,
    resample_upper_factor: float = 1.03,
    resample_lower_factor: float = 0.97,
    do_plots: bool = True,
    collect_timing: bool = False,
    output_units: str = "dimensionless",
    output_length_scale: float = 1.0,
    output_velocity_scale: float = 1.0,
    sinuo_window: int = 100,
    sinuo_rel_tol: float = 5.0e-3,
    sinuo_equiv_transient_step: float | None = 40_000.0,
    sinuo_equiv_drift_tol: float = 0.02,
    sinuo_equiv_confidence: float = 0.90,
    sinuo_equiv_min_points: int = 10,
    sinuo_equiv_hac_lags: int = 50,
    sinuo_equiv_method: str = "increment",
    sinuo_stability_interval: int = 100,
    return_equivalence_stability: bool = False,
    stop_requested_callback=None,
):
    """Run one LDSFL-Meander case using the prepared Input/ files."""
    base_dir = Path(base_dir)
    in_dir = base_dir / "Input"
    out_dir = base_dir / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)

    ER = float(ER)
    if not np.isfinite(ER) or ER <= 0.0:
        raise ValueError("Erosion rate must be finite and > 0")

    if not (
        bool(stop_on_steps)
        or bool(stop_on_time)
        or bool(stop_on_cutoffs)
        or bool(stop_on_sinuosity_stability)
    ):
        raise ValueError("At least one stop criterion must be enabled.")

    df = read_parameter_table(in_dir / "Parameter.csv")
    beta, ds, theta0, flagbed, rpic_0, Mdat = dimensionless_input_table(df, case_i)
    id_files = make_id_files(case_i, beta, ds, theta0, flagbed, rpic_0)
    ensure_dirs(out_dir, id_files)

    xap, yap = read_xy(in_dir / "xy.csv")
    s, x, y, th, Ns, deltas, wave_l, valle_l, sinuo = preprof_3(xap, yap, dsliminicial)
    c = initial_curvature(th, deltas)
    U = np.zeros_like(x, dtype=np.float64)

    rpic, Cf0, CT, CD, phiT, phiD, F0 = resistance_function_flagbed(flagbed, theta0, ds, rpic_0)

    var_name = [
        "jt",
        "dt",
        "dt_cum",
        "rpic",
        "Cf0",
        "CT",
        "CD",
        "phiT",
        "phiD",
        "F0",
        "deltas",
        "wave_l",
        "valle_l",
        "sinuo",
        "beta",
        "theta0",
        "ds",
        "cut_cnt",
    ]
    save_var_n = 18
    # One row per solver step within a save block; flushed every Nprint steps
    # and once more at the end of the run for the final partial block.
    T_var = np.full((max(Nprint, 1), save_var_n), np.nan, dtype=np.float64)

    x_origin = x.copy()
    y_origin = y.copy()
    dxdtl_old = 0.0
    dydtl_old = 0.0
    jt = 1
    dt_cum = 0.0
    dt = 0.0
    cut_cnt = 0
    cnt_f = 1
    Nsold = 0

    step_hist: list[int] = [0]
    sinuo_hist: list[float] = [float(sinuo)]
    stability_interval = max(1, int(sinuo_stability_interval))
    use_equivalence_stability = bool(stop_on_sinuosity_stability) or bool(return_equivalence_stability)
    if bool(stop_on_sinuosity_stability):
        stability_info = _combined_sinuosity_stability_metrics(
            step_hist,
            sinuo_hist,
            window=sinuo_window,
            rel_tol=sinuo_rel_tol,
            equivalence_transient_step=sinuo_equiv_transient_step,
            equivalence_drift_tolerance=sinuo_equiv_drift_tol,
            equivalence_confidence=sinuo_equiv_confidence,
            equivalence_min_points=sinuo_equiv_min_points,
            equivalence_hac_lags=sinuo_equiv_hac_lags,
                equivalence_method=sinuo_equiv_method,
        )
    else:
        stability_info = _sinuosity_stability_metrics(
            step_hist,
            sinuo_hist,
            window=sinuo_window,
            rel_tol=sinuo_rel_tol,
        )

    n1 = np.array([1.0], dtype=np.float64)
    flow_workers = int(flow_workers)
    n_workers = None if flow_workers <= 0 else flow_workers

    if _numba is not None:
        try:
            if str(flow_backend).lower() == "numba" and bool(numba_parallel) and int(flow_paral) == 0:
                _numba.set_num_threads(os.cpu_count() or 1)
            else:
                _numba.set_num_threads(1)
        except Exception:
            pass

    if int(flow_paral) == 1 and bool(numba_parallel):
        warnings.warn(
            "flow_paral=1 with numba_parallel=True may oversubscribe CPU cores. "
            "Consider flow_paral=0 when using numba_parallel, or set numba_parallel=False.",
            RuntimeWarning,
            stacklevel=2,
        )

    steps = 0
    stop_reason = "unknown"
    stop_criteria_reached: list[str] = []

    tim = None
    if bool(collect_timing):
        tim = {
            "flowfield": 0.0,
            "flowfield_first": 0.0,
            "move": 0.0,
            "geometry": 0.0,
            "smoothing": 0.0,
            "neck": 0.0,
            "update": 0.0,
            "saving": 0.0,
        }

    with threadpool_limits(limits=1):
        while True:
            if stop_requested_callback is not None and bool(stop_requested_callback()):
                stop_reason = "stopped by user"
                stop_criteria_reached = ["user_stop"]
                break

            reached = _criterion_names(
                steps=steps,
                dt_cum=dt_cum,
                cut_cnt=cut_cnt,
                max_steps=max_steps,
                max_sim_time=max_sim_time,
                max_cutoffs=Max_Cut,
                stop_on_steps=bool(stop_on_steps),
                stop_on_time=bool(stop_on_time),
                stop_on_cutoffs=bool(stop_on_cutoffs),
                stop_on_sinuosity_stability=bool(stop_on_sinuosity_stability),
                sinuosity_stability_reached=bool(
                    (stability_info.get("equivalence") or {}).get("stable", False)
                ),
            )
            if _should_stop(
                reached,
                stop_mode=stop_mode,
                stop_on_steps=bool(stop_on_steps),
                stop_on_time=bool(stop_on_time),
                stop_on_cutoffs=bool(stop_on_cutoffs),
                stop_on_sinuosity_stability=bool(stop_on_sinuosity_stability),
            ):
                stop_criteria_reached = reached
                stop_reason = f"stop criteria reached: {', '.join(reached)}"
                break

            if 1 <= cnt_f <= Nprint:
                row = T_var[cnt_f - 1]
                row[0] = jt
                row[1] = dt
                row[2] = dt_cum
                row[3] = rpic
                row[4] = Cf0
                row[5] = CT
                row[6] = CD
                row[7] = phiT
                row[8] = phiD
                row[9] = F0
                row[10] = deltas
                row[11] = wave_l
                row[12] = valle_l
                row[13] = sinuo
                row[14] = beta
                row[15] = theta0
                row[16] = ds
                row[17] = cut_cnt

            if str(flow_bc).lower().startswith("per"):
                t0 = perf_counter() if tim is not None else None
                U, flag = parall_u_periodic(
                    c,
                    s,
                    Cf0,
                    CT,
                    CD,
                    phiT,
                    phiD,
                    beta,
                    rpic,
                    theta0,
                    F0,
                    Mdat,
                    1,
                    Ns,
                    n1,
                    deltas,
                    paral=int(flow_paral),
                    n_workers=n_workers,
                )
                if tim is not None and t0 is not None:
                    dt_flow = perf_counter() - t0
                    tim["flowfield"] += float(dt_flow)
                    if steps == 0:
                        tim["flowfield_first"] = float(dt_flow)
            else:
                t0 = perf_counter() if tim is not None else None
                U, flag = parall_u_free(
                    c,
                    s,
                    Cf0,
                    CT,
                    CD,
                    phiT,
                    phiD,
                    beta,
                    rpic,
                    theta0,
                    F0,
                    Mdat,
                    1,
                    Ns,
                    n1,
                    deltas,
                    SL=0,
                    paral=int(flow_paral),
                    n_workers=n_workers,
                    backend=str(flow_backend),
                    numba_parallel=bool(numba_parallel),
                    numba_fastmath=bool(numba_fastmath),
                )
                if tim is not None and t0 is not None:
                    dt_flow = perf_counter() - t0
                    tim["flowfield"] += float(dt_flow)
                    if steps == 0:
                        tim["flowfield_first"] = float(dt_flow)

            if (jt % Nprint) == 0:
                t0s = perf_counter() if tim is not None else None
                save_xystcu(
                    out_dir,
                    x,
                    y,
                    s,
                    th,
                    c,
                    U,
                    Ntstep,
                    jt,
                    id_files,
                    cut_cnt,
                    output_units=output_units,
                    length_scale=output_length_scale,
                    velocity_scale=output_velocity_scale,
                )
                if do_plots:
                    plot_it(
                        out_dir,
                        x_origin,
                        y_origin,
                        x,
                        y,
                        id_files,
                        jt,
                        Ntstep,
                        cut_cnt,
                        output_units=output_units,
                        length_scale=output_length_scale,
                    )
                save_variables(
                    out_dir,
                    T_var[:cnt_f],
                    Ntstep,
                    jt,
                    var_name,
                    id_files,
                    cut_cnt,
                    output_units=output_units,
                    length_scale=output_length_scale,
                )
                save_sinuosity_history(out_dir, id_files, step_hist, sinuo_hist)
                T_var[:] = np.nan
                # cnt_f is incremented at the end of every iteration, so 0 here
                # makes the next iteration write into row 0 of the fresh block.
                cnt_f = 0
                if tim is not None and t0s is not None:
                    tim["saving"] += float(perf_counter() - t0s)

            t0 = perf_counter() if tim is not None else None
            dxdtl, dydtl, dt = dxdy2(ER, U, x, y, th, deltas, Nsold, Ns, jt, cstab=float(cstab))
            if Nsold == Ns:
                x += (0.5 * (dxdtl + dxdtl_old)) * dt
                y += (0.5 * (dydtl + dydtl_old)) * dt
            else:
                x += dxdtl * dt
                y += dydtl * dt
            if tim is not None and t0 is not None:
                tim["move"] += float(perf_counter() - t0)

            dxdtl_old = dxdtl
            dydtl_old = dydtl
            dt_cum = dt + dt_cum

            Nsold = Ns
            wave_l_old = wave_l
            valle_l_old = valle_l
            Cf0_old = Cf0

            geom_sub = {} if tim is not None else None
            t0g = perf_counter() if tim is not None else None
            c, s, x, y, th, Ns, deltas, wave_l, valle_l, sinuo, cut_cnt = geometry4(
                x,
                y,
                jt,
                dsliminicial,
                id_files,
                Ntstep,
                cut_cnt,
                beta,
                out_dir,
                dslim_upper_factor=resample_upper_factor,
                dslim_lower_factor=resample_lower_factor,
                neck_cutoff_interval=neck_cutoff_interval,
                smoothing_enabled=geometry_smoothing_enabled,
                smoothing_wavelength_factor=geometry_smoothing_factor,
                timing=geom_sub,
                output_units=output_units,
                output_length_scale=output_length_scale,
                do_plots=do_plots,
            )
            if tim is not None and t0g is not None:
                tim["geometry"] += float(perf_counter() - t0g)
                tim["smoothing"] += float((geom_sub or {}).get("smoothing", 0.0))
                tim["neck"] += float((geom_sub or {}).get("neck", 0.0))

            t0u = perf_counter() if tim is not None else None
            rpic, Cf0, CT, CD, phiT, phiD, F0 = resistance_function_flagbed(flagbed, theta0, ds, rpic_0)
            beta, theta0, ds = update_parameters(
                Cf0_old,
                Cf0,
                wave_l_old,
                wave_l,
                valle_l_old,
                valle_l,
                beta,
                ds,
                theta0,
            )
            if tim is not None and t0u is not None:
                tim["update"] += float(perf_counter() - t0u)

            steps += 1
            step_hist.append(int(steps))
            sinuo_hist.append(float(sinuo))
            if bool(stop_on_sinuosity_stability) and (steps % stability_interval) == 0:
                stability_info = _combined_sinuosity_stability_metrics(
                    step_hist,
                    sinuo_hist,
                    window=sinuo_window,
                    rel_tol=sinuo_rel_tol,
                    equivalence_transient_step=sinuo_equiv_transient_step,
                    equivalence_drift_tolerance=sinuo_equiv_drift_tol,
                    equivalence_confidence=sinuo_equiv_confidence,
                    equivalence_min_points=sinuo_equiv_min_points,
                    equivalence_hac_lags=sinuo_equiv_hac_lags,
                equivalence_method=sinuo_equiv_method,
                )
            else:
                previous_equivalence = (stability_info or {}).get("equivalence")
                stability_info = _sinuosity_stability_metrics(
                    step_hist,
                    sinuo_hist,
                    window=sinuo_window,
                    rel_tol=sinuo_rel_tol,
                )
                if bool(stop_on_sinuosity_stability) and previous_equivalence is not None:
                    stability_info["equivalence"] = previous_equivalence

            jt += 1
            cnt_f += 1

    # Always write a final geometry snapshot so the GUI can refresh and
    # optionally continue from the last available centerline.
    try:
        final_U = _final_snapshot_velocity(
            U,
            x=x,
            y=y,
            s=s,
            th=th,
            c=c,
            Cf0=Cf0,
            CT=CT,
            CD=CD,
            phiT=phiT,
            phiD=phiD,
            beta=beta,
            rpic=rpic,
            theta0=theta0,
            F0=F0,
            Mdat=Mdat,
            n1=n1,
            deltas=deltas,
            flow_bc=flow_bc,
            flow_paral=flow_paral,
            n_workers=n_workers,
            flow_backend=flow_backend,
            numba_parallel=numba_parallel,
            numba_fastmath=numba_fastmath,
        )
        save_xystcu(
            out_dir,
            x,
            y,
            s,
            th,
            c,
            final_U,
            Ntstep,
            jt,
            id_files,
            cut_cnt,
            output_units=output_units,
            length_scale=output_length_scale,
            velocity_scale=output_velocity_scale,
        )
        if do_plots:
            plot_it(
                out_dir,
                x_origin,
                y_origin,
                x,
                y,
                id_files,
                jt,
                Ntstep,
                cut_cnt,
                output_units=output_units,
                length_scale=output_length_scale,
            )
    except Exception as exc:
        warnings.warn(
            f"Final snapshot could not be saved: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Flush the final partial variable-history block. cnt_f is incremented at
    # the end of each completed iteration, so cnt_f - 1 rows have been recorded
    # since the last Nprint flush.
    n_pending = max(0, int(cnt_f) - 1)
    if n_pending > 0:
        try:
            save_variables(
                out_dir,
                T_var[:n_pending],
                Ntstep,
                jt,
                var_name,
                id_files,
                cut_cnt,
                output_units=output_units,
                length_scale=output_length_scale,
            )
        except Exception as exc:
            warnings.warn(
                f"Final variable history block could not be saved: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


    save_sinuosity_history(out_dir, id_files, step_hist, sinuo_hist)
    if use_equivalence_stability:
        stability_info = _combined_sinuosity_stability_metrics(
            step_hist,
            sinuo_hist,
            window=sinuo_window,
            rel_tol=sinuo_rel_tol,
            equivalence_transient_step=sinuo_equiv_transient_step,
            equivalence_drift_tolerance=sinuo_equiv_drift_tol,
            equivalence_confidence=sinuo_equiv_confidence,
            equivalence_min_points=sinuo_equiv_min_points,
            equivalence_hac_lags=sinuo_equiv_hac_lags,
                equivalence_method=sinuo_equiv_method,
        )
    else:
        stability_info = _sinuosity_stability_metrics(
            step_hist,
            sinuo_hist,
            window=sinuo_window,
            rel_tol=sinuo_rel_tol,
        )

    return {
        "id_files": id_files,
        "beta": beta,
        "ds": ds,
        "theta0": theta0,
        "flagbed": flagbed,
        "rpic_0": rpic_0,
        "Mdat": Mdat,
        "erosion_rate": float(ER),
        "steps": steps,
        "cut_cnt": cut_cnt,
        "jt": jt,
        "dt_cum": dt_cum,
        "stop_reason": stop_reason,
        "stop_criteria_reached": stop_criteria_reached,
        "timings": tim,
        "output_units": output_units,
        "output_length_scale": output_length_scale,
        "output_velocity_scale": output_velocity_scale,
        "sinuo_final": float(sinuo_hist[-1]),
        "resonance": resonance_report(beta, theta0, ds, rpic_0, flagbed, Mdat),
        "sinuosity_stability": stability_info,
    }


def run_project(
    base_dir: Path,
    cases: list[int] | None = None,
    *,
    flow_bc: str = "free",
    flow_paral: int = 0,
    flow_workers: int = 0,
    flow_backend: str = "numpy",
    numba_parallel: bool = False,
    numba_fastmath: bool = False,
    output_units: str = "dimensionless",
    output_length_scale: float = 1.0,
    output_velocity_scale: float = 1.0,
    **kwargs,
):
    base_dir = Path(base_dir)

    df = read_parameter_table(base_dir / "Input" / "Parameter.csv")
    ncases = len(df)

    if cases is None:
        cases = list(range(1, ncases + 1))

    flow_bc = str(flow_bc).lower()
    flow_paral = int(flow_paral)
    flow_workers = int(flow_workers)
    flow_backend = str(flow_backend).lower()
    numba_parallel = bool(numba_parallel)
    numba_fastmath = bool(numba_fastmath)

    results = []
    for case_i in cases:
        results.append(
            run_case(
                base_dir,
                case_i,
                flow_bc=flow_bc,
                flow_paral=flow_paral,
                flow_workers=flow_workers,
                flow_backend=flow_backend,
                numba_parallel=numba_parallel,
                numba_fastmath=numba_fastmath,
                output_units=output_units,
                output_length_scale=output_length_scale,
                output_velocity_scale=output_velocity_scale,
                **kwargs,
            )
        )
    return results
