from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from .mathutils import (
    find_neck_cutoff_kdtree_with_refine,
    matlab_gradient,
    matlab_spline,
    smooth_xy_via_theta,
    unwrap_angles_like_matlab,
)
from .outputs import plot_cut, save_xy_cut


def _maybe_smooth_xy(
    xa: np.ndarray,
    ya: np.ndarray,
    *,
    smoothing_enabled: bool,
    smoothing_wavelength_factor: float,
    timing: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if (not smoothing_enabled) or xa.size < 5:
        return xa, ya
    ds_est = float(np.mean(np.sqrt(np.diff(xa) ** 2 + np.diff(ya) ** 2)))
    if not np.isfinite(ds_est) or ds_est <= 0.0:
        return xa, ya
    fc = 1.0 / (float(smoothing_wavelength_factor) * ds_est)
    t0 = perf_counter() if timing is not None else None
    xa, ya = smooth_xy_via_theta(
        xa,
        ya,
        method="sos",
        cutoff=fc,
        sos_order=3,
        periodic=False,
    )
    if timing is not None and t0 is not None:
        timing["smoothing"] = float(timing.get("smoothing", 0.0) + (perf_counter() - t0))
    return xa, ya


def geometry4(
    xa: np.ndarray,
    ya: np.ndarray,
    jt: int,
    dsliminicial: float,
    id_files: str,
    Ntstep: int,
    cut_cnt: int,
    beta: float,
    base_out: Path,
    *,
    dslim_upper_factor: float = 1.03,
    dslim_lower_factor: float = 0.97,
    neck_cutoff_interval: int = 3,
    smoothing_enabled: bool = True,
    smoothing_wavelength_factor: float = 8.0,
    timing: dict | None = None,
    output_units: str = "dimensionless",
    output_length_scale: float = 1.0,
    do_plots: bool = True,
):
    """Geometry processing, resampling, smoothing, and cutoff detection."""
    dslim = dsliminicial * float(dslim_upper_factor)
    dslim2 = dsliminicial * float(dslim_lower_factor)
    dslim3 = beta
    ss = int(dslim3 * 2)

    xa = np.asarray(xa, dtype=np.float64)
    ya = np.asarray(ya, dtype=np.float64)

    def _recompute_sa(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
        delta_x = np.diff(xv)
        delta_y = np.diff(yv)
        return np.concatenate(([0.0], np.cumsum(np.sqrt(delta_x ** 2 + delta_y ** 2))))

    Na = xa.size
    Naold = xa.size

    sa = _recompute_sa(xa, ya)
    Nnew = int(1 + round(sa[-1] / dsliminicial))
    if Nnew > 4 * Na:
        raise RuntimeError("Error por alargamento excesivo")

    saux1 = np.linspace(sa[0], sa[-1], Na)
    xa = matlab_spline(sa, xa, saux1)
    ya = matlab_spline(sa, ya, saux1)
    Na = xa.size
    Naold = xa.size
    sa = _recompute_sa(xa, ya)

    if neck_cutoff_interval > 0 and (jt % int(neck_cutoff_interval)) == 0 and (Na - ss) > 1 and ss > 0:
        t_neck0 = perf_counter() if timing is not None else None
        while True:
            hit = find_neck_cutoff_kdtree_with_refine(xa, ya, ss, dslim3)
            if hit is None:
                break
            i0, j0 = hit
            start = i0 + ss
            i = i0 + 1
            AA = (j0 - start) + 1
            cut_cnt = int(cut_cnt + 1)
            if do_plots:
                plot_cut(base_out, xa, ya, i, AA, id_files, jt, Ntstep, cut_cnt, ss, output_units=output_units, length_scale=output_length_scale)
            save_xy_cut(base_out, xa, ya, i, AA, id_files, jt, Ntstep, cut_cnt, ss, output_units=output_units, length_scale=output_length_scale)
            end_mat = ss + AA + i + 1
            start_py = i - 1
            end_inc_py = min(end_mat - 1, Na - 1)
            mask = np.ones(Na, dtype=bool)
            mask[start_py:end_inc_py + 1] = False
            xa = xa[mask]
            ya = ya[mask]
            Na = xa.size
            if (Na - ss) <= 1:
                break
        if timing is not None and t_neck0 is not None:
            timing["neck"] = float(timing.get("neck", 0.0) + (perf_counter() - t_neck0))

    if xa.size != Naold:
        sa = _recompute_sa(xa, ya)
        Nnew = int(1 + round(sa[-1] / dsliminicial))
        _ = Nnew
        xa, ya = _maybe_smooth_xy(
            xa,
            ya,
            smoothing_enabled=smoothing_enabled,
            smoothing_wavelength_factor=smoothing_wavelength_factor,
            timing=timing,
        )
        sa = _recompute_sa(xa, ya)

    delt_sa = np.diff(sa)
    if delt_sa.size:
        mean_delt = float(np.mean(delt_sa))
        if mean_delt > dslim:
            xa, ya = _maybe_smooth_xy(
                xa,
                ya,
                smoothing_enabled=smoothing_enabled,
                smoothing_wavelength_factor=smoothing_wavelength_factor,
                timing=timing,
            )
            sa = _recompute_sa(xa, ya)
            Na = xa.size
            Nnew = int(1 + round(sa[-1] / dsliminicial))
            saux1 = np.linspace(sa[0], sa[-1], Nnew)
            xa = matlab_spline(sa, xa, saux1)
            ya = matlab_spline(sa, ya, saux1)
            sa = _recompute_sa(xa, ya)
            delt_sa = np.diff(sa)
            mean_delt = float(np.mean(delt_sa)) if delt_sa.size else mean_delt

        if delt_sa.size and mean_delt < dslim2:
            xa, ya = _maybe_smooth_xy(
                xa,
                ya,
                smoothing_enabled=smoothing_enabled,
                smoothing_wavelength_factor=smoothing_wavelength_factor,
                timing=timing,
            )
            sa = _recompute_sa(xa, ya)
            Na = xa.size
            Nnew = int(1 + round(sa[-1] / dsliminicial))
            saux1 = np.linspace(sa[0], sa[-1], Nnew)
            xa = matlab_spline(sa, xa, saux1)
            ya = matlab_spline(sa, ya, saux1)
            sa = _recompute_sa(xa, ya)

    Ns = int(sa.size)
    deltas = float(sa[-1] / (Ns - 1))

    dxg = matlab_gradient(xa)
    dyg = matlab_gradient(ya)
    theta_raw = np.arctan2(dyg, dxg)
    theta = unwrap_angles_like_matlab(theta_raw)
    theta = -1.0 * theta

    c = matlab_gradient(theta)

    wave_l = float(sa[-1])
    valle_l = float(np.sqrt((xa[0] - xa[-1]) ** 2 + (ya[0] - ya[-1]) ** 2))
    sinuo = float(wave_l / valle_l) if valle_l != 0 else np.inf

    return c, sa, xa, ya, theta, Ns, deltas, wave_l, valle_l, sinuo, int(cut_cnt)
