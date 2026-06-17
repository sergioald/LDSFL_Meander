from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .mathutils import jt_p_string


def ensure_dirs(base_out: Path, id_files: str):
    root = base_out / id_files
    (root / "plot").mkdir(parents=True, exist_ok=True)
    (root / "files").mkdir(parents=True, exist_ok=True)
    (root / "xyu").mkdir(parents=True, exist_ok=True)
    (root / "xy_cut").mkdir(parents=True, exist_ok=True)


def _convert_length(arr, output_units: str, length_scale: float):
    arr = np.asarray(arr, dtype=np.float64)
    if str(output_units).lower() == "dimensional":
        return arr * float(length_scale)
    return arr


def _convert_curvature(arr, output_units: str, length_scale: float):
    arr = np.asarray(arr, dtype=np.float64)
    if str(output_units).lower() == "dimensional":
        return arr / float(length_scale)
    return arr


def _convert_velocity(arr, output_units: str, velocity_scale: float):
    arr = np.asarray(arr, dtype=np.float64)
    if str(output_units).lower() == "dimensional":
        return arr * float(velocity_scale)
    return arr


def save_xystcu(
    base_out: Path,
    x,
    y,
    s,
    th,
    c,
    U,
    Ntstep: int,
    jt: int,
    id_files: str,
    cut_cnt: int,
    *,
    output_units: str = "dimensionless",
    length_scale: float = 1.0,
    velocity_scale: float = 1.0,
):
    jt_p = jt_p_string(Ntstep, cut_cnt)
    arr = np.column_stack(
        [
            _convert_length(x, output_units, length_scale),
            _convert_length(y, output_units, length_scale),
            _convert_length(s, output_units, length_scale),
            np.asarray(th, dtype=np.float64),
            _convert_curvature(c, output_units, length_scale),
            _convert_velocity(U, output_units, velocity_scale),
        ]
    ).astype(np.float64)

    df = pd.DataFrame(arr, columns=["x", "y", "s", "th", "c", "U"])
    csv_path = base_out / id_files / "xyu" / f"xyu_{id_files}_{jt_p}_{jt:d}.csv"
    df.to_csv(csv_path, index=False)


def save_variables(
    base_out: Path,
    T_var,
    Ntstep: int,
    jt: int,
    var_name,
    id_files: str,
    cut_cnt: int,
    *,
    output_units: str = "dimensionless",
    length_scale: float = 1.0,
):
    jt_p = jt_p_string(Ntstep, cut_cnt)
    df = pd.DataFrame(np.asarray(T_var), columns=list(var_name))

    if str(output_units).lower() == "dimensional":
        for col in ("deltas", "wave_l", "valle_l"):
            if col in df.columns:
                df[col] = df[col].astype(float) * float(length_scale)

    csv_path = base_out / id_files / "files" / f"var_{id_files}_{jt_p}_{jt:d}.csv"
    df.to_csv(csv_path, index=False)


def plot_it(
    base_out: Path,
    x_origin,
    y_origin,
    x,
    y,
    id_files: str,
    jt: int,
    Ntstep: int,
    cut_cnt: int,
    *,
    output_units: str = "dimensionless",
    length_scale: float = 1.0,
):
    jt_p = jt_p_string(Ntstep, cut_cnt)

    x_origin = _convert_length(x_origin, output_units, length_scale)
    y_origin = _convert_length(y_origin, output_units, length_scale)
    x = _convert_length(x, output_units, length_scale)
    y = _convert_length(y, output_units, length_scale)

    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(x_origin, y_origin, label="old")
    ax.plot(x, y, label="new")
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    ax.set_xlabel("x [m]" if str(output_units).lower() == "dimensional" else "x [-]")
    ax.set_ylabel("y [m]" if str(output_units).lower() == "dimensional" else "y [-]")
    ax.set_title(f"{id_files}-{jt}")

    out_path = base_out / id_files / "plot" / f"{id_files}_{jt_p}_{jt:d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def plot_cut(
    base_out: Path,
    xa,
    ya,
    i1: int,
    AA: int,
    id_files: str,
    jt: int,
    Ntstep: int,
    cut_cnt: int,
    ss: int,
    *,
    output_units: str = "dimensionless",
    length_scale: float = 1.0,
):
    jt_p = jt_p_string(Ntstep, cut_cnt)

    xa = _convert_length(xa, output_units, length_scale)
    ya = _convert_length(ya, output_units, length_scale)

    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(xa, ya)

    start = i1 - 1
    end_inclusive = (ss + AA + i1 + 1) - 1
    segx = xa[start : end_inclusive + 1]
    segy = ya[start : end_inclusive + 1]

    ax.plot(segx, segy)
    ax.plot(segx, segy, ".", markersize=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{id_files}-{jt}")

    out_path = base_out / id_files / "plot" / f"{id_files}_{jt_p}_{jt:d}_cut.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def save_xy_cut(
    base_out: Path,
    xa,
    ya,
    i1: int,
    AA: int,
    id_files: str,
    jt: int,
    Ntstep: int,
    cut_cnt: int,
    ss: int,
    *,
    output_units: str = "dimensionless",
    length_scale: float = 1.0,
):
    jt_p = jt_p_string(Ntstep, cut_cnt)
    start = i1 - 1
    end_inclusive = (ss + AA + i1 + 1) - 1

    seg = np.column_stack(
        [
            _convert_length(xa[start : end_inclusive + 1], output_units, length_scale),
            _convert_length(ya[start : end_inclusive + 1], output_units, length_scale),
        ]
    ).astype(np.float64)

    csv_path = base_out / id_files / "xy_cut" / f"xy_cut_{id_files}_{jt_p}_{jt:d}.csv"
    pd.DataFrame(seg, columns=["x", "y"]).to_csv(csv_path, index=False)


def save_sinuosity_history(
    base_out: Path,
    id_files: str,
    time_hist,
    sinuo_hist,
    *,
    x_label: str = "Morphodynamic time [-]",
):
    """
    Save sinuosity history both as CSV and PNG.

    Notes:
    - sinuosity is dimensionless, so it is never rescaled.
    - time_hist is assumed to be the solver cumulative time dt_cum, which is
      dimensionless in the current solver.
    """
    time_hist = np.asarray(time_hist, dtype=np.float64)
    sinuo_hist = np.asarray(sinuo_hist, dtype=np.float64)

    if time_hist.size == 0 or sinuo_hist.size == 0:
        return

    df = pd.DataFrame(
        {
            "time": time_hist,
            "sinuo": sinuo_hist,
        }
    )
    csv_path = base_out / id_files / "files" / f"sinuosity_history_{id_files}.csv"
    df.to_csv(csv_path, index=False)

    fig = Figure(figsize=(7.0, 4.5), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(time_hist, sinuo_hist, linewidth=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sinuosity [-]")
    ax.set_title("Sinuosity evolution")
    ax.grid(True, alpha=0.3)

    png_path = base_out / id_files / "plot" / f"sinuosity_history_{id_files}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
