from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mathutils import jt_p_string


def ensure_dirs(base_out: Path, id_files: str):
    (base_out / id_files).mkdir(parents=True, exist_ok=True)
    (base_out / id_files / "plot").mkdir(parents=True, exist_ok=True)
    (base_out / id_files / "files").mkdir(parents=True, exist_ok=True)
    (base_out / id_files / "xyu").mkdir(parents=True, exist_ok=True)
    (base_out / id_files / "xy_cut").mkdir(parents=True, exist_ok=True)


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


def save_xystcu(base_out: Path, x, y, s, th, c, U, Ntstep: int, jt: int, id_files: str, cut_cnt: int, *, output_units: str = "dimensionless", length_scale: float = 1.0, velocity_scale: float = 1.0):
    jt_p = jt_p_string(Ntstep, cut_cnt)
    arr = np.column_stack([
        _convert_length(x, output_units, length_scale),
        _convert_length(y, output_units, length_scale),
        _convert_length(s, output_units, length_scale),
        np.asarray(th, dtype=np.float64),
        _convert_curvature(c, output_units, length_scale),
        _convert_velocity(U, output_units, velocity_scale),
    ]).astype(np.float64)
    df = pd.DataFrame(arr, columns=["x", "y", "s", "th", "c", "U"])
    csv_path = base_out / id_files / "xyu" / f"xyu_{id_files}_{jt_p}_{jt:d}.csv"
    df.to_csv(csv_path, index=False)


def save_variables(base_out: Path, T_var, Ntstep: int, jt: int, var_name, id_files: str, cut_cnt: int, *, output_units: str = "dimensionless", length_scale: float = 1.0):
    jt_p = jt_p_string(Ntstep, cut_cnt)
    df = pd.DataFrame(np.asarray(T_var), columns=list(var_name))
    if str(output_units).lower() == "dimensional":
        for col in ("deltas", "wave_l", "valle_l"):
            if col in df.columns:
                df[col] = df[col].astype(float) * float(length_scale)
    csv_path = base_out / id_files / "files" / f"var_{id_files}_{jt_p}_{jt:d}.csv"
    df.to_csv(csv_path, index=False)


def plot_it(base_out: Path, x_origin, y_origin, x, y, id_files: str, jt: int, Ntstep: int, cut_cnt: int, *, output_units: str = "dimensionless", length_scale: float = 1.0):
    jt_p = jt_p_string(Ntstep, cut_cnt)
    fig, ax = plt.subplots()
    x_origin = _convert_length(x_origin, output_units, length_scale)
    y_origin = _convert_length(y_origin, output_units, length_scale)
    x = _convert_length(x, output_units, length_scale)
    y = _convert_length(y, output_units, length_scale)
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
    plt.close(fig)


def plot_cut(base_out: Path, xa, ya, i1: int, AA: int, id_files: str, jt: int, Ntstep: int, cut_cnt: int, ss: int, *, output_units: str = "dimensionless", length_scale: float = 1.0):
    jt_p = jt_p_string(Ntstep, cut_cnt)
    xa = _convert_length(xa, output_units, length_scale)
    ya = _convert_length(ya, output_units, length_scale)
    fig = plt.figure()
    plt.plot(xa, ya)
    start = i1 - 1
    end_inclusive = (ss + AA + i1 + 1) - 1
    segx = xa[start:end_inclusive + 1]
    segy = ya[start:end_inclusive + 1]
    plt.plot(segx, segy)
    plt.plot(segx, segy, ".", markersize=4)
    plt.axis("equal")
    plt.title(f"{id_files}-{jt}")
    out_path = base_out / id_files / "plot" / f"{id_files}_{jt_p}_{jt:d}_cut.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_xy_cut(base_out: Path, xa, ya, i1: int, AA: int, id_files: str, jt: int, Ntstep: int, cut_cnt: int, ss: int, *, output_units: str = "dimensionless", length_scale: float = 1.0):
    jt_p = jt_p_string(Ntstep, cut_cnt)
    start = i1 - 1
    end_inclusive = (ss + AA + i1 + 1) - 1
    seg = np.column_stack([
        _convert_length(xa[start:end_inclusive + 1], output_units, length_scale),
        _convert_length(ya[start:end_inclusive + 1], output_units, length_scale),
    ]).astype(np.float64)
    csv_path = base_out / id_files / "xy_cut" / f"xy_cut_{id_files}_{jt_p}_{jt:d}.csv"
    pd.DataFrame(seg, columns=["x", "y"]).to_csv(csv_path, index=False)

