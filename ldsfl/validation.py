"""Shared validation for scientific inputs and core solver controls."""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd

REQUIRED_PARAMETER_COLUMNS = ("Id", "Beta", "ds", "Thetha", "flagbed", "r", "Mdat")

_DISPLAY_NAMES = {
    "ER": "Erosion rate (ER)",
    "sinuo_equiv_drift_tol": "Equivalence drift tolerance (sinuo_equiv_drift_tol)",
    "sinuo_equiv_confidence": "Equivalence confidence (sinuo_equiv_confidence)",
    "sinuo_equiv_min_points": "Equivalence minimum points (sinuo_equiv_min_points)",
    "sinuo_equiv_hac_lags": "Equivalence HAC lags (sinuo_equiv_hac_lags)",
    "sinuo_equiv_method": "Equivalence method (sinuo_equiv_method)",
    "sinuo_stability_interval": "Sinuosity stability check interval (sinuo_stability_interval)",
}


def _display_name(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


def _finite_float(name: str, value) -> float:
    label = _display_name(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(name: str, value, *, minimum: int | None = None) -> int:
    label = _display_name(name)
    number = _finite_float(name, value)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    result = int(number)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _positive(name: str, value) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{_display_name(name)} must be > 0")
    return result


def validate_scientific_parameters(
    *, case_id, beta, ds, theta0, flagbed, rpic_0, mdat
) -> dict[str, int | float]:
    """Validate and normalize one scientific parameter set."""
    normalized = {
        "Id": _integer("Id", case_id, minimum=1),
        "Beta": _positive("Beta", beta),
        "ds": _positive("ds", ds),
        "Thetha": _positive("Thetha", theta0),
        "flagbed": _integer("flagbed", flagbed),
        "r": _positive("r", rpic_0),
        "Mdat": _integer("Mdat", mdat, minimum=1),
    }
    if normalized["flagbed"] not in (1, 2):
        raise ValueError("flagbed must be exactly 1 or 2")
    return normalized


def validate_parameter_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, numerically normalized Parameter.csv table."""
    missing = [name for name in REQUIRED_PARAMETER_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Parameter table is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Parameter table must contain at least one row")

    validated = df.copy()
    rows: list[dict[str, int | float]] = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            rows.append(
                validate_scientific_parameters(
                    case_id=row["Id"],
                    beta=row["Beta"],
                    ds=row["ds"],
                    theta0=row["Thetha"],
                    flagbed=row["flagbed"],
                    rpic_0=row["r"],
                    mdat=row["Mdat"],
                )
            )
        except ValueError as exc:
            raise ValueError(f"Invalid Parameter.csv row {row_number}: {exc}") from exc

    ids = [int(row["Id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Parameter.csv Id values must be unique")
    for name in REQUIRED_PARAMETER_COLUMNS:
        validated[name] = [row[name] for row in rows]
    return validated


def validate_run_controls(**values) -> dict:
    """Validate controls accepted by the core ``run_case`` API."""
    result = dict(values)
    for name, minimum in (
        ("Nprint", 1),
        ("Ntstep", 1),
        ("Max_Cut", 0),
        ("neck_cutoff_interval", 0),
        ("sinuo_window", 2),
        ("sinuo_equiv_min_points", 3),
        ("sinuo_equiv_hac_lags", 0),
        ("sinuo_stability_interval", 1),
        ("flow_workers", 0),
    ):
        result[name] = _integer(name, values[name], minimum=minimum)

    if values["max_steps"] is not None:
        result["max_steps"] = _integer("max_steps", values["max_steps"], minimum=0)
    if values["max_sim_time"] is not None:
        result["max_sim_time"] = _finite_float("max_sim_time", values["max_sim_time"])
        if result["max_sim_time"] < 0.0:
            raise ValueError("max_sim_time must be >= 0")

    for name in (
        "dsliminicial",
        "ER",
        "cstab",
        "geometry_smoothing_factor",
        "sinuo_rel_tol",
        "sinuo_equiv_drift_tol",
    ):
        result[name] = _positive(name, values[name])

    transient = values["sinuo_equiv_transient_step"]
    if transient is not None:
        result["sinuo_equiv_transient_step"] = _finite_float("sinuo_equiv_transient_step", transient)
        if result["sinuo_equiv_transient_step"] < 0.0:
            raise ValueError("sinuo_equiv_transient_step must be >= 0 or None")

    result["sinuo_equiv_confidence"] = _finite_float(
        "sinuo_equiv_confidence", values["sinuo_equiv_confidence"]
    )
    if not 0.0 < result["sinuo_equiv_confidence"] < 1.0:
        raise ValueError("Equivalence confidence (sinuo_equiv_confidence) must be between 0 and 1")
    result["resample_upper_factor"] = _finite_float(
        "resample_upper_factor", values["resample_upper_factor"]
    )
    if result["resample_upper_factor"] <= 1.0:
        raise ValueError("resample_upper_factor must be > 1")
    result["resample_lower_factor"] = _finite_float(
        "resample_lower_factor", values["resample_lower_factor"]
    )
    if not 0.0 < result["resample_lower_factor"] < 1.0:
        raise ValueError("resample_lower_factor must be between 0 and 1")

    for name, choices in (
        ("flow_bc", {"free", "periodic"}),
        ("flow_backend", {"numpy", "numba"}),
        ("output_units", {"dimensionless", "dimensional"}),
        ("sinuo_equiv_method", {"increment", "hac"}),
        ("stop_mode", {"first", "all"}),
    ):
        result[name] = str(values[name]).lower()
        if result[name] not in choices:
            raise ValueError(f"{_display_name(name)} must be one of: {', '.join(sorted(choices))}")

    result["flow_paral"] = _integer("flow_paral", values["flow_paral"])
    if result["flow_paral"] not in (0, 1):
        raise ValueError("flow_paral must be 0 or 1")

    units = result["output_units"]
    for name in ("output_length_scale", "output_velocity_scale"):
        value = values[name]
        if units == "dimensional":
            if value is None:
                raise ValueError(f"Dimensional output requires an explicit finite positive {name}")
            result[name] = _positive(name, value)
        elif value is None:
            result[name] = 1.0
        else:
            result[name] = _positive(name, value)

    return result
