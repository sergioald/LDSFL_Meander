"""Statistical stability diagnostics for LDSFL-Meander time series."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def sinuosity_equivalence_stability(
    steps,
    sinuosity,
    *,
    transient_step: float | None = 40_000.0,
    drift_tolerance: float = 0.02,
    confidence: float = 0.90,
    min_points: int = 10,
    hac_lags: int = 50,
) -> dict[str, Any]:
    """Test whether post-transient sinuosity drift is practically negligible.

    This is an equivalence-style diagnostic. It does not test whether the fitted
    slope is exactly zero. Instead, it estimates the total fitted drift over the
    analysis window and accepts stability only when the confidence interval for
    that total drift lies fully inside ``[-drift_tolerance, drift_tolerance]``.

    Parameters
    ----------
    steps, sinuosity:
        Step and sinuosity histories.
    transient_step:
        Discard values with ``step < transient_step``. Pass ``None`` to use all
        finite values.
    drift_tolerance:
        Maximum practically negligible total drift in sinuosity units over the
        analysis window.
    confidence:
        Confidence level used for the drift interval. A 90% interval is common
        for two one-sided equivalence-style checks.
    min_points:
        Minimum number of post-transient points required.
    hac_lags:
        Newey-West/HAC lag count used to make the slope uncertainty more robust
        to autocorrelation. Set to 0 for ordinary least-squares uncertainty.

    Returns
    -------
    dict
        A JSON-serializable diagnostic dictionary.
    """

    x_raw = np.asarray(steps, dtype=np.float64)
    y_raw = np.asarray(sinuosity, dtype=np.float64)

    n_pair = min(x_raw.size, y_raw.size)
    x_raw = x_raw[:n_pair]
    y_raw = y_raw[:n_pair]

    finite = np.isfinite(x_raw) & np.isfinite(y_raw)
    if transient_step is not None:
        finite &= x_raw >= float(transient_step)

    x = x_raw[finite]
    y = y_raw[finite]

    base_result: dict[str, Any] = {
        "state": "insufficient data",
        "stable": False,
        "analysis_start_step": None if transient_step is None else float(transient_step),
        "drift_tolerance": float(drift_tolerance),
        "confidence": float(confidence),
        "n_points": int(x.size),
        "window_steps": 0.0,
        "estimated_total_drift": math.nan,
        "drift_ci_low": math.nan,
        "drift_ci_high": math.nan,
        "slope_per_step": math.nan,
        "slope_se": math.nan,
        "hac_lags": int(hac_lags),
    }

    if x.size < int(min_points):
        return base_result

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    window_steps = float(x[-1] - x[0])
    base_result["window_steps"] = window_steps
    if not np.isfinite(window_steps) or window_steps <= 0.0:
        return base_result

    # Center the step coordinate for numerical conditioning.
    x_centered = x - float(np.mean(x))
    design = np.column_stack((np.ones_like(x_centered), x_centered))

    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return base_result

    intercept = float(beta[0])
    slope = float(beta[1])
    residual = y - design @ beta
    dof = max(int(x.size) - 2, 1)

    xtx_inv = np.linalg.pinv(design.T @ design)

    # Newey-West/HAC covariance for autocorrelated time series residuals.
    lag_count = max(0, min(int(hac_lags), int(x.size) - 1))
    meat = np.zeros((2, 2), dtype=np.float64)
    for i in range(x.size):
        xi = design[i : i + 1].T
        meat += float(residual[i] ** 2) * (xi @ xi.T)

    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / float(lag_count + 1)
        for i in range(lag, x.size):
            xi = design[i : i + 1].T
            xlag = design[i - lag : i - lag + 1].T
            term = float(residual[i] * residual[i - lag])
            meat += weight * term * (xi @ xlag.T + xlag @ xi.T)

    cov = xtx_inv @ meat @ xtx_inv
    slope_var = float(cov[1, 1])
    if not np.isfinite(slope_var) or slope_var < 0.0:
        # Fallback to ordinary least-squares covariance if the small-sample HAC
        # matrix is numerically ill-conditioned.
        sigma2 = float(np.dot(residual, residual) / dof)
        cov = sigma2 * xtx_inv
        slope_var = float(cov[1, 1])
        lag_count = 0

    slope_se = float(math.sqrt(max(slope_var, 0.0)))
    alpha = 1.0 - float(confidence)
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, dof))

    drift = float(slope * window_steps)
    drift_half_width = float(tcrit * slope_se * window_steps)
    drift_ci_low = drift - drift_half_width
    drift_ci_high = drift + drift_half_width

    stable = bool(drift_ci_low > -float(drift_tolerance) and drift_ci_high < float(drift_tolerance))

    base_result.update(
        {
            "state": "stable within tolerance" if stable else "stability not demonstrated",
            "stable": stable,
            "analysis_start_step": float(x[0]),
            "n_points": int(x.size),
            "window_steps": window_steps,
            "estimated_total_drift": drift,
            "drift_ci_low": float(drift_ci_low),
            "drift_ci_high": float(drift_ci_high),
            "slope_per_step": slope,
            "slope_se": slope_se,
            "intercept": intercept,
            "hac_lags": int(lag_count),
        }
    )
    return base_result
