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
    method: str = "increment",
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
        Newey-West/HAC lag count. Only used when ``method="hac"``. Set to 0 for
        ordinary least-squares uncertainty.
    method:
        ``"increment"`` (default) estimates drift from the mean of the per-step
        increments; ``"hac"`` reproduces the previous least-squares plus
        Newey-West behaviour.

        A sinuosity history behaves like a smooth random walk plus a slow
        trend, i.e. it is integrated rather than short-memory stationary. HAC
        estimators are inconsistent under a unit root, so no bandwidth restores
        the nominal coverage. Measured coverage of the nominal 90% interval
        over 150 synthetic trials with random-walk residuals:

            method / bandwidth                     coverage
            hac, 50 lags (previous default)            17%
            hac, 4*(n/100)^(2/9)                        4%
            hac, n^(1/3)                                4%
            hac, n^(1/2)                                9%
            hac, 0.25*n                                27%
            hac, 50 lags on a 25x thinned series        25%
            increment (new default)                    91%

        Differencing removes the unit root: if ``y_t = a + d*t + w_t`` with
        ``w`` a random walk, the increments are ``d`` plus iid noise, so their
        mean and standard error are consistent. On short-memory residuals the
        increment method over-differences and is conservative (measured 100%
        coverage), which errs toward *not* declaring stability - the safe
        direction for a stopping criterion.

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
        "method": str(method),
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

    method_key = str(method).strip().lower()
    if method_key not in ("increment", "hac"):
        raise ValueError(f"Unknown method: {method!r}. Use 'increment' or 'hac'.")

    if method_key == "increment":
        # Differencing removes the unit root that makes HAC inconsistent here.
        dx = np.diff(x)
        dy = np.diff(y)
        usable = dx > 0.0
        dx = dx[usable]
        dy = dy[usable]
        if dx.size < 2:
            return base_result
        per_step = dy / dx
        slope = float(per_step.mean())
        dof = max(int(per_step.size) - 1, 1)
        slope_se = float(per_step.std(ddof=1) / math.sqrt(per_step.size))
        intercept = float(np.mean(y) - slope * np.mean(x))
        lag_count = 0
        if not np.isfinite(slope_se):
            return base_result
    else:
        xtx_inv = np.linalg.pinv(design.T @ design)

        # Newey-West/HAC covariance for autocorrelated time series residuals.
        #
        # Vectorised form of the textbook double loop. With Xe[i] =
        # residual[i] * design[i], the zero-lag term is Xe.T @ Xe and each lag
        # term is A + A.T with A = Xe[lag:].T @ Xe[:-lag]. Algebraically
        # identical to the elementwise accumulation but replaces O(n * lags)
        # Python-level iterations with small matrix products.
        lag_count = max(0, min(int(hac_lags), int(x.size) - 1))
        weighted = design * residual[:, None]
        meat = weighted.T @ weighted

        for lag in range(1, lag_count + 1):
            weight = 1.0 - lag / float(lag_count + 1)
            cross = weighted[lag:].T @ weighted[:-lag]
            meat += weight * (cross + cross.T)

        cov = xtx_inv @ meat @ xtx_inv
        slope_var = float(cov[1, 1])
        if not np.isfinite(slope_var) or slope_var < 0.0:
            # Fallback to ordinary least-squares covariance if the small-sample
            # HAC matrix is numerically ill-conditioned.
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
            "method": method_key,
        }
    )
    return base_result
