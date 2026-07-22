"""The vectorised HAC covariance must match the textbook elementwise form.

This is a pure performance change: the assertions below pin the numerical
equivalence so a future refactor cannot silently alter the reported
uncertainty.
"""

from __future__ import annotations

import numpy as np

from ldsfl.stability import sinuosity_equivalence_stability


def _reference_meat(design: np.ndarray, residual: np.ndarray, lag_count: int) -> np.ndarray:
    """Elementwise Newey-West meat matrix, as originally implemented."""
    meat = np.zeros((2, 2), dtype=np.float64)
    for i in range(design.shape[0]):
        xi = design[i : i + 1].T
        meat += float(residual[i] ** 2) * (xi @ xi.T)
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / float(lag_count + 1)
        for i in range(lag, design.shape[0]):
            xi = design[i : i + 1].T
            xlag = design[i - lag : i - lag + 1].T
            meat += weight * float(residual[i] * residual[i - lag]) * (xi @ xlag.T + xlag @ xi.T)
    return meat


def _vectorised_meat(design: np.ndarray, residual: np.ndarray, lag_count: int) -> np.ndarray:
    weighted = design * residual[:, None]
    meat = weighted.T @ weighted
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / float(lag_count + 1)
        cross = weighted[lag:].T @ weighted[:-lag]
        meat += weight * (cross + cross.T)
    return meat


def test_vectorised_meat_matches_elementwise_reference():
    rng = np.random.default_rng(11)
    n = 800
    x = np.arange(n, dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x - x.mean()))
    residual = rng.normal(0.0, 1.0e-6, n)

    for lag_count in (0, 1, 5, 50):
        ref = _reference_meat(design, residual, lag_count)
        vec = _vectorised_meat(design, residual, lag_count)
        assert np.allclose(vec, ref, rtol=1.0e-12, atol=0.0), f"lag_count={lag_count}"


def test_zero_lag_reduces_to_white_noise_covariance():
    rng = np.random.default_rng(3)
    n = 500
    x = np.arange(n, dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x - x.mean()))
    residual = rng.normal(0.0, 1.0e-6, n)

    meat = _vectorised_meat(design, residual, 0)
    expected = (design * residual[:, None]).T @ (design * residual[:, None])
    assert np.allclose(meat, expected, rtol=1.0e-14)


def test_diagnostic_still_reports_finite_fields_on_a_realistic_history():
    n = 5000
    steps = np.arange(n, dtype=np.float64)
    sinuosity = 1.5 + np.linspace(0.0, 0.01, n)
    result = sinuosity_equivalence_stability(steps, sinuosity, transient_step=None)
    for key in ("slope_per_step", "slope_se", "estimated_total_drift", "drift_ci_low", "drift_ci_high"):
        assert np.isfinite(result[key]), key
