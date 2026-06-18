from __future__ import annotations

import numpy as np

from ldsfl.stability import sinuosity_equivalence_stability


def test_equivalence_stability_accepts_small_post_transient_drift():
    steps = np.arange(0, 100_001, 1_000, dtype=float)
    sinuo = np.where(
        steps < 40_000,
        1.0 + 0.4 * (steps / 40_000),
        1.45 + 1.0e-8 * (steps - 40_000),
    )

    result = sinuosity_equivalence_stability(
        steps,
        sinuo,
        transient_step=40_000,
        drift_tolerance=0.02,
        confidence=0.90,
        hac_lags=0,
    )

    assert result["stable"] is True
    assert result["state"] == "stable within tolerance"
    assert result["drift_ci_low"] > -0.02
    assert result["drift_ci_high"] < 0.02


def test_equivalence_stability_rejects_clear_post_transient_drift():
    steps = np.arange(0, 100_001, 1_000, dtype=float)
    sinuo = np.where(
        steps < 40_000,
        1.0 + 0.4 * (steps / 40_000),
        1.45 + 1.0e-6 * (steps - 40_000),
    )

    result = sinuosity_equivalence_stability(
        steps,
        sinuo,
        transient_step=40_000,
        drift_tolerance=0.02,
        confidence=0.90,
        hac_lags=0,
    )

    assert result["stable"] is False
    assert result["state"] == "stability not demonstrated"
    assert result["estimated_total_drift"] > 0.02


def test_equivalence_stability_reports_insufficient_data():
    result = sinuosity_equivalence_stability(
        [0, 1, 2],
        [1.0, 1.0, 1.0],
        transient_step=40_000,
    )

    assert result["stable"] is False
    assert result["state"] == "insufficient data"
    assert result["n_points"] == 0


def test_equivalence_stability_handles_nonfinite_values():
    steps = np.array([40_000, 41_000, 42_000, 43_000, 44_000, 45_000, 46_000, 47_000, 48_000, 49_000], dtype=float)
    sinuo = np.array([1.5, 1.5, np.nan, 1.5, 1.5, 1.5, 1.5, np.inf, 1.5, 1.5], dtype=float)

    result = sinuosity_equivalence_stability(
        steps,
        sinuo,
        transient_step=40_000,
        min_points=5,
        drift_tolerance=0.02,
    )

    assert result["n_points"] == 8
    assert result["stable"] is True
