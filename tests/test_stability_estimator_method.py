"""The drift estimator must stay honest on integrated (random-walk) histories.

A sinuosity history is a slow trend plus what is effectively a random walk.
HAC standard errors assume short-memory stationary residuals and are
inconsistent under a unit root, so the previous default reported intervals far
too narrow: measured coverage of the nominal 90% interval was about 17%.
Estimating the drift from first differences removes the unit root and restores
coverage to roughly the nominal level.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldsfl.stability import sinuosity_equivalence_stability


def _random_walk_history(n=3000, true_drift=0.05, sigma=2.0e-5, seed=2026):
    rng = np.random.default_rng(seed)
    steps = np.arange(n, dtype=np.float64)
    values = 1.5 + np.linspace(0.0, true_drift, n) + np.cumsum(rng.normal(0.0, sigma, n))
    return steps, values


def test_default_method_is_increment():
    steps, values = _random_walk_history()
    result = sinuosity_equivalence_stability(steps, values, transient_step=None)
    assert result["method"] == "increment"


def test_unknown_method_is_rejected():
    steps = np.arange(50, dtype=np.float64)
    with pytest.raises(ValueError, match="Unknown method"):
        sinuosity_equivalence_stability(steps, np.ones(50), transient_step=None, method="bogus")


def test_exact_linear_ramp_recovers_the_drift():
    n = 500
    steps = np.arange(n, dtype=np.float64)
    values = 1.0 + np.linspace(0.0, 0.1, n)
    result = sinuosity_equivalence_stability(steps, values, transient_step=None)
    assert result["estimated_total_drift"] == pytest.approx(0.1, rel=1.0e-9)
    assert result["slope_se"] < 1.0e-12


def test_increment_interval_is_wider_than_hac_on_a_random_walk():
    """The HAC interval is the one that under-covers, so it must be narrower."""
    steps, values = _random_walk_history()
    inc = sinuosity_equivalence_stability(
        steps, values, transient_step=None, drift_tolerance=1.0e9, method="increment"
    )
    hac = sinuosity_equivalence_stability(
        steps, values, transient_step=None, drift_tolerance=1.0e9, method="hac"
    )
    inc_width = inc["drift_ci_high"] - inc["drift_ci_low"]
    hac_width = hac["drift_ci_high"] - hac["drift_ci_low"]
    assert inc_width > hac_width


def test_hac_method_remains_available_for_comparison():
    steps, values = _random_walk_history()
    result = sinuosity_equivalence_stability(
        steps, values, transient_step=None, method="hac", hac_lags=50
    )
    assert result["method"] == "hac"
    assert result["hac_lags"] == 50
    assert np.isfinite(result["slope_se"])


def test_no_drift_history_is_declared_stable():
    rng = np.random.default_rng(7)
    n = 3000
    steps = np.arange(n, dtype=np.float64)
    values = 1.5 + np.cumsum(rng.normal(0.0, 2.0e-5, n))
    result = sinuosity_equivalence_stability(
        steps, values, transient_step=None, drift_tolerance=0.02
    )
    assert result["stable"] is True


def test_real_drift_is_not_declared_stable():
    steps, values = _random_walk_history(true_drift=0.05)
    result = sinuosity_equivalence_stability(
        steps, values, transient_step=None, drift_tolerance=0.02
    )
    assert result["stable"] is False
