from __future__ import annotations

import numpy as np
import pytest

from ldsfl.main import _sinuosity_stability_metrics


def test_sinuosity_metrics_empty_history():
    metrics = _sinuosity_stability_metrics([], [])

    assert metrics["state"] == "not available"
    assert metrics["stable"] is False
    assert metrics["quasi_stable"] is False
    assert metrics["window_used"] == 0
    assert np.isnan(metrics["sinuo_final"])


def test_sinuosity_metrics_single_value():
    metrics = _sinuosity_stability_metrics([0], [1.25])

    assert metrics["state"] == "not enough history"
    assert metrics["stable"] is False
    assert metrics["quasi_stable"] is False
    assert metrics["window_used"] == 1
    assert metrics["sinuo_final"] == pytest.approx(1.25)


def test_sinuosity_metrics_flat_history_is_stable():
    metrics = _sinuosity_stability_metrics(
        list(range(10)),
        [2.5] * 10,
        window=5,
        rel_tol=5.0e-3,
    )

    assert metrics["state"] == "stable"
    assert metrics["stable"] is True
    assert metrics["quasi_stable"] is True
    assert metrics["window_used"] == 5
    assert metrics["rel_span"] == pytest.approx(0.0)
    assert metrics["rel_trend_per_step"] == pytest.approx(0.0)


def test_sinuosity_metrics_strong_trend_is_not_stable():
    metrics = _sinuosity_stability_metrics(
        list(range(10)),
        np.linspace(1.0, 2.0, 10),
        window=5,
        rel_tol=5.0e-3,
    )

    assert metrics["state"] == "not stable"
    assert metrics["stable"] is False
    assert metrics["quasi_stable"] is False
    assert metrics["window_used"] == 5
