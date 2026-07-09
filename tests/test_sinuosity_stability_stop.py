"""Regression tests for stop-on-sinuosity-stability behaviour."""

from __future__ import annotations

import shutil
from pathlib import Path

from ldsfl.main import _combined_sinuosity_stability_metrics, run_case

ROOT = Path(__file__).resolve().parents[1]


def test_combined_stability_accepts_equivalence_parameters():
    steps = list(range(20))
    sinuosity = [1.25] * 20

    result = _combined_sinuosity_stability_metrics(
        steps,
        sinuosity,
        window=5,
        rel_tol=1.0e-6,
        equivalence_transient_step=0,
        equivalence_drift_tolerance=1.0e-6,
        equivalence_confidence=0.90,
        equivalence_min_points=10,
        equivalence_hac_lags=0,
    )

    assert "equivalence" in result
    assert result["equivalence"]["stable"] is True
    assert result["equivalence"]["analysis_start_step"] == 0.0


def test_run_case_can_stop_on_sinuosity_stability(tmp_path):
    input_dst = tmp_path / "Input"
    input_dst.mkdir()
    shutil.copy(ROOT / "Input" / "Parameter.csv", input_dst / "Parameter.csv")
    shutil.copy(ROOT / "Input" / "xy.csv", input_dst / "xy.csv")

    result = run_case(
        tmp_path,
        case_i=1,
        Nprint=5,
        Ntstep=10,
        Max_Cut=100,
        max_steps=20,
        stop_on_steps=True,
        stop_on_cutoffs=False,
        stop_on_sinuosity_stability=True,
        stop_mode="first",
        do_plots=False,
        sinuo_equiv_transient_step=0,
        sinuo_equiv_drift_tol=100.0,
        sinuo_equiv_confidence=0.90,
        sinuo_equiv_min_points=3,
        sinuo_equiv_hac_lags=0,
        sinuo_stability_interval=1,
    )

    assert "sinuosity_stability" in result["stop_criteria_reached"]
    assert result["steps"] < 20
    assert result["sinuosity_stability"]["equivalence"]["stable"] is True
