"""Regression tests for the second corrective review."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ldsfl.main as solver
import run_ldsfl
from ldsfl.gui_utils import (
    DimensionalInputs,
    GeometrySettings,
    GuiCaseConfig,
    RunControls,
    output_scales,
    validate_case_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _copy_inputs(workspace: Path) -> None:
    shutil.copytree(ROOT / "Input", workspace / "Input")


def test_gui_rejects_invalid_output_units(tmp_path):
    config = GuiCaseConfig(
        mode="dimensional",
        xy_csv=ROOT / "Input" / "xy.csv",
        workspace_dir=tmp_path,
        run=RunControls(output_units="dimensionl"),
        dimensional=DimensionalInputs(
            half_width=90.0,
            dref=10.0,
            d50=0.05,
            mobility_mode="direct_shields",
            theta0=0.3,
            flagbed=2,
            rpic_0=0.5,
            Mdat=6,
        ),
        geometry=GeometrySettings(
            mode="scale_by_dimensional_half_width",
        ),
    )

    with pytest.raises(ValueError, match="output_units"):
        output_scales(config)

    with pytest.raises(ValueError, match="output_units"):
        validate_case_config(config)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"Nprint": 0}, "Nprint"),
        ({"Nprint": 1.5}, "Nprint"),
        ({"Ntstep": 0}, "Ntstep"),
        ({"Max_Cut": -1}, "Max_Cut"),
        ({"Max_Cut": 1.5}, "Max_Cut"),
        ({"max_steps": -1}, "max_steps"),
        ({"max_steps": 1.5}, "max_steps"),
        ({"max_sim_time": np.inf}, "max_sim_time"),
        ({"max_sim_time": -1}, "max_sim_time"),
        ({"dsliminicial": 0}, "dsliminicial"),
        ({"ER": 0}, "ER"),
        ({"cstab": 0}, "cstab"),
        (
            {"geometry_smoothing_factor": np.nan},
            "geometry_smoothing_factor",
        ),
        ({"neck_cutoff_interval": -1}, "neck_cutoff_interval"),
        ({"resample_upper_factor": 1}, "resample_upper_factor"),
        ({"resample_lower_factor": 1}, "resample_lower_factor"),
        ({"sinuo_window": 1}, "sinuo_window"),
        ({"sinuo_rel_tol": 0}, "sinuo_rel_tol"),
        (
            {"sinuo_equiv_transient_step": -1},
            "sinuo_equiv_transient_step",
        ),
        (
            {"sinuo_equiv_drift_tol": 0},
            "sinuo_equiv_drift_tol",
        ),
        (
            {"sinuo_equiv_confidence": 1},
            "sinuo_equiv_confidence",
        ),
        (
            {"sinuo_equiv_min_points": 2},
            "sinuo_equiv_min_points",
        ),
        (
            {"sinuo_equiv_hac_lags": -1},
            "sinuo_equiv_hac_lags",
        ),
        (
            {"sinuo_equiv_method": "bad"},
            "sinuo_equiv_method",
        ),
        (
            {"sinuo_stability_interval": 0},
            "sinuo_stability_interval",
        ),
        ({"flow_bc": "bad"}, "flow_bc"),
        ({"flow_paral": 2}, "flow_paral"),
        ({"flow_workers": -1}, "flow_workers"),
        ({"flow_workers": 1.5}, "flow_workers"),
        ({"flow_backend": "bad"}, "flow_backend"),
        ({"output_units": "bad"}, "output_units"),
    ],
)
def test_run_case_rejects_invalid_core_controls_before_reading_inputs(
    tmp_path,
    override,
    message,
):
    with pytest.raises(ValueError, match=message):
        solver.run_case(tmp_path, 1, **override)


def test_cli_nprint_zero_fails_with_validation_error(monkeypatch, tmp_path):
    _copy_inputs(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ldsfl.py",
            "--base-dir",
            str(tmp_path),
            "--nprint",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="Nprint"):
        run_ldsfl.main()

    assert not (tmp_path / "Output").exists()


def test_dimensional_output_requires_explicit_positive_scales(tmp_path):
    cases = (
        (
            {"output_units": "dimensional"},
            "output_length_scale",
        ),
        (
            {
                "output_units": "dimensional",
                "output_length_scale": 2.0,
            },
            "output_velocity_scale",
        ),
        (
            {
                "output_units": "dimensional",
                "output_length_scale": 2.0,
                "output_velocity_scale": 0,
            },
            "output_velocity_scale",
        ),
    )

    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            solver.run_case(tmp_path, 1, **kwargs)


def test_gui_dimensional_output_rejects_inputs_without_reference_velocity(
    tmp_path,
):
    config = GuiCaseConfig(
        mode="dimensional",
        xy_csv=ROOT / "Input" / "xy.csv",
        workspace_dir=tmp_path,
        run=RunControls(output_units="dimensional"),
        dimensional=DimensionalInputs(
            half_width=90.0,
            dref=10.0,
            d50=0.05,
            mobility_mode="direct_shields",
            theta0=0.3,
            flagbed=2,
            rpic_0=0.5,
            Mdat=6,
        ),
        geometry=GeometrySettings(
            mode="scale_by_dimensional_half_width",
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires a reference velocity U0",
    ):
        output_scales(config)


def test_dimensional_run_scales_velocity_and_dimensionless_run_is_unchanged(
    tmp_path,
):
    dimensionless = tmp_path / "dimensionless"
    dimensional = tmp_path / "dimensional"

    _copy_inputs(dimensionless)
    _copy_inputs(dimensional)

    base = solver.run_case(
        dimensionless,
        1,
        max_steps=1,
        Nprint=10,
        do_plots=False,
    )

    physical = solver.run_case(
        dimensional,
        1,
        max_steps=1,
        Nprint=10,
        do_plots=False,
        output_units="dimensional",
        output_length_scale=2.0,
        output_velocity_scale=3.0,
    )

    base_file = next(
        (
            dimensionless
            / "Output"
            / base["id_files"]
            / "xyu"
        ).glob("*.csv")
    )

    physical_file = next(
        (
            dimensional
            / "Output"
            / physical["id_files"]
            / "xyu"
        ).glob("*.csv")
    )

    base_frame = pd.read_csv(base_file)
    physical_frame = pd.read_csv(physical_file)

    np.testing.assert_allclose(
        physical_frame[["x", "y", "s"]],
        base_frame[["x", "y", "s"]] * 2.0,
    )

    np.testing.assert_allclose(
        physical_frame["U"],
        base_frame["U"] * 3.0,
    )

    np.testing.assert_allclose(
        physical_frame["c"],
        base_frame["c"] / 2.0,
    )


def test_no_plots_writes_incremental_complete_sinuosity_csv_without_png(
    tmp_path,
):
    _copy_inputs(tmp_path)

    result = solver.run_case(
        tmp_path,
        1,
        max_steps=7,
        Nprint=2,
        do_plots=False,
    )

    root = tmp_path / "Output" / result["id_files"]

    assert list(root.rglob("*.png")) == []

    history = pd.read_csv(
        root
        / "files"
        / f"sinuosity_history_{result['id_files']}.csv"
    )

    assert history["step"].tolist() == list(range(8))
    assert history["step"].is_unique


@pytest.mark.parametrize(
    "target, expected_error",
    [
        ("save_xystcu", "final xyu snapshot"),
        ("save_variables", "final variable history"),
        ("save_sinuosity_history", "final sinuosity CSV"),
    ],
)
def test_required_final_output_failure_is_fatal_and_recorded(
    monkeypatch,
    tmp_path,
    target,
    expected_error,
):
    _copy_inputs(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("injected output failure")

    monkeypatch.setattr(solver, target, fail)

    with pytest.raises(RuntimeError, match=expected_error):
        solver.run_case(
            tmp_path,
            1,
            max_steps=1,
            Nprint=10,
            do_plots=False,
        )

    run_root = next((tmp_path / "Output").iterdir())

    status = json.loads(
        (run_root / "run_status.json").read_text(
            encoding="utf-8",
        )
    )

    assert status["simulation_completed"] is True
    assert status["output_complete"] is False

    assert any(
        expected_error in item
        for item in status["output_errors"]
    )