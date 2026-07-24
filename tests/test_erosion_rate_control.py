from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import run_ldsfl
from ldsfl.evolution import dxdy2
from ldsfl.gui_utils import (
    DimensionlessInputs,
    GeometrySettings,
    GuiCaseConfig,
    RunControls,
    config_from_dict,
    config_to_dict,
    preview_case_config,
    validate_case_config,
    write_case_inputs,
)
from ldsfl.main import run_case


def _xy_csv(tmp_path: Path) -> Path:
    path = tmp_path / "xy.csv"
    path.write_text(
        "0,0\n1,0.01\n2,0.03\n3,0.06\n4,0.08\n5,0.09\n",
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, *, erosion_rate: float = 1.0e-8) -> GuiCaseConfig:
    return GuiCaseConfig(
        mode="dimensionless",
        xy_csv=_xy_csv(tmp_path),
        workspace_dir=tmp_path,
        run=RunControls(
            case_id=1,
            nprint=10,
            ntstep=100000,
            max_cut=1,
            max_steps=1,
            stop_on_steps=True,
            stop_on_time=False,
            stop_on_cutoffs=False,
            erosion_rate=erosion_rate,
            do_plots=False,
            save_run_manifest=False,
        ),
        dimensionless=DimensionlessInputs(
            beta=9.0,
            ds=0.005,
            theta0=0.3,
            flagbed=2,
            rpic_0=0.5,
            Mdat=6,
        ),
        geometry=GeometrySettings(),
    )


def test_run_controls_default_preserves_previous_erosion_rate():
    assert RunControls().erosion_rate == pytest.approx(1.0e-8)


def test_gui_config_round_trips_erosion_rate(tmp_path):
    cfg = _config(tmp_path, erosion_rate=2.5e-8)
    restored = config_from_dict(config_to_dict(cfg))
    assert restored.run.erosion_rate == pytest.approx(2.5e-8)


def test_preview_reports_erosion_rate(tmp_path):
    preview = preview_case_config(_config(tmp_path, erosion_rate=2.5e-8))
    assert preview["erosion_rate"] == pytest.approx(2.5e-8)


@pytest.mark.parametrize("erosion_rate", [0.0, -1.0e-8, float("nan"), float("inf")])
def test_gui_validation_rejects_invalid_erosion_rate(tmp_path, erosion_rate):
    with pytest.raises(ValueError, match="Erosion rate"):
        validate_case_config(_config(tmp_path, erosion_rate=erosion_rate))


@pytest.mark.parametrize("value", ["0", "-1e-8", "nan", "inf"])
def test_cli_type_rejects_invalid_erosion_rate(value):
    with pytest.raises(argparse.ArgumentTypeError, match="erosion rate"):
        run_ldsfl._positive_finite_erosion_rate(value)


def test_cli_forwards_erosion_rate(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run_project(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run_ldsfl, "run_project", fake_run_project)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ldsfl.py",
            "--base-dir",
            str(tmp_path),
            "--erosion-rate",
            "2.5e-8",
        ],
    )

    run_ldsfl.main()

    assert captured["ER"] == pytest.approx(2.5e-8)


def test_doubling_erosion_rate_halves_dt_but_preserves_step_displacement():
    U = np.array([0.2, -0.1, 0.3], dtype=float)
    theta = np.array([0.0, 0.2, -0.1], dtype=float)
    x = np.zeros_like(U)
    y = np.zeros_like(U)

    dx1, dy1, dt1 = dxdy2(1.0e-8, U, x, y, theta, 1.0, 3, 3, 1)
    dx2, dy2, dt2 = dxdy2(2.0e-8, U, x, y, theta, 1.0, 3, 3, 1)

    assert dt2 == pytest.approx(0.5 * dt1)
    assert dx2 == pytest.approx(2.0 * dx1)
    assert dy2 == pytest.approx(2.0 * dy1)
    assert dx2 * dt2 == pytest.approx(dx1 * dt1)
    assert dy2 * dt2 == pytest.approx(dy1 * dt1)


def test_run_case_rejects_invalid_erosion_rate_before_solver_work(tmp_path):
    with pytest.raises(ValueError, match="Erosion rate"):
        run_case(tmp_path, 1, ER=0.0)


def test_run_summary_reports_selected_erosion_rate(tmp_path):
    cfg = _config(tmp_path, erosion_rate=2.5e-8)
    write_case_inputs(cfg)

    result = run_case(
        tmp_path,
        1,
        Nprint=10,
        Ntstep=100,
        max_steps=1,
        stop_on_steps=True,
        stop_on_time=False,
        stop_on_cutoffs=False,
        do_plots=False,
        ER=cfg.run.erosion_rate,
    )

    assert result["erosion_rate"] == pytest.approx(2.5e-8)


def test_gui_worker_forwards_erosion_rate_without_tk(monkeypatch, tmp_path):
    gui_ldsfl = pytest.importorskip("gui_ldsfl")
    captured: dict = {}

    def fake_run_project(*args, **kwargs):
        captured.update(kwargs)
        return [{"id_files": "test", "stop_reason": "done"}]

    gui = gui_ldsfl.LdslGui.__new__(gui_ldsfl.LdslGui)
    gui.stop_requested_event = threading.Event()
    gui._log = lambda *_args, **_kwargs: None
    gui._finish_run = lambda _result: None
    gui._fail_run = lambda *_args, **_kwargs: None
    gui.after = lambda _delay, callback: callback()

    monkeypatch.setattr(gui_ldsfl, "write_case_inputs", lambda _config: {})
    monkeypatch.setattr(
        gui_ldsfl,
        "output_scales",
        lambda _config: {
            "resolved_output_units": "dimensionless",
            "output_length_scale": 1.0,
            "output_velocity_scale": 1.0,
        },
    )
    monkeypatch.setattr(gui_ldsfl, "run_project", fake_run_project)

    gui._run_case_worker(_config(tmp_path, erosion_rate=2.5e-8))

    assert captured["ER"] == pytest.approx(2.5e-8)

def test_gui_required_erosion_rate_rejects_blank_without_tk():
    gui_ldsfl = pytest.importorskip("gui_ldsfl")

    class _DummyVar:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    gui = gui_ldsfl.LdslGui.__new__(gui_ldsfl.LdslGui)

    with pytest.raises(ValueError, match="Erosion rate is required"):
        gui._required_float_from_var(_DummyVar("   "), "Erosion rate")

    assert gui._required_float_from_var(
        _DummyVar("2.5e-8"),
        "Erosion rate",
    ) == pytest.approx(2.5e-8)

