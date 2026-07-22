from __future__ import annotations

from pathlib import Path

import pytest

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
from ldsfl.main import _combined_sinuosity_stability_metrics, run_case


def _xy_csv(tmp_path: Path) -> Path:
    path = tmp_path / "xy.csv"
    path.write_text("0,0\n1,0.01\n2,0.03\n3,0.06\n4,0.08\n5,0.09\n", encoding="utf-8")
    return path


def _config(tmp_path: Path, **run_overrides) -> GuiCaseConfig:
    values = {
        "case_id": 1,
        "nprint": 2,
        "ntstep": 100000,
        "max_cut": 1,
        "max_steps": 5,
        "stop_on_steps": True,
        "stop_on_time": False,
        "stop_on_cutoffs": True,
        "stop_on_sinuosity_stability": False,
        "sinuo_equiv_transient_step": None,
        "sinuo_equiv_drift_tol": 0.03,
        "sinuo_equiv_confidence": 0.85,
        "sinuo_equiv_min_points": 4,
        "sinuo_equiv_hac_lags": 7,
        "sinuo_equiv_method": "hac",
        "sinuo_stability_interval": 3,
    }
    values.update(run_overrides)
    return GuiCaseConfig(
        mode="dimensionless",
        xy_csv=_xy_csv(tmp_path),
        workspace_dir=tmp_path,
        run=RunControls(**values),
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


def test_gui_config_round_trips_equivalence_controls(tmp_path):
    cfg = _config(tmp_path)
    restored = config_from_dict(config_to_dict(cfg))

    assert restored.run.sinuo_equiv_transient_step is None
    assert restored.run.sinuo_equiv_drift_tol == pytest.approx(0.03)
    assert restored.run.sinuo_equiv_confidence == pytest.approx(0.85)
    assert restored.run.sinuo_equiv_min_points == 4
    assert restored.run.sinuo_equiv_hac_lags == 7
    assert restored.run.sinuo_equiv_method == "hac"
    assert restored.run.sinuo_stability_interval == 3


def test_config_loader_ignores_unknown_run_fields(tmp_path):
    payload = config_to_dict(_config(tmp_path))
    payload["run"]["future_unknown_field"] = "ignored"

    restored = config_from_dict(payload)

    assert isinstance(restored.run, RunControls)
    assert restored.run.sinuo_equiv_method == "hac"


def test_preview_reports_equivalence_controls(tmp_path):
    preview = preview_case_config(_config(tmp_path))

    assert preview["sinuo_equiv_transient_step"] is None
    assert preview["sinuo_equiv_drift_tol"] == pytest.approx(0.03)
    assert preview["sinuo_equiv_confidence"] == pytest.approx(0.85)
    assert preview["sinuo_equiv_min_points"] == 4
    assert preview["sinuo_equiv_hac_lags"] == 7
    assert preview["sinuo_equiv_method"] == "hac"
    assert preview["sinuo_stability_interval"] == 3


@pytest.mark.parametrize(
    "override, message",
    [
        ({"sinuo_equiv_drift_tol": 0.0}, "Equivalence drift tolerance"),
        ({"sinuo_equiv_confidence": 1.0}, "Equivalence confidence"),
        ({"sinuo_equiv_min_points": 2}, "Equivalence minimum points"),
        ({"sinuo_equiv_hac_lags": -1}, "Equivalence HAC lags"),
        ({"sinuo_equiv_method": "bad"}, "Equivalence method"),
        ({"sinuo_stability_interval": 0}, "Sinuosity stability check interval"),
    ],
)
def test_validate_rejects_invalid_equivalence_controls(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        validate_case_config(_config(tmp_path, **override))


def test_main_rejects_runs_with_no_enabled_stop_criterion(tmp_path):
    cfg = _config(
        tmp_path,
        stop_on_steps=False,
        stop_on_time=False,
        stop_on_cutoffs=False,
        stop_on_sinuosity_stability=False,
    )
    write_case_inputs(cfg)

    with pytest.raises(ValueError, match="At least one stop criterion"):
        run_case(
            tmp_path,
            1,
            stop_on_steps=False,
            stop_on_time=False,
            stop_on_cutoffs=False,
            stop_on_sinuosity_stability=False,
        )


def test_combined_stability_forwards_equivalence_method():
    steps = list(range(20))
    sinuosity = [1.0 + 1.0e-5 * step for step in steps]

    result = _combined_sinuosity_stability_metrics(
        steps,
        sinuosity,
        equivalence_transient_step=None,
        equivalence_min_points=4,
        equivalence_method="hac",
    )

    assert result["equivalence"]["method"] == "hac"
