from __future__ import annotations

from pathlib import Path

from ldsfl.gui_utils import (
    DimensionlessInputs,
    GeometrySettings,
    GuiCaseConfig,
    RunControls,
    config_from_dict,
    config_to_dict,
)


def test_run_controls_default_does_not_stop_on_sinuosity_stability():
    assert RunControls().stop_on_sinuosity_stability is False


def test_stop_on_sinuosity_stability_round_trips_through_config_dict():
    cfg = GuiCaseConfig(
        mode="dimensionless",
        xy_csv=Path("Input/xy.csv"),
        workspace_dir=Path("."),
        run=RunControls(stop_on_sinuosity_stability=True),
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

    payload = config_to_dict(cfg)
    assert payload["run"]["stop_on_sinuosity_stability"] is True

    restored = config_from_dict(payload)
    assert restored.run.stop_on_sinuosity_stability is True
