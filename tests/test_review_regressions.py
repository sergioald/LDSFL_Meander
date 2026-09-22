"""Regression coverage for units, isolated runs, case selection and stop limits."""

from __future__ import annotations

import copy
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ldsfl.main as solver
from ldsfl.gui_utils import (
    DimensionalInputs,
    GeometrySettings,
    GuiCaseConfig,
    RunControls,
    output_scales,
    validate_case_config,
)
from ldsfl.inputs import read_parameter_table
from ldsfl.outputs import reserve_run_directory

ROOT = Path(__file__).resolve().parents[1]


def dimensional_config(tmp_path):
    return GuiCaseConfig(
        mode="dimensional",
        xy_csv=ROOT / "Input" / "xy.csv",
        workspace_dir=tmp_path,
        run=RunControls(output_units="dimensional"),
        dimensional=DimensionalInputs(
            half_width=90.0,
            dref=10.0,
            d50=0.05,
            flagbed=2,
            rpic_0=0.5,
            Mdat=6,
            mobility_mode="direct_shear_stress",
            tau_b=20.0,
            velocity=2.0,
        ),
        geometry=GeometrySettings(mode="scale_by_dimensional_half_width"),
    )


def copy_inputs(tmp_path):
    shutil.copytree(ROOT / "Input", tmp_path / "Input")


def test_shear_stress_in_pascals_matches_other_mobility_methods(tmp_path):
    inp = dimensional_config(tmp_path).dimensional
    expected = 20.0 / ((2650.0 - 1000.0) * 9.81 * 0.05)
    assert inp.resolved_theta0() == pytest.approx(expected)
    slope = copy.deepcopy(inp)
    slope.mobility_mode = "depth_slope_grain"
    slope.slope = 20.0 / (1000.0 * 9.81 * 10.0)
    assert slope.resolved_theta0() == pytest.approx(expected)
    velocity = copy.deepcopy(inp)
    velocity.mobility_mode = "depth_velocity_friction_grain"
    velocity.friction_value = 0.005
    velocity.velocity = np.sqrt(20.0 / (1000.0 * 0.005))
    assert velocity.resolved_theta0() == pytest.approx(expected)


@pytest.mark.parametrize("have_result", [True, False])
def test_gui_continuation_preserves_output_scale_and_solver_coordinates(tmp_path, have_result):
    gui_module = pytest.importorskip("gui_ldsfl")
    cfg = dimensional_config(tmp_path)
    snapshot = tmp_path / "snapshot.csv"
    pd.DataFrame({"x": [0.0, 90.0, 180.0], "y": [0.0, 45.0, 0.0]}).to_csv(snapshot, index=False)
    gui = gui_module.LdslGui.__new__(gui_module.LdslGui)
    gui.run_in_progress = False
    gui.latest_config = cfg
    gui.latest_result = {"output_units": "dimensional", "output_length_scale": 90.0} if have_result else None
    gui._latest_xyu_snapshot_path = lambda: snapshot
    gui._log = lambda *_: None
    captured = {}
    gui._run_case_threaded = lambda **kwargs: captured.update(kwargs)
    gui._show_error = lambda exc: pytest.fail(str(exc))
    gui._continue_from_latest_output()
    continuation = captured["config_override"]
    assert captured["continuation"] is True
    assert continuation.geometry.mode == "as_is"
    assert output_scales(continuation)["output_length_scale"] == 90.0
    xy = pd.read_csv(continuation.xy_csv, header=None).to_numpy()
    np.testing.assert_allclose(xy, [[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]])
    # Multiplying continuation outputs by their physical scale restores metres.
    np.testing.assert_allclose(
        xy * output_scales(continuation)["output_length_scale"], pd.read_csv(snapshot).to_numpy()
    )
    gui._update_plot = lambda **kwargs: None
    gui._refresh_initial_plot(continuation)
    np.testing.assert_allclose(np.column_stack(gui.initial_xy), pd.read_csv(snapshot).to_numpy())


def test_repeated_runs_preserve_previous_files_and_separate_histories(tmp_path):
    copy_inputs(tmp_path)
    announced = []
    first = solver.run_case(tmp_path, 1, max_steps=5, Nprint=2, do_plots=False, run_started_callback=announced.append)
    first_root = tmp_path / "Output" / first["id_files"]
    before = {p.relative_to(first_root): p.read_bytes() for p in first_root.rglob("*") if p.is_file()}
    second = solver.run_case(tmp_path, 1, max_steps=2, Nprint=2, do_plots=False, run_started_callback=announced.append)
    assert first["id_files"] != second["id_files"]
    assert announced == [first["id_files"], second["id_files"]]
    assert before == {p.relative_to(first_root): p.read_bytes() for p in first_root.rglob("*") if p.is_file()}
    second_root = tmp_path / "Output" / second["id_files"]
    rows = pd.concat(pd.read_csv(p) for p in (second_root / "files").glob("var_*.csv"))
    assert sorted(rows.state_step) == [0.0, 1.0, 2.0]
    config = json.loads((second_root / "run_config.json").read_text(encoding="utf-8"))
    assert config["options"]["max_steps"] == 2
    assert config["options"]["ER"] == 1e-8
    assert config["parameters"]["Mdat"] > 0
    assert len(config["input_sha256"]["xy.csv"]) == 64


def test_colliding_legacy_labels_and_concurrent_runs_get_distinct_folders(tmp_path):
    label1 = solver.make_id_files(1, 4.5, 0.005, 0.3, 2, 0.5)
    label2 = solver.make_id_files(1, 45.0, 0.005, 0.3, 2, 0.5)
    # Keep historical readable labels, but never use them as unique run keys.
    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = list(executor.map(lambda label: reserve_run_directory(tmp_path, label), [label1, label2] * 4))
    assert len(set(ids)) == 8
    assert all((tmp_path / run_id / "xyu").is_dir() for run_id in ids)


def test_run_all_uses_actual_ids_in_table_order(tmp_path, monkeypatch):
    copy_inputs(tmp_path)
    df = pd.read_csv(tmp_path / "Input" / "Parameter.csv").iloc[[0, 0]].copy()
    df["Id"] = [2, 3]
    df["Beta"] = [9.0, 12.0]
    df.to_csv(tmp_path / "Input" / "Parameter.csv", index=False)
    calls = []
    monkeypatch.setattr(solver, "run_case", lambda base, case_i, **kw: calls.append(case_i))
    solver.run_project(tmp_path)
    assert calls == [2, 3]
    calls.clear()
    with pytest.raises(ValueError, match="case ID 1"):
        solver.run_project(tmp_path, cases=[2, 1])
    assert calls == []


@pytest.mark.parametrize("ids", [[2, 2], [0, 3], [1.5, 3], [float("nan"), 3]])
def test_parameter_reader_rejects_invalid_case_ids(tmp_path, ids):
    path = tmp_path / "Parameter.csv"
    pd.DataFrame(
        {
            "Id": ids,
            "Beta": [9.0, 9.0],
            "ds": [0.005, 0.005],
            "Thetha": [0.3, 0.3],
            "flagbed": [2, 2],
            "r": [0.5, 0.5],
            "Mdat": [6, 6],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Id"):
        read_parameter_table(path)


@pytest.mark.parametrize("disabled_limit", [0])
def test_all_stop_mode_ignores_disabled_time_and_cutoff_limits(tmp_path, disabled_limit):
    copy_inputs(tmp_path)
    # Safety callback prevents a regression from hanging the test indefinitely.
    polls = iter(range(10))
    result = solver.run_case(
        tmp_path,
        1,
        max_steps=1,
        Nprint=10,
        do_plots=False,
        stop_mode="all",
        max_sim_time=disabled_limit,
        Max_Cut=disabled_limit,
        stop_on_time=True,
        stop_requested_callback=lambda: next(polls) >= 3,
    )
    assert result["steps"] == 1
    assert result["stop_criteria_reached"] == ["max_steps"]


def test_all_stop_mode_ignores_disabled_step_limit(tmp_path):
    copy_inputs(tmp_path)
    polls = iter(range(10))
    result = solver.run_case(
        tmp_path,
        1,
        max_steps=None,
        max_sim_time=1e-12,
        Max_Cut=0,
        stop_on_time=True,
        stop_mode="all",
        Nprint=10,
        do_plots=False,
        stop_requested_callback=lambda: next(polls) >= 3,
    )
    assert result["steps"] == 1
    assert result["stop_criteria_reached"] == ["max_sim_time"]


def test_no_effective_stop_criteria_rejected_before_inputs_are_read(tmp_path):
    with pytest.raises(ValueError, match="At least one stop criterion"):
        solver.run_case(tmp_path, 1, max_steps=0, Max_Cut=0)
    cfg = dimensional_config(tmp_path)
    cfg.run.max_steps = cfg.run.max_cut = 0
    with pytest.raises(ValueError, match="At least one stop criterion"):
        validate_case_config(cfg)


def test_moving_window_does_not_read_the_full_history():
    class WindowOnlyHistory:
        def __len__(self):
            return 1_000_000

        def __getitem__(self, key):
            assert isinstance(key, slice), "Must not iterate the full history"
            assert key.start == -100
            return [2.5] * 100

    result = solver._sinuosity_stability_metrics(WindowOnlyHistory(), WindowOnlyHistory(), window=100)
    assert result["stable"] is True
    assert result["window_used"] == 100


def test_gui_worker_monitors_the_reserved_run_directory(tmp_path, monkeypatch):
    gui_module = pytest.importorskip("gui_ldsfl")
    gui = gui_module.LdslGui.__new__(gui_module.LdslGui)
    gui.stop_requested_event = threading.Event()
    gui._log = lambda *_: None
    gui.after = lambda delay, callback: callback()
    gui._finish_run = lambda result: None
    gui._fail_run = lambda message, tb: pytest.fail(tb)
    cfg = dimensional_config(tmp_path)
    cfg.run.save_run_manifest = False
    monkeypatch.setattr(gui_module, "write_case_inputs", lambda config: {})

    def fake_run_project(*args, **kwargs):
        kwargs["run_started_callback"]("reserved_run2")
        return [{"id_files": "reserved_run2"}]

    monkeypatch.setattr(gui_module, "run_project", fake_run_project)
    gui._run_case_worker(cfg)
    assert gui.current_id_files == "reserved_run2"
    assert gui.current_xyu_dir == tmp_path / "Output" / "reserved_run2" / "xyu"
