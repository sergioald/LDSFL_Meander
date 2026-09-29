"""Exact optional geometry unwrap tests and public-control coverage."""

from __future__ import annotations

import builtins
import json
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from ldsfl import geometry as geometry_module
from ldsfl import main as main_module
from ldsfl.gui_utils import (
    DimensionlessInputs,
    GeometrySettings,
    GuiCaseConfig,
    RunControls,
    config_from_dict,
    config_to_dict,
    validate_case_config,
)
from ldsfl.main import run_case
from ldsfl.mathutils import unwrap_angles_like_matlab


def _numba_unwrap():
    pytest.importorskip("numba")
    from ldsfl.geometry_numba import unwrap_angles_like_matlab_numba

    return unwrap_angles_like_matlab_numba


def _same_exact(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return np.asarray(float(left), dtype=np.float64).tobytes() == np.asarray(
            float(right), dtype=np.float64
        ).tobytes()
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_same_exact(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


_EDGE_ANGLES = np.array(
    [
        np.pi,
        -np.pi,
        np.nextafter(np.pi, 0.0),
        np.nextafter(np.pi, np.inf),
        np.nextafter(-np.pi, 0.0),
        np.nextafter(-np.pi, -np.inf),
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize(
    "angles",
    [
        np.array([], dtype=np.float64),
        np.array([0.25], dtype=np.float64),
        np.array([0.1, 0.3, 0.5, 0.2, -0.1], dtype=np.float64),
        np.sin(np.linspace(0.0, 8.0, 101)).astype(np.float64),
        np.array([0.5, 0.5, 0.5, 0.5, -0.5, -0.5], dtype=np.float64),
        np.array([2.9, -2.9, 2.8, -2.8, 3.0, -3.0], dtype=np.float64),
        _EDGE_ANGLES,
        np.array([10.0 * np.pi, -13.0 * np.pi, 8.25 * np.pi, -9.5 * np.pi], dtype=np.float64),
        np.array([np.nan, 0.5, -0.5], dtype=np.float64),
    ],
    ids=["empty", "length-one", "ordinary", "smooth", "repeated", "alternating-wraps", "pi-boundaries", "multiple-turns", "nan"],
)
def test_numba_unwrap_is_bitwise_equal_to_python_reference(angles):
    unwrap_numba = _numba_unwrap()
    expected = unwrap_angles_like_matlab(angles)
    actual = unwrap_numba(angles)
    assert np.array_equal(actual, expected, equal_nan=True)
    assert _same_exact(actual, expected)


def test_numba_unwrap_keeps_serial_non_fastmath_uncached_options():
    kernel = _numba_unwrap()
    assert kernel.targetoptions.get("cache", False) is False
    assert kernel.targetoptions.get("fastmath") is False
    assert kernel.targetoptions.get("parallel", False) is False


def _geometry_input(kind: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 24.0, 241, dtype=np.float64)
    if kind == "straight":
        return t, 0.15 * t
    if kind == "sinuous":
        return t, 3.0 * np.sin(t / 2.2) + 0.04 * t
    angle = np.linspace(0.0, 5.0 * np.pi, t.size, dtype=np.float64)
    return 0.08 * t + 1.4 * np.cos(angle), 1.4 * np.sin(angle)


@pytest.mark.parametrize("kind", ["straight", "sinuous", "looping"])
def test_geometry4_outputs_are_bitwise_equal_for_numba_unwrap(tmp_path, kind):
    _numba_unwrap()
    x, y = _geometry_input(kind)
    common = dict(
        xa=x,
        ya=y,
        jt=1,
        dsliminicial=1.0,
        id_files="geometry-test",
        Ntstep=10,
        cut_cnt=0,
        beta=12.0,
        base_out=tmp_path,
        neck_cutoff_interval=0,
        smoothing_enabled=False,
        do_plots=False,
    )
    reference = geometry_module.geometry4(**common, unwrap_backend="python")
    accelerated = geometry_module.geometry4(**common, unwrap_backend="numba")
    assert len(reference) == len(accelerated) == 11
    for expected, actual in zip(reference, accelerated, strict=True):
        if isinstance(expected, np.ndarray):
            assert _same_exact(expected, actual)
        elif isinstance(expected, (float, np.floating)):
            assert _same_exact(expected, actual)
        else:
            assert expected == actual


def test_default_geometry_path_does_not_attempt_to_import_numba(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def reject_numba(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise AssertionError("the Python geometry path attempted to import Numba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_numba)
    x, y = _geometry_input("sinuous")
    result = geometry_module.geometry4(
        x,
        y,
        1,
        1.0,
        "python-only",
        10,
        0,
        12.0,
        tmp_path,
        neck_cutoff_interval=0,
        smoothing_enabled=False,
        do_plots=False,
    )
    assert result[5] > 0


def test_requested_numba_without_dependency_fails_before_run_io(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "ldsfl.geometry_numba", raising=False)
    original_import = builtins.__import__

    def reject_numba(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ModuleNotFoundError("No module named 'numba'", name="numba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_numba)
    with pytest.raises(RuntimeError, match="geometry_unwrap_backend='numba'.*optional Numba"):
        run_case(tmp_path, 1, geometry_unwrap_backend="numba")
    assert not (tmp_path / "Output").exists()


def test_unrelated_missing_module_error_is_not_reported_as_missing_numba(monkeypatch):
    monkeypatch.delitem(sys.modules, "ldsfl.geometry_numba", raising=False)
    original_import = builtins.__import__

    def fail_unrelated_dependency(name, *args, **kwargs):
        if name == "numba":
            raise ModuleNotFoundError(
                "No module named 'unexpected_dependency'",
                name="unexpected_dependency",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_unrelated_dependency)
    with pytest.raises(ModuleNotFoundError, match="unexpected_dependency") as exc_info:
        geometry_module.ensure_geometry_unwrap_backend_available("numba")
    assert exc_info.value.name == "unexpected_dependency"


def test_invalid_geometry_unwrap_backend_fails_before_input_read(tmp_path):
    with pytest.raises(ValueError, match="Geometry unwrap backend"):
        run_case(tmp_path, 1, geometry_unwrap_backend="bad")
    assert not (tmp_path / "Output").exists()


def _prepare_case_dir(path: Path) -> None:
    input_dir = path / "Input"
    input_dir.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "Input"
    shutil.copy2(source / "Parameter.csv", input_dir / "Parameter.csv")
    shutil.copy2(source / "xy.csv", input_dir / "xy.csv")


def _run_short_case(path: Path, *, backend: str | None, import_guard=None):
    _prepare_case_dir(path)
    captures = []
    original_geometry4 = main_module.geometry4

    def recording_geometry4(*args, **kwargs):
        result = original_geometry4(*args, **kwargs)
        captures.append(
            tuple(value.copy() if isinstance(value, np.ndarray) else value for value in result)
        )
        return result

    with pytest.MonkeyPatch.context() as local_patch:
        local_patch.setattr(main_module, "geometry4", recording_geometry4)
        if import_guard is not None:
            local_patch.setattr(builtins, "__import__", import_guard)
        kwargs = {}
        if backend is not None:
            kwargs["geometry_unwrap_backend"] = backend
        result = run_case(
            path,
            1,
            Nprint=1,
            Max_Cut=0,
            max_steps=3,
            stop_on_steps=True,
            stop_on_cutoffs=False,
            do_plots=False,
            geometry_smoothing_enabled=False,
            flow_backend="numpy",
            **kwargs,
        )
    return result, captures


def _output_files(case_dir: Path, id_files: str) -> dict[str, bytes]:
    run_dir = case_dir / "Output" / id_files
    return {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "run_config.json"
    }


def _assert_short_runs_equal(reference, candidate, reference_captures, candidate_captures, ref_dir, candidate_dir):
    assert reference["steps"] == candidate["steps"] == 3
    for field in ("cut_cnt", "dt_cum", "stop_reason", "stop_criteria_reached", "sinuo_final"):
        assert _same_exact(reference[field], candidate[field]), field
    assert _same_exact(reference_captures, candidate_captures)
    assert _output_files(ref_dir, reference["id_files"]) == _output_files(candidate_dir, candidate["id_files"])


def test_run_case_default_matches_explicit_python_and_never_imports_numba(tmp_path):
    original_import = builtins.__import__

    def reject_numba(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise AssertionError("default run_case attempted to import Numba")
        return original_import(name, *args, **kwargs)

    default_dir = tmp_path / "default"
    explicit_dir = tmp_path / "explicit-python"
    default, default_captures = _run_short_case(default_dir, backend=None, import_guard=reject_numba)
    explicit, explicit_captures = _run_short_case(explicit_dir, backend="python", import_guard=reject_numba)
    _assert_short_runs_equal(
        default,
        explicit,
        default_captures,
        explicit_captures,
        default_dir,
        explicit_dir,
    )
    for run, run_dir in ((default, default_dir), (explicit, explicit_dir)):
        run_config_path = run_dir / "Output" / run["id_files"] / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        assert run_config["options"]["geometry_unwrap_backend"] == "python"


def test_run_case_python_and_numba_geometry_paths_are_bitwise_equal(tmp_path):
    _numba_unwrap()
    python_dir = tmp_path / "python"
    numba_dir = tmp_path / "numba"
    reference, reference_captures = _run_short_case(python_dir, backend="python")
    accelerated, accelerated_captures = _run_short_case(numba_dir, backend="NUMBA")
    _assert_short_runs_equal(
        reference,
        accelerated,
        reference_captures,
        accelerated_captures,
        python_dir,
        numba_dir,
    )
    for run, run_dir in ((reference, python_dir), (accelerated, numba_dir)):
        run_config_path = run_dir / "Output" / run["id_files"] / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        assert run_config["options"]["geometry_unwrap_backend"] == run["geometry_unwrap_backend"]
    assert accelerated["geometry_unwrap_backend"] == "numba"


def test_run_ldsfl_cli_forwards_default_and_explicit_geometry_backend(monkeypatch, tmp_path):
    import run_ldsfl

    captured = []
    monkeypatch.setattr(run_ldsfl, "run_project", lambda *args, **kwargs: captured.append(kwargs) or [])
    monkeypatch.setattr(sys, "argv", ["run_ldsfl.py", "--base-dir", str(tmp_path), "--no-plots"])
    run_ldsfl.main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ldsfl.py", "--base-dir", str(tmp_path), "--no-plots", "--geometry-unwrap-backend", "numba"],
    )
    run_ldsfl.main()
    assert captured[0]["geometry_unwrap_backend"] == "python"
    assert captured[1]["geometry_unwrap_backend"] == "numba"
    assert captured[0]["flow_backend"] == captured[1]["flow_backend"] == "numpy"


def test_run_ldsfl_cli_rejects_invalid_geometry_backend(monkeypatch, tmp_path):
    import run_ldsfl

    monkeypatch.setattr(sys, "argv", ["run_ldsfl.py", "--base-dir", str(tmp_path), "--geometry-unwrap-backend", "bad"])
    with pytest.raises(SystemExit):
        run_ldsfl.main()


def test_run_py_forwards_geometry_backend_independently(monkeypatch):
    import Run as convenience_launcher

    captured = {}
    monkeypatch.setattr(convenience_launcher, "run_project", lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["Run.py", "--geometry-unwrap-backend", "numba", "--backend", "numpy"])
    convenience_launcher.main()
    assert captured["geometry_unwrap_backend"] == "numba"
    assert captured["flow_backend"] == "numpy"


def _gui_config(tmp_path: Path, geometry_backend: str = "python") -> GuiCaseConfig:
    xy_csv = tmp_path / "xy.csv"
    xy_csv.write_text("0,0\n1,0.01\n2,0.03\n3,0.05\n4,0.08\n5,0.1\n", encoding="utf-8")
    return GuiCaseConfig(
        mode="dimensionless",
        xy_csv=xy_csv,
        workspace_dir=tmp_path,
        run=RunControls(geometry_unwrap_backend=geometry_backend),
        dimensionless=DimensionlessInputs(9.0, 0.005, 0.3, 2, 0.5, 6),
        geometry=GeometrySettings(),
    )


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_gui_config_round_trips_geometry_backend(tmp_path, backend):
    config = _gui_config(tmp_path, backend)
    restored = config_from_dict(config_to_dict(config))
    assert restored.run.geometry_unwrap_backend == backend
    assert validate_case_config(restored) == []


def test_gui_config_defaults_geometry_backend_to_python():
    assert RunControls().geometry_unwrap_backend == "python"


def test_legacy_gui_config_without_geometry_backend_defaults_to_python(tmp_path):
    legacy = config_to_dict(_gui_config(tmp_path))
    legacy["run"].pop("geometry_unwrap_backend")
    restored = config_from_dict(legacy)
    assert restored.run.geometry_unwrap_backend == "python"


def test_gui_worker_forwards_geometry_backend(monkeypatch, tmp_path):
    gui_module = pytest.importorskip("gui_ldsfl")
    config = _gui_config(tmp_path, "numba")
    captured = {}
    gui = gui_module.LdslGui.__new__(gui_module.LdslGui)
    gui.stop_requested_event = threading.Event()
    gui._log = lambda *_args, **_kwargs: None
    gui._finish_run = lambda _result: None
    gui._fail_run = lambda message, traceback: pytest.fail(traceback)
    gui.after = lambda _delay, callback: callback()
    monkeypatch.setattr(gui_module, "write_case_inputs", lambda _config: {})

    def fake_run_project(*args, **kwargs):
        captured.update(kwargs)
        return [{"id_files": "gui-geometry-backend"}]

    monkeypatch.setattr(gui_module, "run_project", fake_run_project)
    gui._run_case_worker(config)
    assert captured["geometry_unwrap_backend"] == "numba"
    assert captured["flow_backend"] == "numpy"
