"""Exactness, optional-dependency, and routing tests for the Numba neck grid."""

from __future__ import annotations

import builtins
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from ldsfl import geometry, mathutils
from ldsfl import main as main_module
from ldsfl.main import run_case
from ldsfl.validation import validate_run_controls


def _numba_detector():
    pytest.importorskip("numba")
    from ldsfl.neck_numba import spatial_grid_first_hit_point_pair

    return spatial_grid_first_hit_point_pair


def test_numba_grid_keeps_serial_non_fastmath_uncached_options():
    pytest.importorskip("numba")
    from ldsfl.neck_numba import _spatial_grid_first_hit_kernel

    assert _spatial_grid_first_hit_kernel.targetoptions.get("cache", False) is False
    assert _spatial_grid_first_hit_kernel.targetoptions.get("fastmath") is False
    assert _spatial_grid_first_hit_kernel.targetoptions.get("parallel", False) is False


def test_unsafe_grid_coordinates_delegate_to_kdtree_reference(monkeypatch):
    detector = _numba_detector()
    from ldsfl.neck_numba import _spatial_grid_first_hit_kernel

    x = np.array([1.0e20, 3.0e20, 1.0e20, 4.0e20, 5.0e20, 6.0e20])
    y = np.zeros_like(x)
    ss = 2
    radius = 1.0
    cell_width = radius * (1.0 + 16.0 * np.finfo(np.float64).eps)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert x[0] / cell_width > np.iinfo(np.int64).max
    assert _spatial_grid_first_hit_kernel(x, y, ss, radius, cell_width) == (-2, -2)

    reference = mathutils._kdtree_first_hit_point_pair
    expected = reference(x, y, ss, radius)
    calls = []

    def observe_reference(xa, ya, separation, threshold):
        calls.append((xa, ya, separation, threshold))
        return reference(xa, ya, separation, threshold)

    monkeypatch.setattr(mathutils, "_kdtree_first_hit_point_pair", observe_reference)
    assert detector(x, y, ss, radius) == expected == (0, 2)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], x)
    np.testing.assert_array_equal(calls[0][1], y)
    assert calls[0][2:] == (ss, radius)


def test_safe_grid_coordinates_do_not_delegate_to_kdtree(monkeypatch):
    detector = _numba_detector()
    x, y = _pair_case(16, 0, 3, 0.25)
    monkeypatch.setattr(
        mathutils,
        "_kdtree_first_hit_point_pair",
        lambda *_args, **_kwargs: pytest.fail("safe grid search used KDTree fallback"),
    )
    assert detector(x, y, 3, 1.0) == (0, 3)


def _separated(n: int) -> tuple[np.ndarray, np.ndarray]:
    return 10.0 * np.arange(n, dtype=np.float64), np.zeros(n, dtype=np.float64)


def _pair_case(n: int, i: int, j: int, distance: float):
    x, y = _separated(n)
    x[j] = x[i] + distance
    return x, y


def _detector_cases():
    cases = []
    cases.append(("no-hit", *_separated(64), 3, 1.0, None))
    cases.append(("immediate", *_pair_case(64, 0, 3, 0.25), 3, 1.0, (0, 3)))
    cases.append(("middle", *_pair_case(64, 20, 23, 0.25), 3, 1.0, (20, 23)))
    cases.append(("late", *_pair_case(80, 72, 75, 0.25), 3, 1.0, (72, 75)))
    cases.append(("dense", np.zeros(64), np.zeros(64), 3, 1.0, (0, 3)))
    repeated = np.repeat(np.arange(8, dtype=np.float64) * 10.0, 8)
    cases.append(("repeated", repeated, np.zeros(64), 3, 1.0, (0, 3)))
    x, y = _separated(16)
    x[11] = x[7] + 0.5
    cases.append(("domain-boundary", x, y, 4, 1.0, (7, 11)))

    x, y = _separated(64)
    x[3], x[4] = 0.8, 0.2
    cases.append(("closest-j", x, y, 3, 1.0, (0, 4)))
    x, y = _separated(64)
    x[3], x[4] = 0.5, -0.5
    cases.append(("equal-distance-smallest-j", x, y, 3, 1.0, (0, 3)))

    cases.append(("exact-threshold", *_pair_case(16, 0, 3, 1.0), 3, 1.0, None))
    cases.append(("nextafter-below", *_pair_case(16, 0, 3, np.nextafter(1.0, 0.0)), 3, 1.0, (0, 3)))
    cases.append(("nextafter-above", *_pair_case(16, 0, 3, np.nextafter(1.0, np.inf)), 3, 1.0, None))
    for ss in (1, 2, 4):
        cases.append((f"ss-{ss}", *_pair_case(16, 0, ss, 0.25), ss, 1.0, (0, ss)))
    cases.append(("small-four-point", *_pair_case(4, 0, 1, 0.25), 1, 1.0, (0, 1)))
    cases.append(("too-small", np.arange(3.0), np.zeros(3), 1, 1.0, None))
    return cases


@pytest.mark.parametrize(
    "label,x,y,ss,radius,expected",
    _detector_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_spatial_grid_point_pair_matches_reference(label, x, y, ss, radius, expected):
    detector = _numba_detector()
    reference = mathutils._kdtree_first_hit_point_pair(x, y, ss, radius)
    accelerated = detector(x, y, ss, radius)
    assert reference == expected, label
    assert accelerated == expected, label


def test_refinement_path_and_refinement_edge_rejections_match():
    _numba_detector()
    real_cases = [
        (*_pair_case(64, 0, 3, 0.25), 3, 1.0),
        (*_separated(32), 3, 1.0),
    ]
    for x, y, ss, radius in real_cases:
        reference = mathutils.find_neck_cutoff_kdtree_with_refine(x, y, ss, radius)
        accelerated = mathutils.find_neck_cutoff_numba_grid_with_refine(x, y, ss, radius)
        assert accelerated == reference

    x = np.array([0.0, 5.0, 10.0, 0.25, 20.0, 25.0])
    y = np.zeros_like(x)
    calls = []

    def mapped_hit_detector(xv, yv, ss_ref, radius):
        calls.append((xv.copy(), int(ss_ref)))
        if len(calls) == 1:
            return None
        # These refined indices map to original points 0 and 3.
        return 1, 38

    mapped_hit = mathutils._find_neck_cutoff_with_refine_impl(
        x,
        y,
        2,
        1.0,
        point_pair_detector=mapped_hit_detector,
    )
    assert len(calls) == 2
    assert calls[1][0].size > x.size
    assert mapped_hit == (0, 3)

    def rejected_result(xv_original, yv_original, ss_original, points):
        calls = []

        def detector_with_forced_refined_hit(xv, yv, ss_ref, radius):
            calls.append((xv.copy(), int(ss_ref)))
            if len(calls) == 1:
                return None
            return points

        result = mathutils._find_neck_cutoff_with_refine_impl(
            xv_original,
            yv_original,
            ss_original,
            1.0,
            point_pair_detector=detector_with_forced_refined_hit,
        )
        assert len(calls) == 2
        return result

    separated_x, separated_y = _separated(6)
    # Maps to a valid index domain but fails the final original-coordinate distance test.
    assert rejected_result(separated_x, separated_y, 3, (1, 41)) is None

    domain_x = np.array([0.0, 5.0, 0.2, 10.0, 15.0, 20.0])
    domain_y = np.zeros_like(domain_x)
    # Maps to original points 1 and 2, which violate the original ss=3 constraint.
    assert rejected_result(domain_x, domain_y, 3, (10, 20)) is None


def _base_controls(**overrides):
    values = {
        "Nprint": 1,
        "Ntstep": 1,
        "Max_Cut": 0,
        "max_steps": 1,
        "max_sim_time": None,
        "dsliminicial": 1.0,
        "ER": 1.0e-8,
        "cstab": 0.01,
        "geometry_smoothing_factor": 8.0,
        "neck_cutoff_interval": 3,
        "resample_upper_factor": 1.03,
        "resample_lower_factor": 0.97,
        "sinuo_window": 100,
        "sinuo_rel_tol": 5.0e-3,
        "sinuo_equiv_transient_step": 40_000.0,
        "sinuo_equiv_drift_tol": 0.02,
        "sinuo_equiv_confidence": 0.90,
        "sinuo_equiv_min_points": 10,
        "sinuo_equiv_hac_lags": 50,
        "sinuo_equiv_method": "increment",
        "sinuo_stability_interval": 100,
        "flow_bc": "free",
        "flow_paral": 0,
        "flow_workers": 0,
        "flow_backend": "numpy",
        "geometry_unwrap_backend": "python",
        "neck_detector_backend": "kdtree",
        "output_units": "dimensionless",
        "output_length_scale": None,
        "output_velocity_scale": None,
        "stop_mode": "first",
    }
    values.update(overrides)
    return validate_run_controls(**values)


def test_backend_controls_are_independent_and_normalized():
    controls = _base_controls(
        neck_detector_backend="NUMBA_GRID",
        flow_backend="numpy",
        geometry_unwrap_backend="python",
    )
    assert controls["neck_detector_backend"] == "numba_grid"
    assert _base_controls(
        flow_backend="numba",
        geometry_unwrap_backend="numba",
        neck_detector_backend="kdtree",
    )["neck_detector_backend"] == "kdtree"
    with pytest.raises(ValueError, match="neck_detector_backend.*kdtree, numba_grid"):
        _base_controls(neck_detector_backend="auto")


def test_run_project_normalizes_and_forwards_neck_backend(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(main_module, "read_parameter_table", lambda _path: object())
    monkeypatch.setattr(main_module, "dimensionless_input_table", lambda _table, _case: None)
    monkeypatch.setattr(
        main_module,
        "ensure_neck_detector_backend_available",
        lambda backend: str(backend).lower(),
    )

    def fake_run_case(*args, **kwargs):
        captured.update(kwargs)
        return {"neck_detector_backend": kwargs["neck_detector_backend"]}

    monkeypatch.setattr(main_module, "run_case", fake_run_case)
    results = main_module.run_project(
        tmp_path,
        cases=[1],
        neck_detector_backend="NUMBA_GRID",
    )
    assert captured["neck_detector_backend"] == "numba_grid"
    assert results == [{"neck_detector_backend": "numba_grid"}]


def test_default_neck_backend_does_not_import_numba(monkeypatch):
    original_import = builtins.__import__

    def reject_numba(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise AssertionError("KDTree default attempted to import Numba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_numba)
    assert geometry.ensure_neck_detector_backend_available("KDTREE") == "kdtree"


def _reject_numba_import(original_import):
    def reject(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ModuleNotFoundError("No module named 'numba'", name="numba")
        return original_import(name, *args, **kwargs)

    return reject


def test_explicit_numba_grid_without_dependency_fails_before_run_output(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "ldsfl.neck_numba", raising=False)
    monkeypatch.setattr(builtins, "__import__", _reject_numba_import(builtins.__import__))
    with pytest.raises(RuntimeError, match="neck_detector_backend='numba_grid'.*optional Numba"):
        run_case(tmp_path, 1, neck_detector_backend="numba_grid")
    assert not (tmp_path / "Output").exists()


def test_geometry4_missing_numba_grid_fails_before_processing_or_output(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "ldsfl.neck_numba", raising=False)
    monkeypatch.setattr(builtins, "__import__", _reject_numba_import(builtins.__import__))
    monkeypatch.setattr(
        geometry,
        "matlab_spline",
        lambda *_args, **_kwargs: pytest.fail("spline processing began before backend validation"),
    )
    with pytest.raises(RuntimeError, match="neck_detector_backend='numba_grid'.*optional Numba"):
        geometry.geometry4(
            np.arange(8.0),
            np.zeros(8),
            1,
            1.0,
            "missing-numba",
            10,
            0,
            12.0,
            tmp_path,
            do_plots=False,
            neck_detector_backend="numba_grid",
        )
    assert list(tmp_path.iterdir()) == []


def test_unrelated_missing_module_error_is_re_raised(monkeypatch):
    monkeypatch.delitem(sys.modules, "ldsfl.neck_numba", raising=False)
    original_import = builtins.__import__

    def fail_for_other_dependency(name, *args, **kwargs):
        if name == "numba":
            raise ModuleNotFoundError("No module named 'other_dependency'", name="other_dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_for_other_dependency)
    with pytest.raises(ModuleNotFoundError, match="other_dependency") as exc_info:
        geometry.ensure_neck_detector_backend_available("numba_grid")
    assert exc_info.value.name == "other_dependency"


def test_invalid_neck_backend_fails_before_run_output(tmp_path):
    with pytest.raises(ValueError, match="Neck detector backend.*kdtree, numba_grid"):
        run_case(tmp_path, 1, neck_detector_backend="bad")
    assert not (tmp_path / "Output").exists()


def _prepare_case_dir(path: Path) -> None:
    input_dir = path / "Input"
    input_dir.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "Input"
    shutil.copy2(source / "Parameter.csv", input_dir / "Parameter.csv")
    shutil.copy2(source / "xy.csv", input_dir / "xy.csv")


def test_public_run_case_vertical_controls_with_numba_grid(tmp_path):
    _numba_detector()
    reference_dir = tmp_path / "reference"
    accelerated_dir = tmp_path / "accelerated"
    _prepare_case_dir(reference_dir)
    _prepare_case_dir(accelerated_dir)
    common = {
        "Nprint": 100,
        "Max_Cut": 0,
        "max_steps": 2,
        "stop_on_steps": True,
        "stop_on_cutoffs": False,
        "do_plots": False,
        "geometry_smoothing_enabled": False,
        "flow_backend": "numpy",
        "geometry_unwrap_backend": "python",
    }
    reference = run_case(reference_dir, 1, neck_detector_backend="kdtree", **common)
    accelerated = run_case(accelerated_dir, 1, neck_detector_backend="NUMBA_GRID", **common)
    assert reference["steps"] == accelerated["steps"] == 2
    for field in ("cut_cnt", "dt_cum", "sinuo_final", "resonance"):
        assert reference[field] == accelerated[field]
    assert accelerated["neck_detector_backend"] == "numba_grid"
    run_config = (
        accelerated_dir
        / "Output"
        / accelerated["id_files"]
        / "run_config.json"
    )
    import json

    assert json.loads(run_config.read_text(encoding="utf-8"))["options"]["neck_detector_backend"] == "numba_grid"
