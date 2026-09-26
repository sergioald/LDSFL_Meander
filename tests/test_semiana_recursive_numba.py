"""Parity tests for the validated recursive finite-window SEMIANA response."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from shutil import copy2

import numpy as np
import pytest

from ldsfl import flowfield as flowfield_module
from ldsfl.inputs import dimensionless_input_table, read_parameter_table
from ldsfl.resistance import resistance_function_flagbed

pytestmark = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba optional extra is not installed")


@pytest.mark.parametrize("n_points", [5, 17, 61, 257])
@pytest.mark.parametrize("direction", ["upstream", "downstream"])
@pytest.mark.parametrize("real_part", [0.02, 0.7])
def test_recursive_finite_convolution_matches_cached_legacy(n_points, direction, real_part):
    from ldsfl.flowfield_numba import (
        _cached_tables_for_lam_np,
        _fill_dwstr_recursive_real_nb,
        _fill_upstr_recursive_real_nb,
        _semiana1_cached_nb,
        _semiana2_cached_nb,
    )

    rng = np.random.default_rng(7314 + n_points)
    c_pad = np.zeros(n_points + 1, dtype=np.float64)
    c_pad[1:] = rng.normal(scale=1.0e-3, size=n_points)
    lam = (real_part + 0.43j) if direction == "upstream" else (-real_part + 0.43j)
    jtolls = sorted({2, 3, 5, max(3, n_points // 2), n_points - 1, n_points, n_points + 1})

    for jtoll in jtolls:
        kmax = max(1, min(n_points, jtoll - 1))
        cA, cB, lmds, exp_table, kmax = _cached_tables_for_lam_np(
            lam, c_pad, 0.1, kmax, allow_pos_k=direction == "downstream",
        )
        real = np.zeros(n_points + 1, dtype=np.float64)
        neg_imag = np.zeros_like(real)
        if direction == "upstream":
            fill = _fill_upstr_recursive_real_nb
            stations = range(1, n_points)
            expected = np.array([
                _semiana1_cached_nb(js, min(js + jtoll - 1, n_points), cA, cB, lmds, exp_table, kmax)
                for js in stations
            ])
        else:
            fill = _fill_dwstr_recursive_real_nb
            stations = range(2, n_points + 1)
            expected = np.array([
                _semiana2_cached_nb(js, max(js - jtoll + 1, 1), cA, cB, lmds, exp_table, kmax)
                for js in stations
            ])
        fill(real, cA, cB, lmds, exp_table, kmax, jtoll, n_points, 1.0 + 0.0j)
        fill(neg_imag, cA, cB, lmds, exp_table, kmax, jtoll, n_points, 0.0 + 1.0j)
        actual = real[list(stations)] - 1j * neg_imag[list(stations)]
        np.testing.assert_allclose(actual, expected, rtol=5.0e-11, atol=5.0e-13)


def _flow_args(n_points: int, beta: float) -> tuple:
    root = Path(__file__).resolve().parents[1]
    _case_beta, ds, theta0, flagbed, rpic_0, mdat = dimensionless_input_table(
        read_parameter_table(root / "Input" / "Parameter.csv"), 1,
    )
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(flagbed, theta0, ds, rpic_0)
    s = np.linspace(0.0, 10.0, n_points)
    c = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    return c, s, cf0, ct, cd, phit, phid, beta, rpic, theta0, f0, mdat, 1, n_points, np.array([1.0]), s[1] - s[0]


def _three_responses(monkeypatch, n_points: int, sl: int, beta: float):
    args = _flow_args(n_points, beta)
    opts = {"SL": sl, "paral": 0, "vertical_backend": "numba", "numba_parallel": False, "numba_fastmath": False}
    reference, reference_flag = flowfield_module.parall_u_free(*args, backend="numpy", **opts)
    original = flowfield_module._run_modes_numba
    selected_strategies = []

    def observe_public_strategy(**kwargs):
        selected_strategies.append(kwargs["strategy"])
        return original(**kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(flowfield_module, "_run_modes_numba", observe_public_strategy)
        public, public_flag = flowfield_module.parall_u_free(*args, backend="numba", **opts)

    def run_explicit(strategy: str):
        def select_strategy(**kwargs):
            kwargs["strategy"] = strategy
            return original(**kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(flowfield_module, "_run_modes_numba", select_strategy)
            return flowfield_module.parall_u_free(*args, backend="numba", **opts)

    legacy, legacy_flag = run_explicit("legacy")
    recursive, recursive_flag = run_explicit("recursive")
    return (
        (reference, reference_flag),
        (public, public_flag),
        (legacy, legacy_flag),
        (recursive, recursive_flag),
        selected_strategies,
    )


@pytest.mark.parametrize("n_points", [51, 250, 1000])
def test_complete_sl0_flow_matches_legacy_and_numpy(monkeypatch, n_points):
    reference, public, legacy, recursive, selected = _three_responses(monkeypatch, n_points, 0, 9.0)

    assert selected == ["recursive"]
    assert reference[1] == public[1] == legacy[1] == recursive[1]
    np.testing.assert_allclose(public[0], recursive[0], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(public[0], reference[0], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(recursive[0], legacy[0], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(legacy[0], reference[0], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(recursive[0], reference[0], rtol=1.0e-9, atol=1.0e-11)


@pytest.mark.parametrize("n_points", [51, 250])
@pytest.mark.parametrize("beta, expected_flag", [(6.0, 1), (12.0, -1)])
def test_complete_sl1_flow_matches_legacy_and_numpy(monkeypatch, n_points, beta, expected_flag):
    reference, public, legacy, recursive, selected = _three_responses(monkeypatch, n_points, 1, beta)

    assert selected == ["recursive"]
    assert reference[1] == public[1] == legacy[1] == recursive[1] == expected_flag
    np.testing.assert_allclose(public[0], recursive[0], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(public[0], reference[0], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(recursive[0], legacy[0], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(legacy[0], reference[0], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(recursive[0], reference[0], rtol=1.0e-9, atol=1.0e-11)


@pytest.mark.parametrize(
    "paral, numba_parallel, numba_fastmath",
    [(1, False, False), (0, True, False), (0, False, True)],
)
def test_public_numba_keeps_legacy_for_unvalidated_execution_modes(
    monkeypatch, paral, numba_parallel, numba_fastmath,
):
    selected = []

    def capture_strategy(**kwargs):
        selected.append(kwargs["strategy"])
        return np.zeros(len(kwargs["c_pad"]), dtype=np.float64)

    monkeypatch.setattr(flowfield_module, "_run_modes_numba", capture_strategy)
    flowfield_module.parall_u_free(
        *_flow_args(51, 9.0),
        SL=0,
        paral=paral,
        backend="numba",
        vertical_backend="numba",
        numba_parallel=numba_parallel,
        numba_fastmath=numba_fastmath,
    )

    assert selected == ["legacy"]


def test_sl1_borderline_fallback_uses_sl0_response():
    from ldsfl.flowfield_numba import get_mode_adders

    add_sl0, add_sl1 = get_mode_adders(parallel=False, fastmath=False)
    n_points = 51
    s = np.linspace(0.0, 10.0, n_points)
    c_pad = np.zeros(n_points + 1)
    s_pad = np.zeros(n_points + 1)
    c_pad[1:] = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    s_pad[1:] = s
    am = np.array([0.2])
    lambs = (0.7 + 0.2j, 0.0 + 0.5j, -0.5 + 0.3j, -0.4 + 0.1j)
    g0s = (0.1 + 0.2j, -0.2 + 0.1j, 0.3 - 0.1j, 0.2 + 0.1j)
    args = (1, am, lambs, g0s, 0.15 + 0.04j, c_pad, s_pad, s[1] - s[0], True, 1.0e-4)
    sl0 = np.zeros(n_points + 1)
    sl1_recursive = np.zeros_like(sl0)
    add_sl0(sl0, *args, True)
    add_sl1(sl1_recursive, *args, True)

    np.testing.assert_allclose(sl1_recursive, sl0, rtol=1.0e-12, atol=1.0e-13)


@pytest.mark.parametrize(
    "parallel, fastmath, use_cached",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_direct_recursive_adders_reject_unsupported_modes(parallel, fastmath, use_cached):
    from ldsfl.flowfield_numba import get_mode_adders

    adders = get_mode_adders(parallel=parallel, fastmath=fastmath)
    c_pad = np.zeros(5, dtype=np.float64)
    s_pad = np.arange(5, dtype=np.float64)
    args = (
        1,
        np.array([0.2]),
        (0.7 + 0.2j, 0.2 + 0.1j, -0.5 + 0.3j, -0.4 + 0.1j),
        (0.1 + 0.2j, -0.2 + 0.1j, 0.3 - 0.1j, 0.2 + 0.1j),
        0.15 + 0.04j,
        c_pad,
        s_pad,
        1.0,
        use_cached,
        1.0e-4,
        True,
    )
    for add_mode in adders:
        with pytest.raises(ValueError, match="recursive SEMIANA requires"):
            add_mode(np.zeros_like(c_pad), *args)


def test_three_step_solver_smoke_matches_explicit_recursive(monkeypatch, tmp_path):
    from ldsfl.main import run_case

    input_source = Path(__file__).resolve().parents[1] / "Input"

    def run(label):
        base = tmp_path / label
        inputs = base / "Input"
        inputs.mkdir(parents=True)
        copy2(input_source / "Parameter.csv", inputs / "Parameter.csv")
        copy2(input_source / "xy.csv", inputs / "xy.csv")
        return run_case(
            base, case_i=1, Nprint=10, Ntstep=10, Max_Cut=100,
            max_steps=3, stop_on_steps=True, stop_on_time=False,
            stop_on_cutoffs=True, do_plots=False, flow_backend="numba",
        )

    public = run("public")
    original = flowfield_module._run_modes_numba

    def select_recursive(**kwargs):
        kwargs["strategy"] = "recursive"
        return original(**kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(flowfield_module, "_run_modes_numba", select_recursive)
        recursive = run("recursive")

    assert public["steps"] == recursive["steps"] == 3
    assert public["cut_cnt"] == recursive["cut_cnt"]
    np.testing.assert_allclose(recursive["sinuo_final"], public["sinuo_final"], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(recursive["dt_cum"], public["dt_cum"], rtol=1.0e-9, atol=1.0e-11)
