"""Regression tests for avoiding unnecessary equivalence/HAC diagnostics."""

from __future__ import annotations

import shutil
from pathlib import Path

import ldsfl.main as main_mod

ROOT = Path(__file__).resolve().parents[1]


def _copy_inputs(tmp_path):
    input_dst = tmp_path / "Input"
    input_dst.mkdir()
    shutil.copy(ROOT / "Input" / "Parameter.csv", input_dst / "Parameter.csv")
    shutil.copy(ROOT / "Input" / "xy.csv", input_dst / "xy.csv")


def test_default_run_does_not_compute_expensive_equivalence_diagnostic(monkeypatch, tmp_path):
    _copy_inputs(tmp_path)
    calls = {"n": 0}

    def fake_equivalence(*args, **kwargs):
        calls["n"] += 1
        return {"stable": False, "state": "fake"}

    monkeypatch.setattr(main_mod, "sinuosity_equivalence_stability", fake_equivalence)

    result = main_mod.run_case(
        tmp_path,
        case_i=1,
        Nprint=5,
        Ntstep=10,
        Max_Cut=100,
        max_steps=3,
        stop_on_sinuosity_stability=False,
        do_plots=False,
    )

    assert calls["n"] == 0
    assert "equivalence" not in result["sinuosity_stability"]


def test_equivalence_diagnostic_is_available_when_explicitly_requested(monkeypatch, tmp_path):
    _copy_inputs(tmp_path)
    calls = {"n": 0}

    def fake_equivalence(*args, **kwargs):
        calls["n"] += 1
        return {"stable": True, "state": "fake"}

    monkeypatch.setattr(main_mod, "sinuosity_equivalence_stability", fake_equivalence)

    result = main_mod.run_case(
        tmp_path,
        case_i=1,
        Nprint=5,
        Ntstep=10,
        Max_Cut=100,
        max_steps=3,
        stop_on_sinuosity_stability=False,
        return_equivalence_stability=True,
        do_plots=False,
    )

    assert calls["n"] == 1
    assert result["sinuosity_stability"]["equivalence"]["stable"] is True
