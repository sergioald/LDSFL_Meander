from __future__ import annotations

import numpy as np
import pytest

gui_ldsfl = pytest.importorskip("gui_ldsfl")


class _DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return str(self.value)


def _make_gui_without_tk(*, window=5, rel_tol=5.0e-3):
    gui = gui_ldsfl.LdslGui.__new__(gui_ldsfl.LdslGui)
    gui.sinuo_window_var = _DummyVar(window)
    gui.sinuo_rel_tol_var = _DummyVar(rel_tol)
    return gui


def test_gui_sinuosity_metrics_compute_stable_state_without_full_tk_app():
    gui = _make_gui_without_tk(window=5, rel_tol=5.0e-3)

    stability = gui._compute_sinuosity_stability_from_arrays(
        np.arange(10, dtype=float),
        np.full(10, 2.5, dtype=float),
    )

    assert stability["state"] == "stable"
    assert stability["stable"] is True
    assert stability["quasi_stable"] is True
    assert stability["window_used"] == 5
    assert stability["sinuo_final"] == pytest.approx(2.5)


def test_gui_sinuosity_metrics_compute_not_stable_state_without_full_tk_app():
    gui = _make_gui_without_tk(window=5, rel_tol=5.0e-3)

    stability = gui._compute_sinuosity_stability_from_arrays(
        np.arange(10, dtype=float),
        np.linspace(1.0, 2.0, 10, dtype=float),
    )

    assert stability["state"] == "not stable"
    assert stability["stable"] is False
    assert stability["quasi_stable"] is False
    assert stability["window_used"] == 5
    assert stability["sinuo_final"] == pytest.approx(2.0)

def test_gui_raw_equivalence_transient_step_preserves_blank_without_default():
    gui = _make_gui_without_tk(window=5, rel_tol=5.0e-3)
    gui.sinuo_equiv_transient_step_var = _DummyVar("")

    assert gui._raw_tk_string("sinuo_equiv_transient_step_var", "") == ""
    assert gui._safe_tk_string("sinuo_equiv_transient_step_var", "40000") == "40000"

