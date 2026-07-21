from __future__ import annotations

import numpy as np
import pytest

import ldsfl.main as main_mod
from ldsfl.main import _final_snapshot_velocity


def _geometry(n: int):
    x = np.linspace(0.0, 1.0, n)
    y = np.zeros(n)
    s = np.linspace(0.0, 1.0, n)
    th = np.zeros(n)
    c = np.zeros(n)
    return x, y, s, th, c


def _call_kwargs(n: int, **overrides):
    x, y, s, th, c = _geometry(n)
    kwargs = {
        "x": x,
        "y": y,
        "s": s,
        "th": th,
        "c": c,
        "Cf0": 0.01,
        "CT": 0.02,
        "CD": 0.03,
        "phiT": 0.04,
        "phiD": 0.05,
        "beta": 9.0,
        "rpic": 0.5,
        "theta0": 0.3,
        "F0": 1.0,
        "Mdat": 3,
        "n1": np.array([1.0], dtype=np.float64),
        "deltas": 0.005,
        "flow_bc": "free",
        "flow_paral": 0,
        "n_workers": 0,
        "flow_backend": "numpy",
        "numba_parallel": False,
        "numba_fastmath": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_final_snapshot_velocity_recomputes_free_flow_when_lengths_match(monkeypatch):
    calls = {}

    def fake_free(
        c,
        s,
        Cf0,
        CT,
        CD,
        phiT,
        phiD,
        beta,
        rpic,
        theta0,
        F0,
        Mdat,
        parameter,
        Ns,
        n1,
        deltas,
        **kwargs,
    ):
        calls["Ns"] = Ns
        calls["backend"] = kwargs.get("backend")
        calls["SL"] = kwargs.get("SL")
        return np.array([10.0, 11.0, 12.0], dtype=np.float64), 0

    monkeypatch.setattr(main_mod, "parall_u_free", fake_free)

    stale_U = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = _final_snapshot_velocity(stale_U, **_call_kwargs(3))

    assert np.array_equal(out, np.array([10.0, 11.0, 12.0]))
    assert calls == {"Ns": 3, "backend": "numpy", "SL": 0}


def test_final_snapshot_velocity_recomputes_free_flow_when_length_mismatches(monkeypatch):
    calls = {}

    def fake_free(
        c,
        s,
        Cf0,
        CT,
        CD,
        phiT,
        phiD,
        beta,
        rpic,
        theta0,
        F0,
        Mdat,
        parameter,
        Ns,
        n1,
        deltas,
        **kwargs,
    ):
        calls["Ns"] = Ns
        calls["backend"] = kwargs.get("backend")
        calls["SL"] = kwargs.get("SL")
        return np.arange(Ns, dtype=np.float64), 0

    monkeypatch.setattr(main_mod, "parall_u_free", fake_free)

    out = _final_snapshot_velocity(np.zeros(5), **_call_kwargs(3))

    assert np.array_equal(out, np.array([0.0, 1.0, 2.0]))
    assert calls == {"Ns": 3, "backend": "numpy", "SL": 0}


def test_final_snapshot_velocity_recomputes_periodic_flow_when_length_mismatches(monkeypatch):
    calls = {}

    def fake_periodic(
        c,
        s,
        Cf0,
        CT,
        CD,
        phiT,
        phiD,
        beta,
        rpic,
        theta0,
        F0,
        Mdat,
        parameter,
        Ns,
        n1,
        deltas,
        **kwargs,
    ):
        calls["Ns"] = Ns
        calls["paral"] = kwargs.get("paral")
        return np.full(Ns, 7.0, dtype=np.float64), 0

    monkeypatch.setattr(main_mod, "parall_u_periodic", fake_periodic)

    out = _final_snapshot_velocity(np.zeros(1), **_call_kwargs(4, flow_bc="periodic", flow_paral=1))

    assert np.array_equal(out, np.full(4, 7.0))
    assert calls == {"Ns": 4, "paral": 1}


def test_final_snapshot_velocity_rejects_inconsistent_final_geometry():
    kwargs = _call_kwargs(3)
    kwargs["y"] = np.zeros(2)

    with pytest.raises(ValueError, match="Final geometry arrays have inconsistent lengths"):
        _final_snapshot_velocity(np.zeros(3), **kwargs)
