"""Backend and numerical guardrail tests for free-flow evaluation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ldsfl.flowfield import parall_u_free
from ldsfl.resistance import resistance_function_flagbed


def _kwargs(n_points: int = 51):
    beta = 9.0
    theta0 = 0.3
    ds = 0.005
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(2, theta0, ds, 0.5)
    s = np.linspace(0.0, 10.0, n_points, dtype=np.float64)
    return {
        "s": s,
        "Cf0": cf0,
        "CT": ct,
        "CD": cd,
        "phiT": phit,
        "phiD": phid,
        "beta": beta,
        "rpic": rpic,
        "theta0": theta0,
        "F0": f0,
        "Mdat": 3,
        "Nn": 1,
        "Ns": n_points,
        "n": np.array([1.0], dtype=np.float64),
        "deltas": float(s[1] - s[0]),
    }


def _call(c, *, backend="numpy", sl=0):
    kwargs = _kwargs(len(c))
    return parall_u_free(
        np.asarray(c, dtype=np.float64),
        kwargs["s"], kwargs["Cf0"], kwargs["CT"], kwargs["CD"], kwargs["phiT"], kwargs["phiD"],
        kwargs["beta"], kwargs["rpic"], kwargs["theta0"], kwargs["F0"], kwargs["Mdat"], kwargs["Nn"],
        kwargs["Ns"], kwargs["n"], kwargs["deltas"], SL=sl, paral=0, backend=backend,
    )


def test_invalid_free_flow_backend_is_rejected():
    c = np.zeros(51, dtype=np.float64)
    with pytest.raises(ValueError, match="Unknown backend"):
        _call(c, backend="not-a-backend")


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba optional extra is not installed")
def test_numba_backend_matches_numpy_for_default_sl0_path():
    s = _kwargs()["s"]
    c = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    u_numpy, flag_numpy = _call(c, backend="numpy", sl=0)
    u_numba, flag_numba = _call(c, backend="numba", sl=0)
    assert flag_numpy == flag_numba
    assert np.allclose(u_numba, u_numpy, rtol=1.0e-9, atol=1.0e-11)


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba optional extra is not installed")
@pytest.mark.xfail(reason="Known review finding: SL=1 numba path can disagree with numpy; keep documented until investigated.")
def test_numba_sl1_matches_numpy_documented_known_issue():
    s = _kwargs()["s"]
    c = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    u_numpy, flag_numpy = _call(c, backend="numpy", sl=1)
    u_numba, flag_numba = _call(c, backend="numba", sl=1)
    assert flag_numpy == flag_numba
    assert np.allclose(u_numba, u_numpy, rtol=1.0e-9, atol=1.0e-11)
