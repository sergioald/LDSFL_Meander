"""Physics-oriented regression tests for the free-flow solver."""

from __future__ import annotations

import numpy as np

from ldsfl.flowfield import parall_u_free
from ldsfl.resistance import resistance_function_flagbed


def _reference_flow_inputs(n_points: int = 81):
    """Return a compact, deterministic free-flow test case."""
    beta = 9.0
    theta0 = 0.3
    ds = 0.005
    flagbed = 2
    rpic_0 = 0.5
    mdat = 4

    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(flagbed, theta0, ds, rpic_0)

    s = np.linspace(0.0, 20.0, n_points, dtype=np.float64)
    deltas = float(s[1] - s[0])
    n = np.array([1.0], dtype=np.float64)

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
        "Mdat": mdat,
        "Nn": 1,
        "Ns": n_points,
        "n": n,
        "deltas": deltas,
    }


def _run_free(c, *, paral: int = 0):
    kwargs = _reference_flow_inputs(len(c))
    return parall_u_free(
        np.asarray(c, dtype=np.float64),
        kwargs["s"], kwargs["Cf0"], kwargs["CT"], kwargs["CD"], kwargs["phiT"], kwargs["phiD"],
        kwargs["beta"], kwargs["rpic"], kwargs["theta0"], kwargs["F0"], kwargs["Mdat"], kwargs["Nn"],
        kwargs["Ns"], kwargs["n"], kwargs["deltas"], SL=0, paral=paral, n_workers=2, backend="numpy",
    )


def test_zero_curvature_produces_zero_velocity():
    c = np.zeros(81, dtype=np.float64)
    u, flag = _run_free(c)
    assert flag in (-1, 1)
    assert u.shape == c.shape
    assert np.all(np.isfinite(u))
    assert np.allclose(u, 0.0, atol=1.0e-12)


def test_free_flow_is_linear_in_curvature_amplitude():
    s = _reference_flow_inputs()["s"]
    c = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    u1, flag1 = _run_free(c)
    u2, flag2 = _run_free(2.0 * c)
    u_neg, flag_neg = _run_free(-c)
    assert flag1 == flag2 == flag_neg
    assert np.allclose(u2, 2.0 * u1, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(u_neg, -u1, rtol=1.0e-10, atol=1.0e-12)


def test_parallel_and_serial_numpy_free_flow_match():
    s = _reference_flow_inputs()["s"]
    c = 2.0e-3 * np.cos(3.0 * np.pi * s / s[-1])
    u_serial, flag_serial = _run_free(c, paral=0)
    u_parallel, flag_parallel = _run_free(c, paral=1)
    assert flag_serial == flag_parallel
    assert np.allclose(u_parallel, u_serial, rtol=1.0e-12, atol=1.0e-12)
