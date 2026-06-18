from __future__ import annotations

import warnings

import numpy as np
import pytest

from ldsfl.evolution import dxdy2, update_parameters


def test_dxdy2_computes_velocity_components_and_stable_timestep():
    theta = np.array([0.0, np.pi / 2.0])
    velocity = np.array([2.0, 3.0])

    dxdtl, dydtl, dt = dxdy2(
        0.5,
        velocity,
        np.zeros_like(velocity),
        np.zeros_like(velocity),
        theta,
        deltas=1.0,
        Nsold=0,
        Ns=2,
        jt=1,
        cstab=0.03,
    )

    np.testing.assert_allclose(dxdtl, [0.0, 1.5], atol=1.0e-12)
    np.testing.assert_allclose(dydtl, [1.0, 0.0], atol=1.0e-12)
    assert dt == pytest.approx(0.03 / 1.5)


def test_dxdy2_zero_velocity_uses_fallback_timestep_and_warns():
    with pytest.warns(RuntimeWarning, match="zero or very small"):
        dxdtl, dydtl, dt = dxdy2(
            1.0,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            deltas=1.0,
            Nsold=0,
            Ns=3,
            jt=7,
            cstab=0.01,
        )

    np.testing.assert_allclose(dxdtl, np.zeros(3))
    np.testing.assert_allclose(dydtl, np.zeros(3))
    assert dt == pytest.approx(0.01)


def test_dxdy2_non_finite_velocity_raises_without_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(FloatingPointError, match="Non-finite migration speed"):
            dxdy2(
                1.0,
                np.array([1.0, np.inf]),
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                deltas=1.0,
                Nsold=0,
                Ns=2,
                jt=1,
            )


def test_update_parameters_identity_case_keeps_values_unchanged():
    beta, theta0, ds = update_parameters(
        Cf0_old=0.01,
        Cf0=0.01,
        wave_l_old=10.0,
        wave_l=10.0,
        valle_l_old=5.0,
        valle_l=5.0,
        beta_old=9.0,
        ds_old=0.005,
        theta0_old=0.3,
    )

    assert beta == pytest.approx(9.0)
    assert theta0 == pytest.approx(0.3)
    assert ds == pytest.approx(0.005)


def test_update_parameters_known_scaling():
    beta, theta0, ds = update_parameters(
        Cf0_old=4.0,
        Cf0=1.0,
        wave_l_old=10.0,
        wave_l=20.0,
        valle_l_old=5.0,
        valle_l=20.0,
        beta_old=9.0,
        ds_old=0.006,
        theta0_old=0.3,
    )

    d_cf = 4.0
    d_ls = (20.0 / 20.0) / (5.0 / 10.0)
    assert beta == pytest.approx(9.0 * d_ls ** (1.0 / 3.0) * d_cf ** (1.0 / 3.0))
    assert theta0 == pytest.approx(0.3 * d_ls ** (2.0 / 3.0) * d_cf ** (-1.0 / 3.0))
    assert ds == pytest.approx(0.006 * d_ls ** (1.0 / 3.0) * d_cf ** (1.0 / 3.0))
