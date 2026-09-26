"""Numba implementation of the reference vertical coefficient integration.

The equations, constants, grid size, and RK4 update mirror ``vertical.k0123``.
This module is imported lazily only when the Numba flow backend is selected.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .flowfield_numba import njit


@njit(cache=True, nogil=True)
def _fgj_nb(zrk, deltaz, yrk1, yrk2, ig, Cf0, kvk, A, B, z0, G0aux):
    aux1 = np.log(zrk / z0)
    aux2 = A * (zrk**2 - z0**2)
    aux3 = B * (zrk**3 - z0**3)
    uzero = (aux1 + aux2 + aux3) * np.sqrt(Cf0) / kvk

    if zrk < (1.0 - deltaz / 4.0):
        auxnu = 1.0 + 2.0 * A * (zrk**2) + 3.0 * B * (zrk**3)
        nuTzero = kvk * zrk * (1.0 - zrk) / auxnu
        Dauxnu = 4.0 * A * zrk + 9.0 * B * (zrk**2)
        DnuTzero = kvk * ((1.0 - 2.0 * zrk) * auxnu - zrk * (1.0 - zrk) * Dauxnu) / (auxnu**2)
    else:
        nuTzero = 0.0614
        DnuTzero = -0.0338

    if ig == 0:
        pf = 0.0
        p0 = 0.0
        p1 = DnuTzero / nuTzero
    elif ig == 1:
        pf = 1.0 / nuTzero
        p0 = 0.0
        p1 = DnuTzero / nuTzero
    elif ig == 2:
        pf = -(uzero**2) / nuTzero
        p0 = 0.0
        p1 = DnuTzero / nuTzero
    elif ig == 21:
        pf = (uzero * G0aux) / nuTzero
        p0 = 0.0
        p1 = DnuTzero / nuTzero
    else:
        raise ValueError("Unsupported ig value; expected 0, 1, 2, or 21")
    return -p1 * yrk2 - p0 * yrk1 + pf


@njit(cache=True, nogil=True)
def _rungeg_nb(zz, deltaz, yj1, yj2, ig, Cf0, kvk, A, B, z0, G0aux):
    fff = _fgj_nb(zz, deltaz, yj1, yj2, ig, Cf0, kvk, A, B, z0, G0aux)
    k11 = deltaz * yj2
    k12 = deltaz * fff

    zrk = zz + deltaz / 2.0
    yrk1 = yj1 + k11 / 2.0
    yrk2 = yj2 + k12 / 2.0
    fff = _fgj_nb(zrk, deltaz, yrk1, yrk2, ig, Cf0, kvk, A, B, z0, G0aux)
    k21 = deltaz * yrk2
    k22 = deltaz * fff

    zrk = zz + deltaz / 2.0
    yrk1 = yj1 + k21 / 2.0
    yrk2 = yj2 + k22 / 2.0
    fff = _fgj_nb(zrk, deltaz, yrk1, yrk2, ig, Cf0, kvk, A, B, z0, G0aux)
    k31 = deltaz * yrk2
    k32 = deltaz * fff

    zrk = zz + deltaz
    yrk1 = yj1 + k31
    yrk2 = yj2 + k32
    fff = _fgj_nb(zrk, deltaz, yrk1, yrk2, ig, Cf0, kvk, A, B, z0, G0aux)
    k41 = deltaz * yrk2
    k42 = deltaz * fff

    yjj1 = yj1 + (k11 + 2.0 * (k21 + k31) + k41) / 6.0
    yjj2 = yj2 + (k12 + 2.0 * (k22 + k32) + k42) / 6.0
    return yjj1, yjj2


@njit(cache=True, nogil=True)
def _trapz_nb(y, x):
    total = 0.0
    for i in range(y.size - 1):
        total += (y[i] + y[i + 1]) * (x[i + 1] - x[i]) * 0.5
    return total


@njit(cache=True, nogil=True)
def _k0123_kernel(Cf0):
    kvk = 0.41
    Nz = 1000
    A = 1.84
    B = -1.56
    z0 = np.exp(-kvk / np.sqrt(Cf0) - 0.777)

    xiini = 0.0
    xiend = -np.log(z0)
    deltaxi = (xiend - xiini) / (Nz - 1)
    g0 = np.zeros(Nz, dtype=np.float64)
    dg0 = np.ones(Nz, dtype=np.float64)
    g1 = np.zeros(Nz, dtype=np.float64)
    dg1 = np.ones(Nz, dtype=np.float64)
    g2 = np.zeros(Nz, dtype=np.float64)
    dg2 = np.ones(Nz, dtype=np.float64)
    zg = np.zeros(Nz, dtype=np.float64)
    u0 = np.zeros(Nz, dtype=np.float64)
    exp_dxi = np.exp(deltaxi)

    for j in range(Nz - 1):
        xi = xiini + j * deltaxi
        zg[j] = z0 * np.exp(xi)
        deltaz = zg[j] * (exp_dxi - 1.0)
        aux1 = np.log(zg[j] / z0)
        aux2 = A * (zg[j]**2 - z0**2)
        aux3 = B * (zg[j]**3 - z0**3)
        u0[j] = (aux1 + aux2 + aux3) * np.sqrt(Cf0) / kvk

        g0[j + 1], dg0[j + 1] = _rungeg_nb(zg[j], deltaz, g0[j], dg0[j], 0, Cf0, kvk, A, B, z0, 0.0)
        g1[j + 1], dg1[j + 1] = _rungeg_nb(zg[j], deltaz, g1[j], dg1[j], 1, Cf0, kvk, A, B, z0, 0.0)
        g2[j + 1], dg2[j + 1] = _rungeg_nb(zg[j], deltaz, g2[j], dg2[j], 2, Cf0, kvk, A, B, z0, 0.0)

    xi = xiini + (Nz - 1) * deltaxi
    zg[Nz - 1] = z0 * np.exp(xi)
    aux1 = np.log(zg[Nz - 1] / z0)
    aux2 = A * (zg[Nz - 1]**2 - z0**2)
    aux3 = B * (zg[Nz - 1]**3 - z0**3)
    u0[Nz - 1] = (aux1 + aux2 + aux3) * np.sqrt(Cf0) / kvk

    q1 = -dg1[Nz - 1] / dg0[Nz - 1]
    q2 = -dg2[Nz - 1] / dg0[Nz - 1]
    Int0 = _trapz_nb(g0, zg)
    Int1 = _trapz_nb(g1, zg)
    Int2 = _trapz_nb(g2, zg)
    ha0 = -(Int2 + q2 * Int0) / (Int1 + q1 * Int0)
    q0 = q1 * ha0 + q2
    G0 = q0 * g0 + ha0 * g1 + g2
    dG0 = q0 * dg0 + ha0 * dg1 + dg2

    g2[0] = 0.0
    dg2[0] = 1.0
    for j in range(Nz - 1):
        deltaz = zg[j] * (exp_dxi - 1.0)
        G0aux = 0.5 * (G0[j] + G0[j + 1])
        g2[j + 1], dg2[j + 1] = _rungeg_nb(
            zg[j], deltaz, g2[j], dg2[j], 21, Cf0, kvk, A, B, z0, G0aux
        )

    q2 = -dg2[Nz - 1] / dg0[Nz - 1]
    Int0 = _trapz_nb(g0, zg)
    Int1 = _trapz_nb(g1, zg)
    Int2 = _trapz_nb(g2, zg)
    ha1 = -(Int2 + q2 * Int0) / (Int1 + q1 * Int0)
    q0 = q1 * ha1 + q2
    G1 = q0 * g0 + ha1 * g1 + g2
    dG1 = q0 * dg0 + ha1 * dg1 + dg2

    k0 = _trapz_nb(G0 * u0, zg)
    k1 = _trapz_nb(G1 * u0, zg)
    Du0 = np.sqrt(Cf0) * (1.0 / z0 + 2.0 * A * z0 + 3.0 * B * (z0**2)) / kvk
    k2 = dG0[0] / Du0
    k3 = dG1[0] / Du0
    return k0, k1, k2, k3, u0, G0, G1, zg, z0


@lru_cache(maxsize=16)
def _k0123_cached(Cf0: float):
    """Exact-key cache; callers receive copies of cached array data."""
    return _k0123_kernel(Cf0)


def k0123_numba_cached(Cf0: float):
    """Return the full reference-compatible tuple using exact ``Cf0`` caching."""
    result = _k0123_cached(float(Cf0))
    return (*result[:4], result[4].copy(), result[5].copy(), result[6].copy(), result[7].copy(), result[8])


def k0123_coefficients_numba_cached(Cf0: float) -> tuple[float, float, float, float]:
    """Return cached scalar coefficients for modal precomputation."""
    result = _k0123_cached(float(Cf0))
    return float(result[0]), float(result[1]), float(result[2]), float(result[3])


def clear_k0123_cache() -> None:
    """Clear cached exact ``Cf0`` entries; useful for isolated benchmarks/tests."""
    _k0123_cached.cache_clear()


def k0123_cache_info():
    return _k0123_cached.cache_info()
