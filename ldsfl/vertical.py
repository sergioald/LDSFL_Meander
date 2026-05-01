# ldsfl/vertical.py
from __future__ import annotations
import numpy as np


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    # NumPy 2.x: trapezoid; older: trapz
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def fgj(zrk: float, deltaz: float, yrk1: float, yrk2: float,
        ig: int, Cf0: float, kvk: float, A: float, B: float, z0: float, G0aux: float) -> float:
    """
    Port of fgj.m
    """
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
        raise ValueError(f"Unsupported ig={ig}")

    return float(-p1 * yrk2 - p0 * yrk1 + pf)


def rungeg(zz: float, deltaz: float, yj1: float, yj2: float,
           ig: int, Cf0: float, kvk: float, A: float, B: float, z0: float, G0aux: float) -> tuple[float, float]:
    """
    Port of Rungeg.m (RK4 for a 2nd-order ODE written as a 1st-order system).
    """
    # k1
    fff = fgj(zz, deltaz, yj1, yj2, ig, Cf0, kvk, A, B, z0, G0aux)
    k11 = deltaz * yj2
    k12 = deltaz * fff

    # k2
    zrk = zz + deltaz / 2.0
    yrk1 = yj1 + k11 / 2.0
    yrk2 = yj2 + k12 / 2.0
    fff = fgj(zrk, deltaz, yrk1, yrk2, ig, Cf0, kvk, A, B, z0, G0aux)
    k21 = deltaz * yrk2
    k22 = deltaz * fff

    # k3
    zrk = zz + deltaz / 2.0
    yrk1 = yj1 + k21 / 2.0
    yrk2 = yj2 + k22 / 2.0
    fff = fgj(zrk, deltaz, yrk1, yrk2, ig, Cf0, kvk, A, B, z0, G0aux)
    k31 = deltaz * yrk2
    k32 = deltaz * fff

    # k4
    zrk = zz + deltaz
    yrk1 = yj1 + k31
    yrk2 = yj2 + k32
    fff = fgj(zrk, deltaz, yrk1, yrk2, ig, Cf0, kvk, A, B, z0, G0aux)
    k41 = deltaz * yrk2
    k42 = deltaz * fff

    yjj1 = yj1 + (k11 + 2.0 * (k21 + k31) + k41) / 6.0
    yjj2 = yj2 + (k12 + 2.0 * (k22 + k32) + k42) / 6.0
    return float(yjj1), float(yjj2)


def k0123(Cf0: float):
    """
    Port of k0123.m (MATLAB version in your ZIP).
    Returns: k0,k1,k2,k3,u0,G0,G1,zg,z0
    """
    kvk = 0.41
    Nz = 1000
    A = 1.84
    B = -1.56

    z0 = float(np.exp(-kvk / np.sqrt(Cf0) - 0.777))

    xiini = 0.0
    xiend = -float(np.log(z0))
    deltaxi = (xiend - xiini) / (Nz - 1)

    # Arrays (match MATLAB: g*=zeros, dg*=ones)
    g0 = np.zeros(Nz, dtype=np.float64)
    dg0 = np.ones(Nz, dtype=np.float64)
    g1 = np.zeros(Nz, dtype=np.float64)
    dg1 = np.ones(Nz, dtype=np.float64)
    g2 = np.zeros(Nz, dtype=np.float64)
    dg2 = np.ones(Nz, dtype=np.float64)

    zg = np.zeros(Nz, dtype=np.float64)
    u0 = np.zeros(Nz, dtype=np.float64)

    # --- First pass: build g0,g1,g2 and u0 ---
    exp_dxi = float(np.exp(deltaxi))

    for j in range(0, Nz - 1):
        xi = xiini + j * deltaxi
        zg[j] = z0 * np.exp(xi)
        deltaz = zg[j] * (exp_dxi - 1.0)

        # u0(j)
        aux1 = np.log(zg[j] / z0)
        aux2 = A * (zg[j]**2 - z0**2)
        aux3 = B * (zg[j]**3 - z0**3)
        u0[j] = (aux1 + aux2 + aux3) * np.sqrt(Cf0) / kvk

        # g0
        yjj1, yjj2 = rungeg(zg[j], deltaz, g0[j], dg0[j], 0, Cf0, kvk, A, B, z0, 0.0)
        g0[j + 1] = yjj1
        dg0[j + 1] = yjj2

        # g1
        yjj1, yjj2 = rungeg(zg[j], deltaz, g1[j], dg1[j], 1, Cf0, kvk, A, B, z0, 0.0)
        g1[j + 1] = yjj1
        dg1[j + 1] = yjj2

        # g2
        yjj1, yjj2 = rungeg(zg[j], deltaz, g2[j], dg2[j], 2, Cf0, kvk, A, B, z0, 0.0)
        g2[j + 1] = yjj1
        dg2[j + 1] = yjj2

    # last point z=1
    xi = xiini + (Nz - 1) * deltaxi
    zg[Nz - 1] = z0 * np.exp(xi)

    aux1 = np.log(zg[Nz - 1] / z0)
    aux2 = A * (zg[Nz - 1]**2 - z0**2)
    aux3 = B * (zg[Nz - 1]**3 - z0**3)
    u0[Nz - 1] = (aux1 + aux2 + aux3) * np.sqrt(Cf0) / kvk

    q1 = -dg1[-1] / dg0[-1]
    q2 = -dg2[-1] / dg0[-1]

    Int0 = _trapz(g0, zg)
    Int1 = _trapz(g1, zg)
    Int2 = _trapz(g2, zg)

    ha0 = -(Int2 + q2 * Int0) / (Int1 + q1 * Int0)
    q0 = q1 * ha0 + q2

    G0 = q0 * g0 + ha0 * g1 + g2
    dG0 = q0 * dg0 + ha0 * dg1 + dg2

    # --- Second pass: build G1 (overwrite g2,dg2 only, like MATLAB) ---
    g2[0] = 0.0
    dg2[0] = 1.0

    for j in range(0, Nz - 1):
        deltaz = zg[j] * (exp_dxi - 1.0)
        G0aux = 0.5 * (G0[j] + G0[j + 1])
        yjj1, yjj2 = rungeg(zg[j], deltaz, g2[j], dg2[j], 21, Cf0, kvk, A, B, z0, float(G0aux))
        g2[j + 1] = yjj1
        dg2[j + 1] = yjj2

    q2 = -dg2[-1] / dg0[-1]

    Int0 = _trapz(g0, zg)
    Int1 = _trapz(g1, zg)
    Int2 = _trapz(g2, zg)

    ha1 = -(Int2 + q2 * Int0) / (Int1 + q1 * Int0)
    q0 = q1 * ha1 + q2

    G1 = q0 * g0 + ha1 * g1 + g2
    dG1 = q0 * dg0 + ha1 * dg1 + dg2

    # --- k0,k1 integrals ---
    k0 = _trapz(G0 * u0, zg)
    k1 = _trapz(G1 * u0, zg)

    # --- k2,k3 (THIS is where your Python currently differs) ---
    Du0 = np.sqrt(Cf0) * (1.0 / z0 + 2.0 * A * z0 + 3.0 * B * (z0**2)) / kvk
    k2 = dG0[0] / Du0   # MATLAB: dG0(1)/Du0
    k3 = dG1[0] / Du0   # MATLAB: dG1(1)/Du0

    return float(k0), float(k1), float(k2), float(k3), u0, G0, G1, zg, z0
