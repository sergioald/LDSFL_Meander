# ldsfl/modes.py
"""Shared modal root and coefficient machinery for LDSFL flow solvers.

Both the free-boundary and periodic flow-field implementations need the same
quartic-root ordering, modal coefficient precomputation, and resonance flag.
Keeping this machinery in one module avoids subtle drift between the two flow
paths while preserving the MATLAB-compatible numerical ordering.
"""

from __future__ import annotations

import numpy as np

from .vertical import k0123


def _roots_companion_matlab(p: np.ndarray, real_tol: float = 1e-13) -> np.ndarray:
    """MATLAB roots(p) via companion-matrix eigenvalues (real-path when possible)."""
    p = np.asarray(p).ravel()
    if p.size < 2:
        return np.array([], dtype=np.complex128)

    # Trim leading zeros
    i0 = 0
    while i0 < p.size and p[i0] == 0:
        i0 += 1
    p = p[i0:]
    n = p.size - 1
    if n <= 0:
        return np.array([], dtype=np.complex128)

    pc = p.astype(np.complex128, copy=False)
    scale = max(1.0, float(np.max(np.abs(pc.real))))
    is_real_poly = np.max(np.abs(pc.imag)) <= real_tol * scale

    if is_real_poly:
        pr = np.asarray(pc.real, dtype=np.float64)
        A = np.zeros((n, n), dtype=np.float64)
        if n > 1:
            A[1:, :-1] = np.eye(n - 1, dtype=np.float64)
        A[0, :] = -pr[1:] / pr[0]
        return np.linalg.eigvals(A).astype(np.complex128)

    A = np.zeros((n, n), dtype=np.complex128)
    if n > 1:
        A[1:, :-1] = np.eye(n - 1, dtype=np.complex128)
    A[0, :] = -pc[1:] / pc[0]
    return np.linalg.eigvals(A).astype(np.complex128)


def _sort_roots_like_matlab_swaps(r: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """MATLAB swap ordering from Parall_U_free.m, with imag==0 -> abs(imag)<tol."""
    r = np.asarray(r, dtype=np.complex128).copy()
    if r.size != 4:
        raise ValueError("Expected 4 roots for quartic.")

    for j in range(0, 4):
        if (r[j].real > 0.0) and (abs(r[j].imag) < tol):
            r[0], r[j] = r[j], r[0]

    for j in range(1, 4):
        if (r[j].real < 0.0) and (abs(r[j].imag) < tol):
            r[3], r[j] = r[j], r[3]

    return r


def _precompute_modes(
    Cf0: float,
    CT: float,
    CD: float,
    phiT: float,
    phiD: float,
    beta: float,
    rpic: float,
    theta0: float,
    F0: float,
    Mdat: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute Am, lamb1..4, g10..g41 exactly as MATLAB Parall_U_free.m."""

    k0, k1, k2, k3, *_ = k0123(Cf0)

    s1 = 2.0 / (1.0 - CT)
    s2 = CD / (1.0 - CT)
    f1 = 2.0 * phiT / (1.0 - CT)
    f2 = phiD + CD * phiT / (1.0 - CT)

    a1 = beta * Cf0 * s1
    a2 = beta * Cf0 * (s2 - 1.0)
    a3 = beta * Cf0
    a4 = f1
    a5 = f2
    a6 = rpic / (beta * (theta0**0.5))

    b1 = -beta * Cf0
    b2 = 1.0 - (Cf0**0.5) * k2
    b3 = -k0 / (beta * (Cf0**0.5)) - k3 / beta
    b4 = -k1 / (Cf0 * (beta**2))
    b5 = k2 * (theta0**0.5) / (rpic * (Cf0**0.5))
    b6 = k3 * (theta0**0.5) / (beta * Cf0 * rpic)

    h1bar = b2
    h2bar = b3
    h3bar = b4
    d1bar = (F0**2) * h1bar - b5
    d2bar = (F0**2) * h2bar - b6
    d3bar = (F0**2) * h3bar

    alf0 = a2
    alf1 = 1.0 / (F0**2)

    bet2 = a1
    bet3 = 1.0

    gam2 = b1 - a2 * d1bar
    gam3 = -h1bar - a2 * d2bar

    del1 = a5 - 1.0 - (F0**2) * a3 * a6
    del2 = -(F0**2) * a6

    eps3 = a4 - 1.0 - (F0**2) * a3 * a6
    eps4 = del2

    et3 = -del1 * d1bar
    et4 = -del1 * d2bar + (F0**2) * a6 * d1bar
    et5 = -del1 * d3bar + (F0**2) * a6 * d2bar

    Am = np.zeros(Mdat, dtype=np.float64)
    lamb1 = np.zeros(Mdat, dtype=np.complex128)
    lamb2 = np.zeros(Mdat, dtype=np.complex128)
    lamb3 = np.zeros(Mdat, dtype=np.complex128)
    lamb4 = np.zeros(Mdat, dtype=np.complex128)
    g10 = np.zeros(Mdat, dtype=np.complex128)
    g11 = np.zeros(Mdat, dtype=np.complex128)
    g20 = np.zeros(Mdat, dtype=np.complex128)
    g21 = np.zeros(Mdat, dtype=np.complex128)
    g30 = np.zeros(Mdat, dtype=np.complex128)
    g31 = np.zeros(Mdat, dtype=np.complex128)
    g40 = np.zeros(Mdat, dtype=np.complex128)
    g41 = np.zeros(Mdat, dtype=np.complex128)

    SORT_TOL = 1e-10

    for jm in range(1, Mdat + 1):
        M = (2 * (jm - 1) + 1) * np.pi / 2.0
        Am[jm - 1] = ((-1) ** (jm - 1)) * 2.0 / (M**2)

        alf2 = (1.0 - a5) / ((M**2) * (F0**2) * a6)
        bet4 = (1.0 - a4) / ((M**2) * (F0**2) * a6)

        gam4 = -alf2 * d1bar - h2bar - a2 * d3bar
        gam5 = -alf2 * d2bar - h3bar

        del0 = -(M**2) * a6

        Delta = del2 * alf1 - del1 * alf2
        Delta0 = (del2 * alf0 - del0 * alf2) / Delta
        Delta1 = del2 * Delta0 - del1
        Delta2 = Delta1 * Delta0 + del0

        T1 = -del2 * bet2 / Delta
        T2 = -(del2 * bet3 - alf2 * eps3) / Delta
        T3 = -(del2 * bet4 - alf2 * eps4) / Delta

        Tc1 = del2 * gam2 / Delta
        Tc2 = (del2 * gam3 - alf2 * et3) / Delta
        Tc3 = (del2 * gam4 - alf2 * et4) / Delta
        Tc4 = (del2 * gam5 - alf2 * et5) / Delta

        csi1 = -Delta * Delta1 * T1
        csi2 = Delta * (-Delta1 * T2 + del2 * T1 + eps3)
        csi3 = Delta * (-Delta1 * T3 + del2 * T2 + eps4)
        csi4 = Delta * del2 * T3

        mu1 = Delta * Delta1 * Tc1
        mu2 = Delta * (Delta1 * Tc2 - del2 * Tc1 + et3)
        mu3 = Delta * (Delta1 * Tc3 - del2 * Tc2 + et4)
        mu4 = Delta * (Delta1 * Tc4 - del2 * Tc3 + et5)

        sigma0 = (Delta0 * csi1 + Delta * Delta2 * T1) / csi4
        sigma1 = (csi1 + Delta0 * csi2 + Delta * Delta2 * T2) / csi4
        sigma2 = (csi2 + Delta0 * csi3 + Delta * Delta2 * T3) / csi4
        sigma3 = (csi3 + Delta0 * csi4) / csi4
        sigma4 = 1.0

        rho0 = (Delta0 * mu1 - Delta * Delta2 * Tc1) / csi4
        rho1 = (mu1 + Delta0 * mu2 - Delta * Delta2 * Tc2) / csi4
        rho2 = (mu2 + Delta0 * mu3 - Delta * Delta2 * Tc3) / csi4
        rho3 = (mu3 + Delta0 * mu4 - Delta * Delta2 * Tc4) / csi4
        rho4 = mu4 / csi4

        sigm = np.array([sigma4, sigma3, sigma2, sigma1, sigma0], dtype=np.float64)

        roots4 = _roots_companion_matlab(sigm)
        roots4 = _sort_roots_like_matlab_swaps(roots4, tol=SORT_TOL)
        lam1, lam2, lam3, lam4 = roots4[0], roots4[1], roots4[2], roots4[3]
        lamb1[jm - 1], lamb2[jm - 1], lamb3[jm - 1], lamb4[jm - 1] = lam1, lam2, lam3, lam4

        waux1 = lam3 * lam4 * (lam4 - lam3)
        waux2 = -lam2 * lam4 * (lam4 - lam2)
        waux3 = lam2 * lam3 * (lam3 - lam2)
        W1det = -(waux1 + waux2 + waux3)
        wdet = -lam2 * lam3 * lam4 * W1det

        waux1 = lam3 * lam4 * (lam4 - lam3)
        waux2 = -lam1 * lam4 * (lam4 - lam1)
        waux3 = lam1 * lam3 * (lam3 - lam1)
        W2det = (waux1 + waux2 + waux3)
        wdet = wdet - lam1 * lam3 * lam4 * W2det

        waux1 = lam2 * lam4 * (lam4 - lam2)
        waux2 = -lam1 * lam4 * (lam4 - lam1)
        waux3 = lam1 * lam2 * (lam2 - lam1)
        W3det = -(waux1 + waux2 + waux3)
        wdet = wdet - lam1 * lam2 * lam4 * W3det

        waux1 = lam2 * lam3 * (lam3 - lam2)
        waux2 = -lam1 * lam3 * (lam3 - lam1)
        waux3 = lam1 * lam2 * (lam2 - lam1)
        W4det = (waux1 + waux2 + waux3)
        wdet = wdet - lam1 * lam2 * lam3 * W4det

        wj_1 = W1det / wdet
        wj_2 = W2det / wdet
        wj_3 = W3det / wdet
        wj_4 = W4det / wdet

        g10[jm - 1] = wj_1 * (rho0 + rho1 * lam1 + rho2 * lam1**2 + rho3 * lam1**3 + rho4 * lam1**4)
        g20[jm - 1] = wj_2 * (rho0 + rho1 * lam2 + rho2 * lam2**2 + rho3 * lam2**3 + rho4 * lam2**4)
        g30[jm - 1] = wj_3 * (rho0 + rho1 * lam3 + rho2 * lam3**2 + rho3 * lam3**3 + rho4 * lam3**4)
        g40[jm - 1] = wj_4 * (rho0 + rho1 * lam4 + rho2 * lam4**2 + rho3 * lam4**3 + rho4 * lam4**4)

        g11[jm - 1] = wj_1 * (rho1 + rho2 * lam1 + rho3 * lam1**2 + rho4 * lam1**3)
        g21[jm - 1] = wj_2 * (rho1 + rho2 * lam2 + rho3 * lam2**2 + rho4 * lam2**3)
        g31[jm - 1] = wj_3 * (rho1 + rho2 * lam3 + rho3 * lam3**2 + rho4 * lam3**3)
        g41[jm - 1] = wj_4 * (rho1 + rho2 * lam4 + rho3 * lam4**2 + rho4 * lam4**3)

    return Am, lamb1, lamb2, lamb3, lamb4, g10, g20, g30, g40, g11, g21, g31, g41


def _compute_flag(lamb2: np.ndarray) -> int:
    """Resonance indicator: +1 sub-resonant, -1 super-resonant.

    The sign of ``Re(lambda2)`` for the fundamental lateral mode distinguishes
    the two regimes. It is negative below the resonant aspect ratio and
    positive above it.
    """
    if lamb2.size == 0:
        return 1
    return 1 if np.real(lamb2[0]) < 0 else -1
