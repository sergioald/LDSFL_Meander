# ldsfl/flowfield_periodic.py
#
# Periodic-boundary version of the LDSFL-Meander flow-field (near-bank excess velocity U)
#
# This file mirrors the structure of `flowfield_new.py` (free BCs + semiana*_cached),
# but computes U for *periodic* boundary conditions using a literal port of the
# Fortran routine `dUZSBC1` (Zolezzi & Seminara, 2001), provided as
# `dUZS_BCperiodic (2).f`.
#
# Key idea:
#   - Free BCs: set the growing homogeneous constants to zero.
#   - Periodic BCs: add homogeneous terms and solve 4 constants (c1..c4)
#     so U (and its first 3 derivatives) are periodic across s=0..L.
#
# IMPORTANT ASSUMPTION (matches the Fortran algebra):
#   The extra coefficient vectors passed in the Fortran (ph*, dl*, cs*) are
#   the derivative ratios for an eigenmode exp(lambda*s):
#       ph = lambda,   dl = lambda^2,   cs = lambda^3
#   which enforces periodicity of U, U_s, U_ss, U_sss.
#
# If you later find in your original Fortran build that (ph,dl,cs) correspond
# to *other* state variables (e.g., V, D, H) instead of derivatives of U,
# then only the small block that defines ph/dl/cs needs to be replaced.
# Everything else is a direct dUZSBC1 port.

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .vertical import k0123

# --------------------------
# MATLAB-compatible roots()
# --------------------------

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


# --------------------------
# Mode precomputation
# --------------------------

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
    the two regimes: it is negative below the resonant aspect ratio and
    positive above it. This transition flips bend migration from
    downstream- to upstream-dominated in the Zolezzi-Seminara model family.

    The previous implementation indexed mode 6 (``idx = 5``) rather than mode
    1. Mode 6 remains strongly negative across realistic aspect ratios, so the
    reported flag was effectively constant. Mode 1 crosses zero at ``beta_R``.
    """
    if lamb2.size == 0:
        return 1
    return 1 if np.real(lamb2[0]) < 0 else -1

# --------------------------
# Periodic BC core (dUZSBC1)
# --------------------------

TOLL_DEFAULT: float = 1e-4


def _det3(a1, a2, a3, b1, b2, b3, c1, c2, c3):
    """Determinant of a 3x3 complex matrix (Fortran det3.f)."""
    return a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)


def _trapz_uniform_real(dzz: float, ff: np.ndarray) -> float:
    """Composite trapezoid with constant step (Fortran CavSimpds)."""
    ff = np.asarray(ff, dtype=np.float64)
    n = int(ff.size)
    if n <= 1:
        return 0.0
    s = float(np.sum(ff[1:-1])) if n > 2 else 0.0
    return float(dzz) * (s + 0.5 * (float(ff[0]) + float(ff[-1])))


def _trapz_uniform_complex(dzz: float, ff: np.ndarray) -> complex:
    """Composite trapezoid of real & imag parts separately (Fortran CavSimds2)."""
    ff = np.asarray(ff, dtype=np.complex128)
    n = int(ff.size)
    if n <= 1:
        return 0.0 + 0.0j
    sr = float(np.sum(ff.real[1:-1])) if n > 2 else 0.0
    si = float(np.sum(ff.imag[1:-1])) if n > 2 else 0.0
    sr = float(dzz) * (sr + 0.5 * (float(ff.real[0]) + float(ff.real[-1])))
    si = float(dzz) * (si + 0.5 * (float(ff.imag[0]) + float(ff.imag[-1])))
    return complex(sr, si)


def _first_below(arr: np.ndarray, toll: float) -> int:
    """Return smallest index i such that arr[i] < toll; if none, return len(arr)."""
    for i in range(int(arr.size)):
        if float(arr[i]) < toll:
            return i
    return int(arr.size)



def _duzs_periodic_bc_one_mode(
    jm: int,
    *,
    c: np.ndarray,
    deltas: float,
    csi: np.ndarray,
    lamb1: np.ndarray,
    lamb2: np.ndarray,
    lamb3: np.ndarray,
    lamb4: np.ndarray,
    Av_pos: np.ndarray,
    g10v: np.ndarray,
    g20v: np.ndarray,
    g30v: np.ndarray,
    g40v: np.ndarray,
    g11v: np.ndarray,
    g21v: np.ndarray,
    g31v: np.ndarray,
    g41v: np.ndarray,
    toll: float,
) -> np.ndarray:
    """Compute periodic-BC deltaU contribution for a single Fourier mode jm (0-based).

    This is the body of the Fortran loop in dUZSBC1, factored out so we can
    parallelize across modes when `paral=1`.
    """
    c = np.asarray(c, dtype=np.float64)
    csi = np.asarray(csi, dtype=np.float64)
    N = int(c.size)

    # domain length
    Elle = float(deltas) * float(N - 1)

    deltaU = np.zeros(N, dtype=np.float64)

    # workspace arrays (mode-local; thread-safe)
    elm1 = np.empty(N, dtype=np.float64)
    elm4 = np.empty(N, dtype=np.float64)
    Relm2 = np.empty(N, dtype=np.float64)
    Relm3 = np.empty(N, dtype=np.float64)
    elm2 = np.empty(N, dtype=np.complex128)
    elm3 = np.empty(N, dtype=np.complex128)

    A = float(Av_pos[jm])

    l1 = float(np.real(lamb1[jm]))
    l2 = complex(lamb2[jm])
    l3 = complex(lamb3[jm])
    l4 = float(np.real(lamb4[jm]))

    g10 = complex(g10v[jm])
    g20 = complex(g20v[jm])
    g30 = complex(g30v[jm])
    g40 = complex(g40v[jm])

    g11 = complex(g11v[jm])
    g21 = complex(g21v[jm])
    g31 = complex(g31v[jm])
    g41 = complex(g41v[jm])

    # === ph/dl/cs (derivative ratios) ===
    ph1, ph2, ph3, ph4 = complex(l1), l2, l3, complex(l4)
    dl1, dl2, dl3, dl4 = ph1**2, ph2**2, ph3**2, ph4**2
    cs1, cs2, cs3, cs4 = ph1**3, ph2**3, ph3**3, ph4**3

    sub_resonant = (l2.real < 0.0)

    if sub_resonant:
        # exponentials for offsets x=csi
        for i in range(N):
            x = float(csi[i])
            elm1[i] = float(np.exp(-l1 * x))
            elm2[i] = np.exp(l2 * x)
            elm3[i] = np.exp(l3 * x)
            elm4[i] = float(np.exp(l4 * x))
            Relm2[i] = float(np.exp(l2.real * x))
            Relm3[i] = float(np.exp(l3.real * x))

        # I1L (forward, real)
        if elm1[1] < toll:
            sum1 = c[0] * (1.0 - np.exp(-l1 * deltas)) / l1
        else:
            nn = min(_first_below(elm1, toll) + 1, N)
            sum1 = _trapz_uniform_real(deltas, c[:nn] * elm1[:nn])

        # I2L (reverse)
        if Relm2[1] < toll:
            sum2 = c[-1] * (-1.0 + np.exp(l2 * deltas)) / l2
        else:
            nn = min(_first_below(Relm2, toll) + 1, N)
            seg = c[::-1][:nn]
            sum2 = _trapz_uniform_complex(deltas, seg * elm2[:nn])

        # I3L (reverse)
        if Relm3[1] < toll:
            sum3 = c[-1] * (-1.0 + np.exp(l3 * deltas)) / l3
        else:
            nn = min(_first_below(Relm3, toll) + 1, N)
            seg = c[::-1][:nn]
            sum3 = _trapz_uniform_complex(deltas, seg * elm3[:nn])

        # I4L (reverse, real)
        if elm4[1] < toll:
            sum4 = c[-1] * (-1.0 + np.exp(l4 * deltas)) / l4
            sum4 = complex(sum4, 0.0)
        else:
            nn = min(_first_below(elm4, toll) + 1, N)
            seg = c[::-1][:nn]
            sum4r = _trapz_uniform_real(deltas, seg * elm4[:nn])
            sum4 = complex(sum4r, 0.0)

        # B terms (Fortran)
        B1 = A * (g20 * sum2 + g30 * sum3 + g40 * sum4 + g10 * sum1)
        B2 = A * (g20 * ph2 * sum2 + g30 * ph3 * sum3 + g40 * ph4 * sum4 + g10 * ph1 * sum1)
        B3 = A * (g20 * dl2 * sum2 + g30 * dl3 * sum3 + g40 * dl4 * sum4 + g10 * dl1 * sum1)
        B4 = A * (g20 * cs2 * sum2 + g30 * cs3 * sum3 + g40 * cs4 * sum4 + g10 * cs1 * sum1)

        # Solve (c1..c4) via Cramer's rule (exactly as Fortran)
        A1Det = _det3(ph2, ph3, ph4, dl2, dl3, dl4, cs2, cs3, cs4)
        A2Det = _det3(-ph1, ph3, ph4, -dl1, dl3, dl4, -cs1, cs3, cs4)
        A3Det = _det3(-ph1, ph2, ph4, -dl1, dl2, dl4, -cs1, cs2, cs4)
        A4Det = _det3(-ph1, ph2, ph3, -dl1, dl2, dl3, -cs1, cs2, cs3)
        Adet = -A1Det - A2Det + A3Det - A4Det

        A2Det = _det3(B2, ph3, ph4, B3, dl3, dl4, B4, cs3, cs4)
        A3Det = _det3(B2, ph2, ph4, B3, dl2, dl4, B4, cs2, cs4)
        A4Det = _det3(B2, ph2, ph3, B3, dl2, dl3, B4, cs2, cs3)
        c1 = (B1 * A1Det - A2Det + A3Det - A4Det) / Adet

        A1Det = _det3(B2, ph3, ph4, B3, dl3, dl4, B4, cs3, cs4)
        A2Det = _det3(-ph1, ph3, ph4, -dl1, dl3, dl4, -cs1, cs3, cs4)
        A3Det = _det3(-ph1, B2, ph4, -dl1, B3, dl4, -cs1, B4, cs4)
        A4Det = _det3(-ph1, B2, ph3, -dl1, B3, dl3, -cs1, B4, cs3)
        c2 = (-A1Det - B1 * A2Det + A3Det - A4Det) / Adet

        A1Det = _det3(ph2, B2, ph4, dl2, B3, dl4, cs2, B4, cs4)
        A2Det = _det3(-ph1, B2, ph4, -dl1, B3, dl4, -cs1, B4, cs4)
        A3Det = _det3(-ph1, ph2, ph4, -dl1, dl2, dl4, -cs1, cs2, cs4)
        A4Det = _det3(-ph1, ph2, B2, -dl1, dl2, B3, -cs1, cs2, B4)
        c3 = (-A1Det - A2Det + B1 * A3Det - A4Det) / Adet

        A1Det = _det3(ph2, ph3, B2, dl2, dl3, B3, cs2, cs3, B4)
        A2Det = _det3(-ph1, ph3, B2, -dl1, dl3, B3, -cs1, cs3, B4)
        A3Det = _det3(-ph1, ph2, B2, -dl1, dl2, B3, -cs1, cs2, B4)
        A4Det = _det3(-ph1, ph2, ph3, -dl1, dl2, dl3, -cs1, cs2, cs3)
        c4 = (-A1Det - A2Det + A3Det - B1 * A4Det) / Adet

        c1 = c1 / (1.0 - elm1[-1])
        c2 = c2 / (1.0 - elm2[-1])
        c3 = c3 / (1.0 - elm3[-1])
        c4 = c4 / (1.0 - elm4[-1])

        cut1 = _first_below(elm1, toll) + 1
        cut2 = _first_below(Relm2, toll) + 1
        cut3 = _first_below(Relm3, toll) + 1
        cut4 = _first_below(elm4, toll) + 1

        um1 = np.zeros(N, dtype=np.complex128)
        um2 = np.zeros(N, dtype=np.complex128)

        # um1(js): downstream conv with lambda1
        for js in range(N - 1):
            if elm1[1] < toll:
                sum1 = c[js] * (1.0 - np.exp(-l1 * deltas)) / l1
            else:
                nn = min(N - js, cut1)
                sum1 = _trapz_uniform_real(deltas, c[js: js + nn] * elm1[:nn])
            um1[js] = -A * g10 * sum1

        # um2(js): upstream conv with lambdas 2,3,4
        for js in range(1, N):
            # lambda2
            if Relm2[1] < toll:
                sum2 = c[js] * (-1.0 + np.exp(l2 * deltas)) / l2
            else:
                nn = min(js + 1, cut2)
                seg = c[js - nn + 1: js + 1][::-1]
                sum2 = _trapz_uniform_complex(deltas, seg * elm2[:nn])

            # lambda3
            if Relm3[1] < toll:
                sum3 = c[js] * (-1.0 + np.exp(l3 * deltas)) / l3
            else:
                nn3 = min(js + 1, cut3)
                seg3 = c[js - nn3 + 1: js + 1][::-1]
                sum3 = _trapz_uniform_complex(deltas, seg3 * elm3[:nn3])

            # lambda4
            if elm4[1] < toll:
                sum4 = c[js] * (-1.0 + np.exp(l4 * deltas)) / l4
                sum4 = complex(sum4, 0.0)
            else:
                nn4 = min(js + 1, cut4)
                seg4 = c[js - nn4 + 1: js + 1][::-1]
                sum4r = _trapz_uniform_real(deltas, seg4 * elm4[:nn4])
                sum4 = complex(sum4r, 0.0)

            um2[js] = A * (g20 * sum2 + g30 * sum3 + g40 * sum4)

        # add local + homogeneous
        sign = ((-1.0) ** jm)
        for js in range(N):
            um3 = A * (c[js] * (g11 + g21 + g31 + g41))
            um4 = c1 * np.exp(l1 * (csi[js] - Elle))
            um5 = c2 * np.exp(l2 * csi[js]) + c3 * np.exp(l3 * csi[js]) + c4 * np.exp(l4 * csi[js])
            um = um1[js] + um2[js] + um3 + um4 + um5
            deltaU[js] += 2.0 * float(um.real) * sign

    else:
        # --- super-resonant branch ---
        for i in range(N):
            x = float(csi[i])
            elm1[i] = float(np.exp(-l1 * x))
            elm2[i] = np.exp(-l2 * x)
            elm3[i] = np.exp(-l3 * x)
            elm4[i] = float(np.exp(l4 * x))
            Relm2[i] = float(np.exp((-l2.real) * x))
            Relm3[i] = float(np.exp((-l3.real) * x))

        # I1L
        if elm1[1] < toll:
            sum1 = c[0] * (1.0 - np.exp(-l1 * deltas)) / l1
        else:
            nn = min(_first_below(elm1, toll) + 1, N)
            sum1 = _trapz_uniform_real(deltas, c[:nn] * elm1[:nn])

        # I2L (forward)
        if Relm2[1] < toll:
            sum2 = c[0] * (1.0 - np.exp(-l2 * deltas)) / l2
        else:
            nn = min(_first_below(Relm2, toll) + 1, N)
            sum2 = _trapz_uniform_complex(deltas, c[:nn] * elm2[:nn])

        # I3L (forward)
        if Relm3[1] < toll:
            sum3 = c[0] * (1.0 - np.exp(-l3 * deltas)) / l3
        else:
            nn = min(_first_below(Relm3, toll) + 1, N)
            sum3 = _trapz_uniform_complex(deltas, c[:nn] * elm3[:nn])

        # I4L (reverse)
        if elm4[1] < toll:
            sum4 = c[-1] * (-1.0 + np.exp(l4 * deltas)) / l4
            sum4 = complex(sum4, 0.0)
        else:
            nn = min(_first_below(elm4, toll) + 1, N)
            seg = c[::-1][:nn]
            sum4r = _trapz_uniform_real(deltas, seg * elm4[:nn])
            sum4 = complex(sum4r, 0.0)

        B1 = A * (g20 * sum2 + g30 * sum3 + g40 * sum4 + g10 * sum1)
        B2 = A * (g20 * ph2 * sum2 + g30 * ph3 * sum3 + g40 * ph4 * sum4 + g10 * ph1 * sum1)
        B3 = A * (g20 * dl2 * sum2 + g30 * dl3 * sum3 + g40 * dl4 * sum4 + g10 * dl1 * sum1)
        B4 = A * (g20 * cs2 * sum2 + g30 * cs3 * sum3 + g40 * cs4 * sum4 + g10 * cs1 * sum1)

        A1Det = _det3(ph2, ph3, -ph4, dl2, dl3, -dl4, cs2, cs3, -cs4)
        A2Det = _det3(ph1, ph3, -ph4, dl1, dl3, -dl4, cs1, cs3, -cs4)
        A3Det = _det3(ph1, ph2, -ph4, dl1, dl2, -dl4, cs1, cs2, -cs4)
        A4Det = _det3(ph1, ph2, ph3, dl1, dl2, dl3, cs1, cs2, cs3)
        Adet = A1Det - A2Det + A3Det + A4Det

        A2Det = _det3(-B2, ph3, -ph4, -B3, dl3, -dl4, -B4, cs3, -cs4)
        A3Det = _det3(-B2, ph2, -ph4, -B3, dl2, -dl4, -B4, cs2, -cs4)
        A4Det = _det3(-B2, ph2, ph3, -B3, dl2, dl3, -B4, cs2, cs3)
        c1 = (-B1 * A1Det - A2Det + A3Det + A4Det) / Adet

        A1Det = _det3(-B2, ph3, -ph4, -B3, dl3, -dl4, -B4, cs3, -cs4)
        A2Det = _det3(ph1, ph3, -ph4, dl1, dl3, -dl4, cs1, cs3, -cs4)
        A3Det = _det3(ph1, -B2, -ph4, dl1, -B3, -dl4, cs1, -B4, -cs4)
        A4Det = _det3(ph1, -B2, ph3, dl1, -B3, dl3, cs1, -B4, cs3)
        c2 = (A1Det + B1 * A2Det + A3Det + A4Det) / Adet

        A1Det = _det3(ph2, -B2, -ph4, dl2, -B3, -dl4, cs2, -B4, -cs4)
        A2Det = _det3(ph1, -B2, -ph4, dl1, -B3, -dl4, cs1, -B4, -cs4)
        A3Det = _det3(ph1, ph2, -ph4, dl1, dl2, -dl4, cs1, cs2, -cs4)
        A4Det = _det3(ph1, ph2, -B2, dl1, dl2, -B3, cs1, cs2, -B4)
        c3 = (A1Det - A2Det - B1 * A3Det + A4Det) / Adet

        A1Det = _det3(ph2, ph3, -B2, dl2, dl3, -B3, cs2, cs3, -B4)
        A2Det = _det3(ph1, ph3, -B2, dl1, dl3, -B3, cs1, cs3, -B4)
        A3Det = _det3(ph1, ph2, -B2, dl1, dl2, -B3, cs1, cs2, -B4)
        A4Det = _det3(ph1, ph2, ph3, dl1, dl2, dl3, cs1, cs2, cs3)
        c4 = (A1Det - A2Det + A3Det + B1 * A4Det) / Adet

        c1 = c1 / (1.0 - elm1[-1])
        c2 = c2 / (1.0 - elm2[-1])
        c3 = c3 / (1.0 - elm3[-1])
        c4 = c4 / (1.0 - elm4[-1])

        cut1 = _first_below(elm1, toll) + 1
        cut2 = _first_below(Relm2, toll) + 1
        cut3 = _first_below(Relm3, toll) + 1
        cut4 = _first_below(elm4, toll) + 1

        um1 = np.zeros(N, dtype=np.complex128)
        um2 = np.zeros(N, dtype=np.complex128)

        # um1(js): downstream conv with lambdas 1,2,3
        for js in range(N - 1):
            nn = min(N - js, cut1)
            sum1 = _trapz_uniform_real(deltas, c[js: js + nn] * elm1[:nn])

            nn2 = min(N - js, cut2)
            sum2 = _trapz_uniform_complex(deltas, c[js: js + nn2] * elm2[:nn2])

            nn3 = min(N - js, cut3)
            sum3 = _trapz_uniform_complex(deltas, c[js: js + nn3] * elm3[:nn3])

            um1[js] = -A * (g10 * sum1 + g20 * sum2 + g30 * sum3)

        # um2(js): upstream conv with lambda4
        for js in range(1, N):
            if elm4[1] < toll:
                sum4 = c[js] * (-1.0 + np.exp(l4 * deltas)) / l4
                sum4 = complex(sum4, 0.0)
            else:
                nn4 = min(js + 1, cut4)
                seg4 = c[js - nn4 + 1: js + 1][::-1]
                sum4r = _trapz_uniform_real(deltas, seg4 * elm4[:nn4])
                sum4 = complex(sum4r, 0.0)
            um2[js] = A * g40 * sum4

        sign = ((-1.0) ** jm)
        for js in range(N):
            um3 = A * (c[js] * (g11 + g21 + g31 + g41))
            um4 = c4 * np.exp(l4 * csi[js])
            um5 = (
                c1 * np.exp(l1 * (csi[js] - Elle))
                + c2 * np.exp(l2 * (csi[js] - Elle))
                + c3 * np.exp(l3 * (csi[js] - Elle))
            )
            um = um1[js] + um2[js] + um3 + um4 + um5
            deltaU[js] += 2.0 * float(um.real) * sign

    return deltaU


def _duzs_periodic_bc_modes(
    *,
    c: np.ndarray,
    deltas: float,
    csi: np.ndarray,
    lamb1: np.ndarray,
    lamb2: np.ndarray,
    lamb3: np.ndarray,
    lamb4: np.ndarray,
    Av_pos: np.ndarray,
    g10v: np.ndarray,
    g20v: np.ndarray,
    g30v: np.ndarray,
    g40v: np.ndarray,
    g11v: np.ndarray,
    g21v: np.ndarray,
    g31v: np.ndarray,
    g41v: np.ndarray,
    toll: float,
    paral: int,
    n_workers: int | None,
) -> np.ndarray:
    """Compute periodic deltaU for all modes, optionally parallelizing over jm."""
    Mdat = int(Av_pos.size)
    c = np.asarray(c, dtype=np.float64)
    N = int(c.size)

    if int(paral) == 0:
        # Keep the original serial implementation (slightly fewer allocations)
        return _duzs_periodic_bc_all_modes(
            c=c,
            deltas=deltas,
            csi=csi,
            lamb1=lamb1,
            lamb2=lamb2,
            lamb3=lamb3,
            lamb4=lamb4,
            Av_pos=Av_pos,
            g10v=g10v,
            g20v=g20v,
            g30v=g30v,
            g40v=g40v,
            g11v=g11v,
            g21v=g21v,
            g31v=g31v,
            g41v=g41v,
            toll=toll,
        )

    # Parallel over modes (MATLAB paral==1 / parfor analogue).
    # We keep deterministic summation by collecting results and adding in jm order.
    max_workers = int(n_workers) if n_workers is not None else (os.cpu_count() or 1)
    results: list[np.ndarray | None] = [None] * Mdat

    def _task(jm: int) -> np.ndarray:
        return _duzs_periodic_bc_one_mode(
            jm,
            c=c,
            deltas=deltas,
            csi=csi,
            lamb1=lamb1,
            lamb2=lamb2,
            lamb3=lamb3,
            lamb4=lamb4,
            Av_pos=Av_pos,
            g10v=g10v,
            g20v=g20v,
            g30v=g30v,
            g40v=g40v,
            g11v=g11v,
            g21v=g21v,
            g31v=g31v,
            g41v=g41v,
            toll=toll,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_task, jm): jm for jm in range(Mdat)}
        for fut, jm in futs.items():
            results[jm] = fut.result()

    deltaU = np.zeros(N, dtype=np.float64)
    for jm in range(Mdat):
        deltaU += results[jm]  # type: ignore[arg-type]

    return deltaU

def _duzs_periodic_bc_all_modes(
    *,
    c: np.ndarray,
    deltas: float,
    csi: np.ndarray,
    lamb1: np.ndarray,
    lamb2: np.ndarray,
    lamb3: np.ndarray,
    lamb4: np.ndarray,
    Av_pos: np.ndarray,
    g10v: np.ndarray,
    g20v: np.ndarray,
    g30v: np.ndarray,
    g40v: np.ndarray,
    g11v: np.ndarray,
    g21v: np.ndarray,
    g31v: np.ndarray,
    g41v: np.ndarray,
    toll: float,
) -> np.ndarray:
    """Literal port of dUZSBC1 but with ph=lambda, dl=lambda^2, cs=lambda^3."""

    c = np.asarray(c, dtype=np.float64)
    csi = np.asarray(csi, dtype=np.float64)
    N = int(c.size)
    Mdat = int(Av_pos.size)

    # domain length
    Elle = float(deltas) * float(N - 1)

    deltaU = np.zeros(N, dtype=np.float64)

    # workspace arrays (reused)
    elm1 = np.empty(N, dtype=np.float64)
    elm4 = np.empty(N, dtype=np.float64)
    Relm2 = np.empty(N, dtype=np.float64)
    Relm3 = np.empty(N, dtype=np.float64)
    elm2 = np.empty(N, dtype=np.complex128)
    elm3 = np.empty(N, dtype=np.complex128)

    for jm in range(Mdat):
        A = float(Av_pos[jm])

        l1 = float(np.real(lamb1[jm]))
        l2 = complex(lamb2[jm])
        l3 = complex(lamb3[jm])
        l4 = float(np.real(lamb4[jm]))

        g10 = complex(g10v[jm])
        g20 = complex(g20v[jm])
        g30 = complex(g30v[jm])
        g40 = complex(g40v[jm])

        g11 = complex(g11v[jm])
        g21 = complex(g21v[jm])
        g31 = complex(g31v[jm])
        g41 = complex(g41v[jm])

        # === ph/dl/cs (derivative ratios) ===
        ph1, ph2, ph3, ph4 = complex(l1), l2, l3, complex(l4)
        dl1, dl2, dl3, dl4 = ph1**2, ph2**2, ph3**2, ph4**2
        cs1, cs2, cs3, cs4 = ph1**3, ph2**3, ph3**3, ph4**3

        sub_resonant = (l2.real < 0.0)

        if sub_resonant:
            # exponentials for offsets x=csi
            for i in range(N):
                x = float(csi[i])
                elm1[i] = float(np.exp(-l1 * x))
                elm2[i] = np.exp(l2 * x)
                elm3[i] = np.exp(l3 * x)
                elm4[i] = float(np.exp(l4 * x))
                Relm2[i] = float(np.exp(l2.real * x))
                Relm3[i] = float(np.exp(l3.real * x))

            # I1L (forward, real)
            if elm1[1] < toll:
                sum1 = c[0] * (1.0 - np.exp(-l1 * deltas)) / l1
            else:
                nn = min(_first_below(elm1, toll) + 1, N)
                sum1 = _trapz_uniform_real(deltas, c[:nn] * elm1[:nn])

            # I2L (reverse)
            if Relm2[1] < toll:
                sum2 = c[-1] * (-1.0 + np.exp(l2 * deltas)) / l2
            else:
                nn = min(_first_below(Relm2, toll) + 1, N)
                seg = c[::-1][:nn]
                sum2 = _trapz_uniform_complex(deltas, seg * elm2[:nn])

            # I3L (reverse)
            if Relm3[1] < toll:
                sum3 = c[-1] * (-1.0 + np.exp(l3 * deltas)) / l3
            else:
                nn = min(_first_below(Relm3, toll) + 1, N)
                seg = c[::-1][:nn]
                sum3 = _trapz_uniform_complex(deltas, seg * elm3[:nn])

            # I4L (reverse, real)
            if elm4[1] < toll:
                sum4 = c[-1] * (-1.0 + np.exp(l4 * deltas)) / l4
                sum4 = complex(sum4, 0.0)
            else:
                nn = min(_first_below(elm4, toll) + 1, N)
                seg = c[::-1][:nn]
                sum4r = _trapz_uniform_real(deltas, seg * elm4[:nn])
                sum4 = complex(sum4r, 0.0)

            # B terms (Fortran)
            B1 = A * (g20 * sum2 + g30 * sum3 + g40 * sum4 + g10 * sum1)
            B2 = A * (g20 * ph2 * sum2 + g30 * ph3 * sum3 + g40 * ph4 * sum4 + g10 * ph1 * sum1)
            B3 = A * (g20 * dl2 * sum2 + g30 * dl3 * sum3 + g40 * dl4 * sum4 + g10 * dl1 * sum1)
            B4 = A * (g20 * cs2 * sum2 + g30 * cs3 * sum3 + g40 * cs4 * sum4 + g10 * cs1 * sum1)

            # Solve (c1..c4) via Cramer's rule (exactly as Fortran)
            A1Det = _det3(ph2, ph3, ph4, dl2, dl3, dl4, cs2, cs3, cs4)
            A2Det = _det3(-ph1, ph3, ph4, -dl1, dl3, dl4, -cs1, cs3, cs4)
            A3Det = _det3(-ph1, ph2, ph4, -dl1, dl2, dl4, -cs1, cs2, cs4)
            A4Det = _det3(-ph1, ph2, ph3, -dl1, dl2, dl3, -cs1, cs2, cs3)
            Adet = -A1Det - A2Det + A3Det - A4Det

            A2Det = _det3(B2, ph3, ph4, B3, dl3, dl4, B4, cs3, cs4)
            A3Det = _det3(B2, ph2, ph4, B3, dl2, dl4, B4, cs2, cs4)
            A4Det = _det3(B2, ph2, ph3, B3, dl2, dl3, B4, cs2, cs3)
            c1 = (B1 * A1Det - A2Det + A3Det - A4Det) / Adet

            A1Det = _det3(B2, ph3, ph4, B3, dl3, dl4, B4, cs3, cs4)
            A2Det = _det3(-ph1, ph3, ph4, -dl1, dl3, dl4, -cs1, cs3, cs4)
            A3Det = _det3(-ph1, B2, ph4, -dl1, B3, dl4, -cs1, B4, cs4)
            A4Det = _det3(-ph1, B2, ph3, -dl1, B3, dl3, -cs1, B4, cs3)
            c2 = (-A1Det - B1 * A2Det + A3Det - A4Det) / Adet

            A1Det = _det3(ph2, B2, ph4, dl2, B3, dl4, cs2, B4, cs4)
            A2Det = _det3(-ph1, B2, ph4, -dl1, B3, dl4, -cs1, B4, cs4)
            A3Det = _det3(-ph1, ph2, ph4, -dl1, dl2, dl4, -cs1, cs2, cs4)
            A4Det = _det3(-ph1, ph2, B2, -dl1, dl2, B3, -cs1, cs2, B4)
            c3 = (-A1Det - A2Det + B1 * A3Det - A4Det) / Adet

            A1Det = _det3(ph2, ph3, B2, dl2, dl3, B3, cs2, cs3, B4)
            A2Det = _det3(-ph1, ph3, B2, -dl1, dl3, B3, -cs1, cs3, B4)
            A3Det = _det3(-ph1, ph2, B2, -dl1, dl2, B3, -cs1, cs2, B4)
            A4Det = _det3(-ph1, ph2, ph3, -dl1, dl2, dl3, -cs1, cs2, cs3)
            c4 = (-A1Det - A2Det + A3Det - B1 * A4Det) / Adet

            c1 = c1 / (1.0 - elm1[-1])
            c2 = c2 / (1.0 - elm2[-1])
            c3 = c3 / (1.0 - elm3[-1])
            c4 = c4 / (1.0 - elm4[-1])

            cut1 = _first_below(elm1, toll) + 1
            cut2 = _first_below(Relm2, toll) + 1
            cut3 = _first_below(Relm3, toll) + 1
            cut4 = _first_below(elm4, toll) + 1

            um1 = np.zeros(N, dtype=np.complex128)
            um2 = np.zeros(N, dtype=np.complex128)

            # um1(js): downstream conv with lambda1
            for js in range(N - 1):
                if elm1[1] < toll:
                    sum1 = c[js] * (1.0 - np.exp(-l1 * deltas)) / l1
                else:
                    nn = min(N - js, cut1)
                    sum1 = _trapz_uniform_real(deltas, c[js: js + nn] * elm1[:nn])
                um1[js] = -A * g10 * sum1

            # um2(js): upstream conv with lambdas 2,3,4
            for js in range(1, N):
                # lambda2
                if Relm2[1] < toll:
                    sum2 = c[js] * (-1.0 + np.exp(l2 * deltas)) / l2
                else:
                    nn = min(js + 1, cut2)
                    seg = c[js - nn + 1: js + 1][::-1]
                    sum2 = _trapz_uniform_complex(deltas, seg * elm2[:nn])

                # lambda3
                if Relm3[1] < toll:
                    sum3 = c[js] * (-1.0 + np.exp(l3 * deltas)) / l3
                else:
                    nn3 = min(js + 1, cut3)
                    seg3 = c[js - nn3 + 1: js + 1][::-1]
                    sum3 = _trapz_uniform_complex(deltas, seg3 * elm3[:nn3])

                # lambda4
                if elm4[1] < toll:
                    sum4 = c[js] * (-1.0 + np.exp(l4 * deltas)) / l4
                    sum4 = complex(sum4, 0.0)
                else:
                    nn4 = min(js + 1, cut4)
                    seg4 = c[js - nn4 + 1: js + 1][::-1]
                    sum4r = _trapz_uniform_real(deltas, seg4 * elm4[:nn4])
                    sum4 = complex(sum4r, 0.0)

                um2[js] = A * (g20 * sum2 + g30 * sum3 + g40 * sum4)

            # add local + homogeneous
            for js in range(N):
                um3 = A * (c[js] * (g11 + g21 + g31 + g41))
                um4 = c1 * np.exp(l1 * (csi[js] - Elle))
                um5 = c2 * np.exp(l2 * csi[js]) + c3 * np.exp(l3 * csi[js]) + c4 * np.exp(l4 * csi[js])
                um = um1[js] + um2[js] + um3 + um4 + um5
                deltaU[js] += 2.0 * float(um.real) * ((-1.0) ** jm)

        else:
            # --- super-resonant branch ---
            for i in range(N):
                x = float(csi[i])
                elm1[i] = float(np.exp(-l1 * x))
                elm2[i] = np.exp(-l2 * x)
                elm3[i] = np.exp(-l3 * x)
                elm4[i] = float(np.exp(l4 * x))
                Relm2[i] = float(np.exp((-l2.real) * x))
                Relm3[i] = float(np.exp((-l3.real) * x))

            # I1L
            if elm1[1] < toll:
                sum1 = c[0] * (1.0 - np.exp(-l1 * deltas)) / l1
            else:
                nn = min(_first_below(elm1, toll) + 1, N)
                sum1 = _trapz_uniform_real(deltas, c[:nn] * elm1[:nn])

            # I2L (forward)
            if Relm2[1] < toll:
                sum2 = c[0] * (1.0 - np.exp(-l2 * deltas)) / l2
            else:
                nn = min(_first_below(Relm2, toll) + 1, N)
                sum2 = _trapz_uniform_complex(deltas, c[:nn] * elm2[:nn])

            # I3L (forward)
            if Relm3[1] < toll:
                sum3 = c[0] * (1.0 - np.exp(-l3 * deltas)) / l3
            else:
                nn = min(_first_below(Relm3, toll) + 1, N)
                sum3 = _trapz_uniform_complex(deltas, c[:nn] * elm3[:nn])

            # I4L (reverse)
            if elm4[1] < toll:
                sum4 = c[-1] * (-1.0 + np.exp(l4 * deltas)) / l4
                sum4 = complex(sum4, 0.0)
            else:
                nn = min(_first_below(elm4, toll) + 1, N)
                seg = c[::-1][:nn]
                sum4r = _trapz_uniform_real(deltas, seg * elm4[:nn])
                sum4 = complex(sum4r, 0.0)

            B1 = A * (g20 * sum2 + g30 * sum3 + g40 * sum4 + g10 * sum1)
            B2 = A * (g20 * ph2 * sum2 + g30 * ph3 * sum3 + g40 * ph4 * sum4 + g10 * ph1 * sum1)
            B3 = A * (g20 * dl2 * sum2 + g30 * dl3 * sum3 + g40 * dl4 * sum4 + g10 * dl1 * sum1)
            B4 = A * (g20 * cs2 * sum2 + g30 * cs3 * sum3 + g40 * cs4 * sum4 + g10 * cs1 * sum1)

            A1Det = _det3(ph2, ph3, -ph4, dl2, dl3, -dl4, cs2, cs3, -cs4)
            A2Det = _det3(ph1, ph3, -ph4, dl1, dl3, -dl4, cs1, cs3, -cs4)
            A3Det = _det3(ph1, ph2, -ph4, dl1, dl2, -dl4, cs1, cs2, -cs4)
            A4Det = _det3(ph1, ph2, ph3, dl1, dl2, dl3, cs1, cs2, cs3)
            Adet = A1Det - A2Det + A3Det + A4Det

            A2Det = _det3(-B2, ph3, -ph4, -B3, dl3, -dl4, -B4, cs3, -cs4)
            A3Det = _det3(-B2, ph2, -ph4, -B3, dl2, -dl4, -B4, cs2, -cs4)
            A4Det = _det3(-B2, ph2, ph3, -B3, dl2, dl3, -B4, cs2, cs3)
            c1 = (-B1 * A1Det - A2Det + A3Det + A4Det) / Adet

            A1Det = _det3(-B2, ph3, -ph4, -B3, dl3, -dl4, -B4, cs3, -cs4)
            A2Det = _det3(ph1, ph3, -ph4, dl1, dl3, -dl4, cs1, cs3, -cs4)
            A3Det = _det3(ph1, -B2, -ph4, dl1, -B3, -dl4, cs1, -B4, -cs4)
            A4Det = _det3(ph1, -B2, ph3, dl1, -B3, dl3, cs1, -B4, cs3)
            c2 = (A1Det + B1 * A2Det + A3Det + A4Det) / Adet

            A1Det = _det3(ph2, -B2, -ph4, dl2, -B3, -dl4, cs2, -B4, -cs4)
            A2Det = _det3(ph1, -B2, -ph4, dl1, -B3, -dl4, cs1, -B4, -cs4)
            A3Det = _det3(ph1, ph2, -ph4, dl1, dl2, -dl4, cs1, cs2, -cs4)
            A4Det = _det3(ph1, ph2, -B2, dl1, dl2, -B3, cs1, cs2, -B4)
            c3 = (A1Det - A2Det - B1 * A3Det + A4Det) / Adet

            A1Det = _det3(ph2, ph3, -B2, dl2, dl3, -B3, cs2, cs3, -B4)
            A2Det = _det3(ph1, ph3, -B2, dl1, dl3, -B3, cs1, cs3, -B4)
            A3Det = _det3(ph1, ph2, -B2, dl1, dl2, -B3, cs1, cs2, -B4)
            A4Det = _det3(ph1, ph2, ph3, dl1, dl2, dl3, cs1, cs2, cs3)
            c4 = (A1Det - A2Det + A3Det + B1 * A4Det) / Adet

            c1 = c1 / (1.0 - elm1[-1])
            c2 = c2 / (1.0 - elm2[-1])
            c3 = c3 / (1.0 - elm3[-1])
            c4 = c4 / (1.0 - elm4[-1])

            cut1 = _first_below(elm1, toll) + 1
            cut2 = _first_below(Relm2, toll) + 1
            cut3 = _first_below(Relm3, toll) + 1
            cut4 = _first_below(elm4, toll) + 1

            um1 = np.zeros(N, dtype=np.complex128)
            um2 = np.zeros(N, dtype=np.complex128)

            # um1(js): downstream conv with lambdas 1,2,3
            for js in range(N - 1):
                nn = min(N - js, cut1)
                sum1 = _trapz_uniform_real(deltas, c[js: js + nn] * elm1[:nn])

                nn2 = min(N - js, cut2)
                sum2 = _trapz_uniform_complex(deltas, c[js: js + nn2] * elm2[:nn2])

                nn3 = min(N - js, cut3)
                sum3 = _trapz_uniform_complex(deltas, c[js: js + nn3] * elm3[:nn3])

                um1[js] = -A * (g10 * sum1 + g20 * sum2 + g30 * sum3)

            # um2(js): upstream conv with lambda4
            for js in range(1, N):
                if elm4[1] < toll:
                    sum4 = c[js] * (-1.0 + np.exp(l4 * deltas)) / l4
                    sum4 = complex(sum4, 0.0)
                else:
                    nn4 = min(js + 1, cut4)
                    seg4 = c[js - nn4 + 1: js + 1][::-1]
                    sum4r = _trapz_uniform_real(deltas, seg4 * elm4[:nn4])
                    sum4 = complex(sum4r, 0.0)
                um2[js] = A * g40 * sum4

            for js in range(N):
                um3 = A * (c[js] * (g11 + g21 + g31 + g41))
                um4 = c4 * np.exp(l4 * csi[js])
                um5 = (
                    c1 * np.exp(l1 * (csi[js] - Elle))
                    + c2 * np.exp(l2 * (csi[js] - Elle))
                    + c3 * np.exp(l3 * (csi[js] - Elle))
                )
                um = um1[js] + um2[js] + um3 + um4 + um5
                deltaU[js] += 2.0 * float(um.real) * ((-1.0) ** jm)

    return deltaU


# --------------------------
# Public API
# --------------------------

def parall_u_periodic(
    c: np.ndarray,
    s: np.ndarray,
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
    Nn: int,
    Ns: int,
    n: np.ndarray,
    deltas: float,
    *,
    toll: float = TOLL_DEFAULT,
    paral: int = 0,
    n_workers: int | None = None,
) -> tuple[np.ndarray, int]:
    """Periodic-boundary flow-field (U) using the Fortran dUZSBC1 logic.

    Signature matches `parall_u_free` so you can swap in Main.

    Notes
    -----
    - `s` is only used to build the coordinate `csi = s - s[0]`.
      If your `s` is already uniform (0, deltas, 2*deltas, ...), this is identical.
    - `Am` in MATLAB includes (-1)^(jm-1); the Fortran dUZSBC1 applies that sign
      outside. Here we pass `Av_pos = abs(Am)` and keep the external factor,
      matching Fortran.

    Returns
    -------
    U : (Ns,) float64
    flag : int
    """

    c = np.asarray(c, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if c.size != s.size:
        raise ValueError("c and s must have the same length")
    if int(Ns) != int(c.size):
        # keep backward compatibility with older calls that pass Ns explicitly
        Ns = int(c.size)

    Am, lamb1, lamb2, lamb3, lamb4, g10, g20, g30, g40, g11, g21, g31, g41 = _precompute_modes(
        float(Cf0), float(CT), float(CD), float(phiT), float(phiD), float(beta), float(rpic), float(theta0), float(F0), int(Mdat)
    )

    Av_pos = np.abs(Am).astype(np.float64, copy=False)

    csi = s - float(s[0])
    # Some projects store s as 1-based padding; be robust:
    if csi.size >= 2:
        # If deltas passed is inconsistent, still prefer the provided deltas (as Fortran does)
        pass

    U = _duzs_periodic_bc_modes(
        c=c,
        deltas=float(deltas),
        csi=csi,
        lamb1=lamb1,
        lamb2=lamb2,
        lamb3=lamb3,
        lamb4=lamb4,
        Av_pos=Av_pos,
        g10v=g10,
        g20v=g20,
        g30v=g30,
        g40v=g40,
        g11v=g11,
        g21v=g21,
        g31v=g31,
        g41v=g41,
        toll=float(toll),
        paral=int(paral),
        n_workers=n_workers,
    )

    flag = _compute_flag(lamb2)
    return U.astype(np.float64, copy=False), int(flag)


__all__ = [
    "parall_u_periodic",
]
