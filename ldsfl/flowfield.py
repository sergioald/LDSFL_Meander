# ldsfl/flowfield.py
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .semiana import (
    semiana1,
    semiana1_cached,
    semiana2,
    semiana2_cached,
)
from .modes import (
    _compute_flag,
    _precompute_modes,
    _roots_companion_matlab,
    _sort_roots_like_matlab_swaps,
)

__all__ = [
    "parall_u_free",
    "_compute_flag",
    "_precompute_modes",
    "_roots_companion_matlab",
    "_sort_roots_like_matlab_swaps",
]

"""
SL == 1 → :

explicitly splits the solution into um1, um2, plus a “local” term um3

calls int_semiana1 / int_semiana2

has two separate sub-branches depending on resonance:

sub-resonant if real(lm2) < 0

super-resonant if real(lm2) > 0

computes truncation lengths j1toll..j4toll based on exponentials like exp(-lm*s(kk)) etc.

SL == 0 →:

decomposes contributions into upstream-propagating (UPSTR) and downstream-propagating (DWSTR) parts for each eigenvalue

calls SEMIANA1 / SEMIANA2 (or your cached variants in Python)

truncates using exponentials based on distance from the upstream boundary: exp(-real(lambda)*(s(j)-s(1)))

has placeholders for boundary-condition forcing via c_bc, UPSTR_BC, DWSTR_BC (but c_bc=0 in your file, so those are currently inactive)


paral == 0 → runs the jm=1:Mdat Fourier-mode loop serially with normal for loops


# SL=0, serial (your current path)
U, flag = parall_u_free(..., SL=0, paral=0)

# SL=0, parallel over jm (MATLAB paral==1)
U, flag = parall_u_free(..., SL=0, paral=1, n_workers=6)

# SL=1, serial (MATLAB SL==1, paral==0)
U, flag = parall_u_free(..., SL=1, paral=0)

# SL=1, parallel (MATLAB SL==1, paral==1)
U, flag = parall_u_free(..., SL=1, paral=1)


"""

def _prepare_padded(c: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = int(len(c))
    c_pad = np.zeros(N + 1, dtype=np.float64)
    s_pad = np.zeros(N + 1, dtype=np.float64)
    c_pad[1:] = np.asarray(c, dtype=np.float64)
    s_pad[1:] = np.asarray(s, dtype=np.float64)
    return c_pad, s_pad


def _cached_tables_for_lam(
    lam: complex,
    c_pad: np.ndarray,
    deltas: float,
    kmax: int,
    *,
    allow_pos_k: bool,
) -> tuple[np.ndarray, np.ndarray, complex, np.ndarray, int]:
    """Build cA,cB,lmds,exp_table for semiana*_cached."""
    lmds = lam * deltas
    lm2ds = lam * lmds
    aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)

    cA = (c_pad * aux1) / lm2ds
    cB = c_pad / lm2ds

    exp_table = np.zeros(2 * kmax + 1, dtype=np.complex128)
    exp_table[kmax] = 1.0 + 0.0j

    if lam.real > 0.0:
        k_neg = np.arange(-kmax, 0, dtype=np.float64)
        exp_table[0:kmax] = np.exp(lmds * k_neg)
        exp_table[kmax + 1] = np.exp(lmds * 1.0)
    else:
        exp_table[kmax - 1] = np.exp(lmds * (-1.0))
        if allow_pos_k:
            k_pos = np.arange(1, kmax + 1, dtype=np.float64)
            exp_table[kmax + 1:] = np.exp(lmds * k_pos)
        else:
            exp_table[kmax + 1] = np.exp(lmds * 1.0)

    return cA, cB, lmds, exp_table, kmax


def _mode_SL0(
    jm: int,
    Am: np.ndarray,
    lambs: tuple[complex, complex, complex, complex],
    g0s: tuple[complex, complex, complex, complex],
    g1sum: complex,
    c_pad: np.ndarray,
    s_pad: np.ndarray,
    deltas: float,
    *,
    use_cached: bool,
    TOLL: float,
) -> np.ndarray:
    """One jm contribution for SL==0 (UPSTR/DWSTR formulation)."""
    N = len(c_pad) - 1
    UPSTR = np.zeros(N + 1, dtype=np.complex128)
    DWSTR = np.zeros(N + 1, dtype=np.complex128)

    for lam, gj0 in zip(lambs, g0s, strict=True):
        if lam.real > 0.0:
            real_lam = float(lam.real)
            decay = np.exp(-real_lam * (s_pad[1:] - s_pad[1]))
            jtoll = int(1 + np.sum(decay > TOLL))

            if use_cached:
                kmax = max(1, min(N, jtoll - 1))
                cA, cB, lmds, exp_table, kmax = _cached_tables_for_lam(lam, c_pad, deltas, kmax, allow_pos_k=False)
                for j in range(1, N):
                    jsend = min(j + jtoll - 1, N)
                    CONV = semiana1_cached(j, jsend, cA, cB, lmds, exp_table, kmax)
                    UPSTR[j] -= Am[jm - 1] * gj0 * CONV
            else:
                for j in range(1, N):
                    jsend = min(j + jtoll - 1, N)
                    CONV = semiana1(j, N, c_pad, jsend, deltas, lam)
                    UPSTR[j] -= Am[jm - 1] * gj0 * CONV

        elif lam.real < 0.0:
            real_lam = float(lam.real)
            decay = np.exp(real_lam * (s_pad[1:] - s_pad[1]))
            jtoll = int(1 + np.sum(decay > TOLL))

            if use_cached:
                kmax = max(1, min(N, jtoll - 1))
                cA, cB, lmds, exp_table, kmax = _cached_tables_for_lam(lam, c_pad, deltas, kmax, allow_pos_k=True)
                for j in range(N, 1, -1):
                    jsend = max(j - jtoll + 1, 1)
                    CONV = semiana2_cached(j, jsend, cA, cB, lmds, exp_table, kmax)
                    DWSTR[j] += Am[jm - 1] * gj0 * CONV
            else:
                for j in range(N, 1, -1):
                    jsend = max(j - jtoll + 1, 1)
                    CONV = semiana2(j, N, c_pad, jsend, deltas, lam)
                    DWSTR[j] += Am[jm - 1] * gj0 * CONV

    LOCAL = Am[jm - 1] * c_pad * g1sum
    um = np.real(UPSTR + DWSTR + LOCAL)
    return 2.0 * um * ((-1) ** (jm - 1))


def _mode_SL1(
    jm: int,
    Am: np.ndarray,
    lambs: tuple[complex, complex, complex, complex],
    g0s: tuple[complex, complex, complex, complex],
    g1sum: complex,
    c_pad: np.ndarray,
    s_pad: np.ndarray,
    deltas: float,
    *,
    use_cached: bool,
    toll: float,
) -> np.ndarray:
    """One jm contribution for SL==1 (um1/um2/um3 formulation)."""
    N = len(c_pad) - 1
    lm1, lm2, lm3, lm4 = lambs
    g10jm, g20jm, g30jm, g40jm = g0s

    svec = s_pad[1:]
    lm1r = float(np.real(lm1))
    lm2r = float(np.real(lm2))
    lm3r = float(np.real(lm3))
    lm4r = float(np.real(lm4))

    if lm2r < 0.0:
        j1toll = int(1 + np.sum(np.exp(-lm1r * svec) > toll))
        j2toll = int(1 + np.sum(np.exp(lm2r * svec) > toll))
        j3toll = int(1 + np.sum(np.exp(lm3r * svec) > toll))
        j4toll = int(1 + np.sum(np.exp(lm4r * svec) > toll))

        um1 = np.zeros(N + 1, dtype=np.complex128)
        um2 = np.zeros(N + 1, dtype=np.complex128)

        if use_cached:
            cA1, cB1, lmds1, exp1, km1 = _cached_tables_for_lam(lm1, c_pad, deltas, max(1, min(N, j1toll - 1)), allow_pos_k=False)
            cA2, cB2, lmds2, exp2, km2 = _cached_tables_for_lam(lm2, c_pad, deltas, max(1, min(N, j2toll - 1)), allow_pos_k=True)
            cA3, cB3, lmds3, exp3, km3 = _cached_tables_for_lam(lm3, c_pad, deltas, max(1, min(N, j3toll - 1)), allow_pos_k=True)
            cA4, cB4, lmds4, exp4, km4 = _cached_tables_for_lam(lm4, c_pad, deltas, max(1, min(N, j4toll - 1)), allow_pos_k=True)

        for js in range(1, N):
            jsend = min(js + j1toll - 1, N)
            if use_cached:
                sum1 = semiana1_cached(js, jsend, cA1, cB1, lmds1, exp1, km1)
            else:
                sum1 = semiana1(js, N, c_pad, jsend, deltas, lm1)
            um1[js] = -Am[jm - 1] * g10jm * sum1

        for js in range(2, N + 1):
            jsend2 = max(js - j2toll + 1, 1)
            jsend3 = max(js - j3toll + 1, 1)
            jsend4 = max(js - j4toll + 1, 1)
            if use_cached:
                sum2 = semiana2_cached(js, jsend2, cA2, cB2, lmds2, exp2, km2)
                sum3 = semiana2_cached(js, jsend3, cA3, cB3, lmds3, exp3, km3)
                sum4 = semiana2_cached(js, jsend4, cA4, cB4, lmds4, exp4, km4)
            else:
                sum2 = semiana2(js, N, c_pad, jsend2, deltas, lm2)
                sum3 = semiana2(js, N, c_pad, jsend3, deltas, lm3)
                sum4 = semiana2(js, N, c_pad, jsend4, deltas, lm4)
            um2[js] = Am[jm - 1] * (g20jm * sum2 + g30jm * sum3 + g40jm * sum4)

    elif lm2r > 0.0:
        j1toll = int(1 + np.sum(np.exp(-lm1r * svec) > toll))
        j2toll = int(1 + np.sum(np.exp(-lm2r * svec) > toll))
        j3toll = int(1 + np.sum(np.exp(-lm3r * svec) > toll))
        j4toll = int(1 + np.sum(np.exp(lm4r * svec) > toll))

        um1 = np.zeros(N + 1, dtype=np.complex128)
        um2 = np.zeros(N + 1, dtype=np.complex128)

        if use_cached:
            cA1, cB1, lmds1, exp1, km1 = _cached_tables_for_lam(lm1, c_pad, deltas, max(1, min(N, j1toll - 1)), allow_pos_k=False)
            cA2, cB2, lmds2, exp2, km2 = _cached_tables_for_lam(lm2, c_pad, deltas, max(1, min(N, j2toll - 1)), allow_pos_k=False)
            cA3, cB3, lmds3, exp3, km3 = _cached_tables_for_lam(lm3, c_pad, deltas, max(1, min(N, j3toll - 1)), allow_pos_k=False)
            cA4, cB4, lmds4, exp4, km4 = _cached_tables_for_lam(lm4, c_pad, deltas, max(1, min(N, j4toll - 1)), allow_pos_k=True)

        for js in range(1, N):
            jsend1 = min(js + j1toll - 1, N)
            jsend2 = min(js + j2toll - 1, N)
            jsend3 = min(js + j3toll - 1, N)
            if use_cached:
                sum1 = semiana1_cached(js, jsend1, cA1, cB1, lmds1, exp1, km1)
                sum2 = semiana1_cached(js, jsend2, cA2, cB2, lmds2, exp2, km2)
                sum3 = semiana1_cached(js, jsend3, cA3, cB3, lmds3, exp3, km3)
            else:
                sum1 = semiana1(js, N, c_pad, jsend1, deltas, lm1)
                sum2 = semiana1(js, N, c_pad, jsend2, deltas, lm2)
                sum3 = semiana1(js, N, c_pad, jsend3, deltas, lm3)
            um1[js] = -Am[jm - 1] * (g10jm * sum1 + g20jm * sum2 + g30jm * sum3)

        for js in range(2, N + 1):
            jsend4 = max(js - j4toll + 1, 1)
            if use_cached:
                sum4 = semiana2_cached(js, jsend4, cA4, cB4, lmds4, exp4, km4)
            else:
                sum4 = semiana2(js, N, c_pad, jsend4, deltas, lm4)
            um2[js] = Am[jm - 1] * g40jm * sum4

    else:
        # Borderline resonance: fall back to SL==0
        return _mode_SL0(jm, Am, lambs, g0s, g1sum, c_pad, s_pad, deltas, use_cached=use_cached, TOLL=1e-4)

    um3 = Am[jm - 1] * c_pad * g1sum
    um = np.real(um1 + um2 + um3)
    return 2.0 * um * ((-1) ** (jm - 1))


def _run_modes(
    *,
    SL: int,
    paral: int,
    use_cached: bool,
    Am: np.ndarray,
    lamb1: np.ndarray,
    lamb2: np.ndarray,
    lamb3: np.ndarray,
    lamb4: np.ndarray,
    g10: np.ndarray,
    g20: np.ndarray,
    g30: np.ndarray,
    g40: np.ndarray,
    g11: np.ndarray,
    g21: np.ndarray,
    g31: np.ndarray,
    g41: np.ndarray,
    c_pad: np.ndarray,
    s_pad: np.ndarray,
    deltas: float,
    n_workers: int | None,
) -> np.ndarray:
    Mdat = Am.size
    N = len(c_pad) - 1

    def mode_fn(jm: int) -> np.ndarray:
        lambs = (lamb1[jm - 1], lamb2[jm - 1], lamb3[jm - 1], lamb4[jm - 1])
        g0s = (g10[jm - 1], g20[jm - 1], g30[jm - 1], g40[jm - 1])
        g1sum = g11[jm - 1] + g21[jm - 1] + g31[jm - 1] + g41[jm - 1]
        if SL == 1:
            return _mode_SL1(jm, Am, lambs, g0s, g1sum, c_pad, s_pad, deltas, use_cached=use_cached, toll=1e-4)
        return _mode_SL0(jm, Am, lambs, g0s, g1sum, c_pad, s_pad, deltas, use_cached=use_cached, TOLL=1e-4)

    Ucplot = np.zeros(N + 1, dtype=np.float64)

    if paral == 0:
        for jm in range(1, Mdat + 1):
            Ucplot += mode_fn(jm).real.astype(np.float64, copy=False)
        return Ucplot

    max_workers = n_workers if n_workers is not None else (os.cpu_count() or 1)
    dUs: list[np.ndarray | None] = [None] * (Mdat + 1)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(mode_fn, jm): jm for jm in range(1, Mdat + 1)}
        for fut, jm in futures.items():
            dUs[jm] = fut.result()

    for jm in range(1, Mdat + 1):
        Ucplot += dUs[jm].real.astype(np.float64, copy=False)

    return Ucplot


def _run_modes_numba(
    *,
    SL: int,
    paral: int,
    numba_parallel: bool,
    numba_fastmath: bool,
    use_cached: bool,
    Am: np.ndarray,
    lamb1: np.ndarray,
    lamb2: np.ndarray,
    lamb3: np.ndarray,
    lamb4: np.ndarray,
    g10: np.ndarray,
    g20: np.ndarray,
    g30: np.ndarray,
    g40: np.ndarray,
    g11: np.ndarray,
    g21: np.ndarray,
    g31: np.ndarray,
    g41: np.ndarray,
    c_pad: np.ndarray,
    s_pad: np.ndarray,
    deltas: float,
    n_workers: int | None,
) -> np.ndarray:
    """Numba-accelerated mode runner (CPU JIT).

    Improvement 7:
    - Avoids allocating large complex work arrays per mode by accumulating the
      *real* contribution directly into the output.
    - Avoids allocating the large cA/cB arrays (size N) by using scalar
      coefficients in the cached SEMIANA kernels.
    """

    try:
        from .flowfield_numba import get_mode_adders  # noqa: WPS433 (local import)
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "backend='numba' requested but numba acceleration could not be imported. "
            "Install numba (e.g. `conda install numba`) or use backend='numpy'."
        ) from e

    add_sl0, add_sl1 = get_mode_adders(parallel=bool(numba_parallel), fastmath=bool(numba_fastmath))

    Mdat = int(Am.size)
    N = int(len(c_pad) - 1)

    def _add_one_mode(out: np.ndarray, jm: int) -> None:
        lambs = (lamb1[jm - 1], lamb2[jm - 1], lamb3[jm - 1], lamb4[jm - 1])
        g0s = (g10[jm - 1], g20[jm - 1], g30[jm - 1], g40[jm - 1])
        g1sum = g11[jm - 1] + g21[jm - 1] + g31[jm - 1] + g41[jm - 1]
        if SL == 1:
            add_sl1(out, jm, Am, lambs, g0s, g1sum, c_pad, s_pad, float(deltas), bool(use_cached), 1e-4)
        else:
            add_sl0(out, jm, Am, lambs, g0s, g1sum, c_pad, s_pad, float(deltas), bool(use_cached), 1e-4)

    Ucplot = np.zeros(N + 1, dtype=np.float64)

    if paral == 0:
        for jm in range(1, Mdat + 1):
            _add_one_mode(Ucplot, jm)
        return Ucplot

    # Parallel over modes: each worker gets its own accumulator (avoid races)
    max_workers = n_workers if n_workers is not None else (os.cpu_count() or 1)

    def _worker(jm: int) -> np.ndarray:
        tmp = np.zeros(N + 1, dtype=np.float64)
        _add_one_mode(tmp, jm)
        return tmp

    dUs: list[np.ndarray | None] = [None] * (Mdat + 1)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, jm): jm for jm in range(1, Mdat + 1)}
        for fut, jm in futures.items():
            dUs[jm] = fut.result()

    for jm in range(1, Mdat + 1):
        Ucplot += dUs[jm]  # type: ignore[operator]

    return Ucplot


def parall_u_free(
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
    SL: int = 0,
    paral: int = 0,
    n_workers: int | None = None,
    backend: str = "numpy",
    numba_parallel: bool = False,
    numba_fastmath: bool = False,
):
    """
    Full Parall_U_free with switches:
      SL=0/1, paral=0/1.

    This version uses semiana*_cached (fast).
    """
    Am, lamb1, lamb2, lamb3, lamb4, g10, g20, g30, g40, g11, g21, g31, g41 = _precompute_modes(
        Cf0, CT, CD, phiT, phiD, beta, rpic, theta0, F0, Mdat
    )

    c_pad, s_pad = _prepare_padded(c, s)

    backend = str(backend).lower().strip()
    if backend not in ("numpy", "numba"):
        raise ValueError(f"Unknown backend: {backend!r}. Use 'numpy' or 'numba'.")

    if backend == "numba":
        # Avoid nested parallelism: Python threads over modes + Numba threads over points
        # can oversubscribe CPU cores and slow things down.
        if int(paral) == 1 and bool(numba_parallel):
            import warnings
            warnings.warn("Both flow_paral=1 and numba_parallel=True are enabled; this may oversubscribe CPU cores. Consider using only one.",stacklevel=2)

        Ucplot_pad = _run_modes_numba(
            SL=int(SL),
            paral=int(paral),
            numba_parallel=bool(numba_parallel),
            numba_fastmath=bool(numba_fastmath),
            use_cached=True,
            Am=Am,
            lamb1=lamb1,
            lamb2=lamb2,
            lamb3=lamb3,
            lamb4=lamb4,
            g10=g10,
            g20=g20,
            g30=g30,
            g40=g40,
            g11=g11,
            g21=g21,
            g31=g31,
            g41=g41,
            c_pad=c_pad,
            s_pad=s_pad,
            deltas=float(deltas),
            n_workers=n_workers,
        )
    else:
        Ucplot_pad = _run_modes(
            SL=int(SL),
            paral=int(paral),
            use_cached=True,
            Am=Am,
            lamb1=lamb1,
            lamb2=lamb2,
            lamb3=lamb3,
            lamb4=lamb4,
            g10=g10,
            g20=g20,
            g30=g30,
            g40=g40,
            g11=g11,
            g21=g21,
            g31=g31,
            g41=g41,
            c_pad=c_pad,
            s_pad=s_pad,
            deltas=float(deltas),
            n_workers=n_workers,
        )

    flag = _compute_flag(lamb2)
    return Ucplot_pad[1:].astype(np.float64), int(flag)


def parall_u_free_or(
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
    SL: int = 0,
    paral: int = 0,
    n_workers: int | None = None,
    backend: str = "numpy",
    numba_parallel: bool = False,
    numba_fastmath: bool = False,
):
    """
    Same switches as parall_u_free, but uses direct semiana1/2 (debug/matching-first).
    """
    Am, lamb1, lamb2, lamb3, lamb4, g10, g20, g30, g40, g11, g21, g31, g41 = _precompute_modes(
        Cf0, CT, CD, phiT, phiD, beta, rpic, theta0, F0, Mdat
    )

    c_pad, s_pad = _prepare_padded(c, s)

    # The "_or" path is intended for debugging/matching-first and is slower.
    # We currently keep it on the NumPy reference implementation regardless of backend.
    Ucplot_pad = _run_modes(
        SL=int(SL),
        paral=int(paral),
        use_cached=False,
        Am=Am,
        lamb1=lamb1,
        lamb2=lamb2,
        lamb3=lamb3,
        lamb4=lamb4,
        g10=g10,
        g20=g20,
        g30=g30,
        g40=g40,
        g11=g11,
        g21=g21,
        g31=g31,
        g41=g41,
        c_pad=c_pad,
        s_pad=s_pad,
        deltas=float(deltas),
        n_workers=n_workers,
    )

    flag = _compute_flag(lamb2)
    return Ucplot_pad[1:].astype(np.float64), int(flag)
