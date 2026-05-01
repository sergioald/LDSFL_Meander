"""Optional Numba acceleration for the LDSFL-Meander free-boundary flow-field.

This module is imported *only* when backend='numba' is requested.

Design goal
-----------
Keep the reference NumPy algebra intact, but JIT-compile the heavy inner loops:
the (potentially O(N*jtoll)) SEMIANA summations.

Notes
-----
* First call will trigger JIT compilation (one-off cost).
* We intentionally keep truncation-length logic (jtoll) and exp-table building
  in NumPy because they are already fast and simpler to debug.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# --- Compatibility shim for a known `numba` ↔ `coverage` API mismatch ---
# Some environments ship a `coverage` version where `coverage.types.Tracer` (and
# related typing aliases) are missing, causing `import numba` to fail.
# The aliases are only needed for type annotations, so we can safely provide
# minimal placeholders.
try:  # pragma: no cover
    import coverage  # type: ignore

    if not hasattr(coverage, "types"):
        class _Types:  # noqa: WPS431 (nested class)
            pass
        coverage.types = _Types()  # type: ignore[attr-defined]

    if not hasattr(coverage.types, "Tracer"):
        class _Tracer:  # noqa: WPS431
            pass
        coverage.types.Tracer = _Tracer  # type: ignore[attr-defined]

    for _name in (
        "TTraceData",
        "TShouldTraceFn",
        "TFileDisposition",
        "TShouldStartContextFn",
        "TWarnFn",
        "TTraceFn",
    ):
        if not hasattr(coverage.types, _name):
            setattr(coverage.types, _name, object)
except Exception:
    # If coverage is not installed (or anything else goes wrong), just continue.
    pass

try:
    from numba import njit
except Exception as e:  # pragma: no cover
    # Provide a clearer error when imported without numba.
    raise ImportError(
        "Numba is required for backend='numba'. Install it (e.g. `conda install numba`)."
    ) from e


# -----------------------------
# JIT kernels (inner loops)
# -----------------------------


@njit(cache=False, nogil=True)
def _semiana1_cached_nb(
    js: int,
    jsend: int,
    cA: np.ndarray,
    cB: np.ndarray,
    lmds: complex,
    exp_table: np.ndarray,
    kmax: int,
) -> complex:
    """Numba version of semiana1_cached (1-based indexing)."""
    conv = cB[js] * (exp_table[kmax - 1] + lmds - 1.0)
    for j in range(js + 1, jsend):
        conv += cA[j] * exp_table[kmax + (js - j)]
    conv -= cB[jsend] * (1.0 + lmds - exp_table[kmax + 1]) * exp_table[kmax + (js - jsend)]
    return conv


@njit(cache=False, nogil=True)
def _semiana2_cached_nb(
    js: int,
    jsend: int,
    cA: np.ndarray,
    cB: np.ndarray,
    lmds: complex,
    exp_table: np.ndarray,
    kmax: int,
) -> complex:
    """Numba version of semiana2_cached (1-based indexing)."""
    conv = cB[jsend] * (-1.0 + lmds + exp_table[kmax - 1]) * exp_table[kmax + (js - jsend)]
    for j in range(jsend + 1, js):
        conv += cA[j] * exp_table[kmax + (js - j)]
    conv -= cB[js] * (1.0 + lmds - exp_table[kmax + 1])
    return conv


@njit(cache=False, nogil=True)
def _semiana1_direct_nb(js: int, jsend: int, c: np.ndarray, deltas: float, lm: complex) -> complex:
    """Direct SEMIANA1 (no cached exp_table)."""
    lmds = lm * deltas
    lm2ds = lm * lmds
    aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)
    conv = c[js] / lm2ds * (np.exp(-lmds) + lmds - 1.0)
    for j in range(js + 1, jsend):
        conv += (c[j] * aux1 / lm2ds) * np.exp(lmds * float(js - j))
    conv -= c[jsend] / lm2ds * (1.0 + lmds - np.exp(lmds)) * np.exp(lmds * float(js - jsend))
    return conv


@njit(cache=False, nogil=True)
def _semiana2_direct_nb(js: int, jsend: int, c: np.ndarray, deltas: float, lm: complex) -> complex:
    """Direct SEMIANA2 (no cached exp_table)."""
    lmds = lm * deltas
    lm2ds = lm * lmds
    aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)
    conv = c[jsend] / lm2ds * (-1.0 + lmds + np.exp(-lmds)) * np.exp(lmds * float(js - jsend))
    for j in range(jsend + 1, js):
        conv += (c[j] * aux1 / lm2ds) * np.exp(lmds * float(js - j))
    conv -= c[js] / lm2ds * (1.0 + lmds - np.exp(lmds))
    return conv


@njit(cache=False, nogil=True)
def _fill_upstr_cached_nb(
    out: np.ndarray,
    cA: np.ndarray,
    cB: np.ndarray,
    lmds: complex,
    exp_table: np.ndarray,
    kmax: int,
    jtoll: int,
    N: int,
    Am_gj0: complex,
) -> None:
    for js in range(1, N):
        jsend = js + jtoll - 1
        if jsend > N:
            jsend = N
        out[js] -= Am_gj0 * _semiana1_cached_nb(js, jsend, cA, cB, lmds, exp_table, kmax)


@njit(cache=False, nogil=True)
def _fill_dwstr_cached_nb(
    out: np.ndarray,
    cA: np.ndarray,
    cB: np.ndarray,
    lmds: complex,
    exp_table: np.ndarray,
    kmax: int,
    jtoll: int,
    N: int,
    Am_gj0: complex,
) -> None:
    for js in range(N, 1, -1):
        jsend = js - jtoll + 1
        if jsend < 1:
            jsend = 1
        out[js] += Am_gj0 * _semiana2_cached_nb(js, jsend, cA, cB, lmds, exp_table, kmax)


@njit(cache=False, nogil=True)
def _fill_upstr_direct_nb(
    out: np.ndarray,
    c_pad: np.ndarray,
    deltas: float,
    lm: complex,
    jtoll: int,
    N: int,
    Am_gj0: complex,
) -> None:
    for js in range(1, N):
        jsend = js + jtoll - 1
        if jsend > N:
            jsend = N
        out[js] -= Am_gj0 * _semiana1_direct_nb(js, jsend, c_pad, deltas, lm)


@njit(cache=False, nogil=True)
def _fill_dwstr_direct_nb(
    out: np.ndarray,
    c_pad: np.ndarray,
    deltas: float,
    lm: complex,
    jtoll: int,
    N: int,
    Am_gj0: complex,
) -> None:
    for js in range(N, 1, -1):
        jsend = js - jtoll + 1
        if jsend < 1:
            jsend = 1
        out[js] += Am_gj0 * _semiana2_direct_nb(js, jsend, c_pad, deltas, lm)


# -----------------------------
# Small NumPy helpers
# -----------------------------


def _cached_tables_for_lam_np(
    lam: complex,
    c_pad: np.ndarray,
    deltas: float,
    kmax: int,
    *,
    allow_pos_k: bool,
) -> Tuple[np.ndarray, np.ndarray, complex, np.ndarray, int]:
    """Precompute cA/cB + exp_table for cached SEMIANA.

    The earlier "scalar coeff" variant avoided allocating cA/cB arrays, but it
    introduces extra multiplications inside the innermost SEMIANA loops. For the
    typical LDSFL-Meander workloads, that compute cost outweighs the allocation savings.

    Returns
    -------
    cA, cB, lmds, exp_table, kmax
    """
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



def _jtoll_for_lam(lam: complex, s_pad: np.ndarray, toll: float, *, upwind: bool) -> int:
    """Compute truncation length `jtoll` exactly like the reference flowfield.py."""
    if upwind:
        real_lam = float(lam.real)
        decay = np.exp(-real_lam * (s_pad[1:] - s_pad[1]))
    else:
        real_lam = float(lam.real)
        decay = np.exp(real_lam * (s_pad[1:] - s_pad[1]))
    return int(1 + np.sum(decay > toll))


# -----------------------------
# Public entry points
# -----------------------------



from numba import prange

_MODE_FN_CACHE = {}
_MODE_ADDER_CACHE = {}


def _make_fill_kernels(*, parallel: bool, fastmath: bool):
    """Create (and JIT-compile on first use) fill kernels with the given options.

    These kernels update a *real* output array directly (we only ever use the real
    part of the full complex sum), which avoids allocating large complex work
    arrays per mode.
    """

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def semiana1_cached(js, jsend, cA, cB, lmds, exp_table, kmax):
        conv = cB[js] * (exp_table[kmax - 1] + lmds - 1.0)
        for j in range(js + 1, jsend):
            conv += cA[j] * exp_table[kmax + (js - j)]
        conv -= cB[jsend] * (1.0 + lmds - exp_table[kmax + 1]) * exp_table[kmax + (js - jsend)]
        return conv

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def semiana2_cached(js, jsend, cA, cB, lmds, exp_table, kmax):
        conv = cB[jsend] * (-1.0 + lmds + exp_table[kmax - 1]) * exp_table[kmax + (js - jsend)]
        for j in range(jsend + 1, js):
            conv += cA[j] * exp_table[kmax + (js - j)]
        conv -= cB[js] * (1.0 + lmds - exp_table[kmax + 1])
        return conv

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def semiana1_direct(js, jsend, c, deltas, lm):
        lmds = lm * deltas
        lm2ds = lm * lmds
        aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)
        conv = c[js] / lm2ds * (np.exp(-lmds) + lmds - 1.0)
        for j in range(js + 1, jsend):
            conv += (c[j] * aux1 / lm2ds) * np.exp(lmds * float(js - j))
        conv -= c[jsend] / lm2ds * (1.0 + lmds - np.exp(lmds)) * np.exp(lmds * float(js - jsend))
        return conv

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def semiana2_direct(js, jsend, c, deltas, lm):
        lmds = lm * deltas
        lm2ds = lm * lmds
        aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)
        conv = c[jsend] / lm2ds * (-1.0 + lmds + np.exp(-lmds)) * np.exp(lmds * float(js - jsend))
        for j in range(jsend + 1, js):
            conv += (c[j] * aux1 / lm2ds) * np.exp(lmds * float(js - j))
        conv -= c[js] / lm2ds * (1.0 + lmds - np.exp(lmds))
        return conv

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def fill_upstr_cached_real(out, cA, cB, lmds, exp_table, kmax, jtoll, N, coeff):
        for js in prange(1, N):
            jsend = js + jtoll - 1
            if jsend > N:
                jsend = N
            out[js] += (coeff * semiana1_cached(js, jsend, cA, cB, lmds, exp_table, kmax)).real

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def fill_dwstr_cached_real(out, cA, cB, lmds, exp_table, kmax, jtoll, N, coeff):
        for idx in prange(N - 1):
            js = N - idx
            if js <= 1:
                continue
            jsend = js - jtoll + 1
            if jsend < 1:
                jsend = 1
            out[js] += (coeff * semiana2_cached(js, jsend, cA, cB, lmds, exp_table, kmax)).real

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def fill_upstr_direct_real(out, c_pad, deltas, lm, jtoll, N, coeff):
        for js in prange(1, N):
            jsend = js + jtoll - 1
            if jsend > N:
                jsend = N
            out[js] += (coeff * semiana1_direct(js, jsend, c_pad, deltas, lm)).real

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def fill_dwstr_direct_real(out, c_pad, deltas, lm, jtoll, N, coeff):
        for idx in prange(N - 1):
            js = N - idx
            if js <= 1:
                continue
            jsend = js - jtoll + 1
            if jsend < 1:
                jsend = 1
            out[js] += (coeff * semiana2_direct(js, jsend, c_pad, deltas, lm)).real

    @njit(cache=False, nogil=True, parallel=parallel, fastmath=fastmath)
    def add_local_real(out, c_pad, N, coeff_real):
        for js in prange(N + 1):
            out[js] += coeff_real * c_pad[js]

    return (
        fill_upstr_cached_real,
        fill_dwstr_cached_real,
        fill_upstr_direct_real,
        fill_dwstr_direct_real,
        add_local_real,
    )



def get_mode_functions(*, parallel: bool = False, fastmath: bool = False):
    """Return (mode_sl0, mode_sl1) functions using kernels compiled with given options.

    The returned functions allocate and return a fresh float64 array (N+1) with the
    contribution of a single Fourier mode.

    Use `get_mode_adders()` if you want in-place accumulation into a shared array.
    """
    key = (bool(parallel), bool(fastmath))
    if key in _MODE_FN_CACHE:
        return _MODE_FN_CACHE[key]

    (
        fill_upstr_cached_real,
        fill_dwstr_cached_real,
        fill_upstr_direct_real,
        fill_dwstr_direct_real,
        add_local_real,
    ) = _make_fill_kernels(parallel=key[0], fastmath=key[1])

    # Small helpers are inlined inside add_mode_* to avoid repeatedly allocating
    # `dist = s_pad[1:] - s_pad[1]` for every eigenvalue.

    def add_mode_sl0(
        out: np.ndarray,
        jm: int,
        Am: np.ndarray,
        lambs: Tuple[complex, complex, complex, complex],
        g0s: Tuple[complex, complex, complex, complex],
        g1sum: complex,
        c_pad: np.ndarray,
        s_pad: np.ndarray,
        deltas: float,
        use_cached: bool,
        toll: float,
    ) -> None:
        N = int(len(c_pad) - 1)
        Aj = float(Am[jm - 1])
        mode_scale = 2.0 * (1.0 if ((jm - 1) % 2 == 0) else -1.0)

        # Precompute distances once per mode (used by all eigenvalues).
        dist0 = s_pad[1:] - s_pad[1]
        log_toll = -np.log(toll)

        for lam, gj0 in zip(lambs, g0s):
            if lam.real > 0.0:
                real_lam_abs = abs(float(lam.real))
                if real_lam_abs <= 0.0:
                    jtoll = int(len(s_pad))
                else:
                    limit = log_toll / real_lam_abs
                    jtoll = 1 + int(np.searchsorted(dist0, limit, side='left'))
                kmax = max(1, min(N, jtoll - 1))
                coeff = complex(mode_scale * (-Aj), 0.0) * gj0
                if use_cached:
                    cA, cB, lmds, exp_table, kmax = _cached_tables_for_lam_np(
                        lam, c_pad, deltas, kmax, allow_pos_k=False
                    )
                    fill_upstr_cached_real(out, cA, cB, lmds, exp_table, kmax, jtoll, N, coeff)
                else:
                    fill_upstr_direct_real(out, c_pad, float(deltas), lam, jtoll, N, coeff)

            elif lam.real < 0.0:
                real_lam_abs = abs(float(lam.real))
                if real_lam_abs <= 0.0:
                    jtoll = int(len(s_pad))
                else:
                    limit = log_toll / real_lam_abs
                    jtoll = 1 + int(np.searchsorted(dist0, limit, side='left'))
                kmax = max(1, min(N, jtoll - 1))
                coeff = complex(mode_scale * Aj, 0.0) * gj0
                if use_cached:
                    cA, cB, lmds, exp_table, kmax = _cached_tables_for_lam_np(
                        lam, c_pad, deltas, kmax, allow_pos_k=True
                    )
                    fill_dwstr_cached_real(out, cA, cB, lmds, exp_table, kmax, jtoll, N, coeff)
                else:
                    fill_dwstr_direct_real(out, c_pad, float(deltas), lam, jtoll, N, coeff)

        local_coeff = (complex(mode_scale * Aj, 0.0) * g1sum).real
        add_local_real(out, c_pad, N, local_coeff)

    def add_mode_sl1(
        out: np.ndarray,
        jm: int,
        Am: np.ndarray,
        lambs: Tuple[complex, complex, complex, complex],
        g0s: Tuple[complex, complex, complex, complex],
        g1sum: complex,
        c_pad: np.ndarray,
        s_pad: np.ndarray,
        deltas: float,
        use_cached: bool,
        toll: float,
    ) -> None:
        N = int(len(c_pad) - 1)
        Aj = float(Am[jm - 1])
        mode_scale = 2.0 * (1.0 if ((jm - 1) % 2 == 0) else -1.0)

        lm1, lm2, lm3, lm4 = lambs
        g10jm, g20jm, g30jm, g40jm = g0s
        lm1r = float(np.real(lm1))
        lm2r = float(np.real(lm2))
        lm3r = float(np.real(lm3))
        lm4r = float(np.real(lm4))

        # SL1 truncation uses absolute s (view, no allocation)
        svec = s_pad[1:]
        log_toll = -np.log(toll)

        def jtoll_sl1(real_lam_abs: float) -> int:
            if real_lam_abs <= 0.0:
                return int(len(s_pad))
            limit = log_toll / real_lam_abs
            return 1 + int(np.searchsorted(svec, limit, side='left'))

        if lm2r < 0.0:
            j1toll = jtoll_sl1(abs(lm1r))
            j2toll = jtoll_sl1(abs(lm2r))
            j3toll = jtoll_sl1(abs(lm3r))
            j4toll = jtoll_sl1(abs(lm4r))

            # um1: -Am*(g10*sum1 + g20*sum2 + g30*sum3) using SEMIANA1
            if use_cached:
                # lm1 (allow_pos_k False)
                kmax1 = max(1, min(N, j1toll - 1))
                cA1, cB1, lmds1, exp1, kmax1 = _cached_tables_for_lam_np(lm1, c_pad, deltas, kmax1, allow_pos_k=False)
                fill_upstr_cached_real(out, cA1, cB1, lmds1, exp1, kmax1, j1toll, N, complex(mode_scale * (-Aj), 0.0) * g10jm)

                # lm2/lm3 in the reference use allow_pos_k True
                kmax2 = max(1, min(N, j2toll - 1))
                cA2, cB2, lmds2, exp2, kmax2 = _cached_tables_for_lam_np(lm2, c_pad, deltas, kmax2, allow_pos_k=True)
                fill_upstr_cached_real(out, cA2, cB2, lmds2, exp2, kmax2, j2toll, N, complex(mode_scale * (-Aj), 0.0) * g20jm)

                kmax3 = max(1, min(N, j3toll - 1))
                cA3, cB3, lmds3, exp3, kmax3 = _cached_tables_for_lam_np(lm3, c_pad, deltas, kmax3, allow_pos_k=True)
                fill_upstr_cached_real(out, cA3, cB3, lmds3, exp3, kmax3, j3toll, N, complex(mode_scale * (-Aj), 0.0) * g30jm)

                # um2: +Am*g40*sum4 using SEMIANA2
                kmax4 = max(1, min(N, j4toll - 1))
                cA4, cB4, lmds4, exp4, kmax4 = _cached_tables_for_lam_np(lm4, c_pad, deltas, kmax4, allow_pos_k=True)
                fill_dwstr_cached_real(out, cA4, cB4, lmds4, exp4, kmax4, j4toll, N, complex(mode_scale * Aj, 0.0) * g40jm)
            else:
                fill_upstr_direct_real(out, c_pad, float(deltas), lm1, j1toll, N, complex(mode_scale * (-Aj), 0.0) * g10jm)
                fill_upstr_direct_real(out, c_pad, float(deltas), lm2, j2toll, N, complex(mode_scale * (-Aj), 0.0) * g20jm)
                fill_upstr_direct_real(out, c_pad, float(deltas), lm3, j3toll, N, complex(mode_scale * (-Aj), 0.0) * g30jm)
                fill_dwstr_direct_real(out, c_pad, float(deltas), lm4, j4toll, N, complex(mode_scale * Aj, 0.0) * g40jm)

        elif lm2r > 0.0:
            j1toll = jtoll_sl1(abs(lm1r))
            j2toll = jtoll_sl1(abs(lm2r))
            j3toll = jtoll_sl1(abs(lm3r))
            j4toll = jtoll_sl1(abs(lm4r))

            if use_cached:
                kmax1 = max(1, min(N, j1toll - 1))
                cA1, cB1, lmds1, exp1, kmax1 = _cached_tables_for_lam_np(lm1, c_pad, deltas, kmax1, allow_pos_k=False)
                fill_upstr_cached_real(out, cA1, cB1, lmds1, exp1, kmax1, j1toll, N, complex(mode_scale * (-Aj), 0.0) * g10jm)

                kmax2 = max(1, min(N, j2toll - 1))
                cA2, cB2, lmds2, exp2, kmax2 = _cached_tables_for_lam_np(lm2, c_pad, deltas, kmax2, allow_pos_k=False)
                fill_upstr_cached_real(out, cA2, cB2, lmds2, exp2, kmax2, j2toll, N, complex(mode_scale * (-Aj), 0.0) * g20jm)

                kmax3 = max(1, min(N, j3toll - 1))
                cA3, cB3, lmds3, exp3, kmax3 = _cached_tables_for_lam_np(lm3, c_pad, deltas, kmax3, allow_pos_k=False)
                fill_upstr_cached_real(out, cA3, cB3, lmds3, exp3, kmax3, j3toll, N, complex(mode_scale * (-Aj), 0.0) * g30jm)

                kmax4 = max(1, min(N, j4toll - 1))
                cA4, cB4, lmds4, exp4, kmax4 = _cached_tables_for_lam_np(lm4, c_pad, deltas, kmax4, allow_pos_k=True)
                fill_dwstr_cached_real(out, cA4, cB4, lmds4, exp4, kmax4, j4toll, N, complex(mode_scale * Aj, 0.0) * g40jm)
            else:
                fill_upstr_direct_real(out, c_pad, float(deltas), lm1, j1toll, N, complex(mode_scale * (-Aj), 0.0) * g10jm)
                fill_upstr_direct_real(out, c_pad, float(deltas), lm2, j2toll, N, complex(mode_scale * (-Aj), 0.0) * g20jm)
                fill_upstr_direct_real(out, c_pad, float(deltas), lm3, j3toll, N, complex(mode_scale * (-Aj), 0.0) * g30jm)
                fill_dwstr_direct_real(out, c_pad, float(deltas), lm4, j4toll, N, complex(mode_scale * Aj, 0.0) * g40jm)

        else:
            # borderline resonance: fall back to SL0 behavior
            add_mode_sl0(out, jm, Am, lambs, g0s, g1sum, c_pad, s_pad, deltas, use_cached, toll)
            return

        local_coeff = (complex(mode_scale * Aj, 0.0) * g1sum).real
        add_local_real(out, c_pad, N, local_coeff)

    def mode_sl0(
        jm: int,
        Am: np.ndarray,
        lambs: Tuple[complex, complex, complex, complex],
        g0s: Tuple[complex, complex, complex, complex],
        g1sum: complex,
        c_pad: np.ndarray,
        s_pad: np.ndarray,
        deltas: float,
        use_cached: bool,
        toll: float,
    ) -> np.ndarray:
        out = np.zeros(int(len(c_pad)), dtype=np.float64)
        add_mode_sl0(out, jm, Am, lambs, g0s, g1sum, c_pad, s_pad, deltas, use_cached, toll)
        return out

    def mode_sl1(
        jm: int,
        Am: np.ndarray,
        lambs: Tuple[complex, complex, complex, complex],
        g0s: Tuple[complex, complex, complex, complex],
        g1sum: complex,
        c_pad: np.ndarray,
        s_pad: np.ndarray,
        deltas: float,
        use_cached: bool,
        toll: float,
    ) -> np.ndarray:
        out = np.zeros(int(len(c_pad)), dtype=np.float64)
        add_mode_sl1(out, jm, Am, lambs, g0s, g1sum, c_pad, s_pad, deltas, use_cached, toll)
        return out

    # Store the allocation-returning wrappers in the legacy cache.
    _MODE_FN_CACHE[key] = (mode_sl0, mode_sl1)

    # Also store adders in a separate cache for in-place accumulation.
    _MODE_ADDER_CACHE[key] = (add_mode_sl0, add_mode_sl1)

    return _MODE_FN_CACHE[key]




def get_mode_adders(*, parallel: bool = False, fastmath: bool = False):
    """Return (add_mode_sl0, add_mode_sl1) that accumulate into `out` in-place."""
    key = (bool(parallel), bool(fastmath))
    if key in _MODE_ADDER_CACHE:
        return _MODE_ADDER_CACHE[key]

    # Ensure kernels are built and caches are populated.
    get_mode_functions(parallel=key[0], fastmath=key[1])
    return _MODE_ADDER_CACHE[key]


# Backwards-compatible defaults
mode_sl0_numba, mode_sl1_numba = get_mode_functions(parallel=False, fastmath=False)

__all__ = [
    "get_mode_functions",
    "get_mode_adders",
    "mode_sl0_numba",
    "mode_sl1_numba",
]
