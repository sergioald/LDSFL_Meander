
from __future__ import annotations

import numpy as np

# --- Speed helpers (non-Numba path) ---
# In the original MATLAB, SEMIANA1/2 are called *a lot* inside Parall_U_free.
# The most expensive operation per inner-loop iteration is exp(lmds * k).
# The helpers below allow Parall_U_free to precompute exp(lmds*k) for all
# integer k in [-N, N] once per eigenvalue (lam), while keeping the same
# accumulation order as the original MATLAB loops.

def semiana1(js: int, N: int, c: np.ndarray, jsend: int, deltas: float, lm: complex) -> complex:
    """
    Port of SEMIANA1.m (expects 1-based indexing for js/jsend and c).
    """
    lmds = lm*deltas
    lm2ds = lm*lmds
    aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)

    conv = c[js] / lm2ds * (np.exp(-lmds) + lmds - 1.0)

    for j in range(js+1, jsend):
        conv = conv + (c[j] * aux1 / lm2ds) * np.exp(lmds * float(js - j))

    conv = conv - c[jsend] / lm2ds * (1.0 + lmds - np.exp(lmds)) * np.exp(lmds * float(js - jsend))
    return conv

def semiana2(js: int, N: int, c: np.ndarray, jsend: int, deltas: float, lm: complex) -> complex:
    """
    Port of SEMIANA2.m (expects 1-based indexing).
    """
    lmds = lm*deltas
    lm2ds = lm*lmds
    aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)

    conv = c[jsend] / lm2ds * (-1.0 + lmds + np.exp(-lmds)) * np.exp(lmds * float(js - jsend))

    for j in range(jsend+1, js):
        conv = conv + (c[j] * aux1 / lm2ds) * np.exp(lmds * float(js - j))

    conv = conv - c[js] / lm2ds * (1.0 + lmds - np.exp(lmds))
    return conv


def semiana1_cached(js: int, jsend: int,
                    cA: np.ndarray, cB: np.ndarray,
                    lmds: complex,
                    exp_table: np.ndarray, kmax: int) -> complex:
    """
    Cached SEMIANA1 using:
      exp_table[kmax+k] = exp(lmds*float(k)) for k in [-kmax, +kmax]
      cA = (c*aux1)/lm2ds
      cB = c/lm2ds
    """
    # exp(-lmds) is k=-1 => index kmax-1
    conv = cB[js] * (exp_table[kmax - 1] + lmds - 1.0)

    L = jsend - js - 1
    if L > 0:
        # k = js-j = -1, -2, ..., -L  => indices kmax-1, kmax-2, ..., kmax-L
        exps = exp_table[kmax - 1 : kmax - 1 - L : -1]
        terms = cA[js + 1:jsend] * exps
        conv = conv + np.add.accumulate(terms)[-1]

    # exp(+lmds) is k=+1 => index kmax+1
    conv = conv - cB[jsend] * (1.0 + lmds - exp_table[kmax + 1]) * exp_table[kmax + (js - jsend)]
    return conv


def semiana2_cached(js: int, jsend: int,
                    cA: np.ndarray, cB: np.ndarray,
                    lmds: complex,
                    exp_table: np.ndarray, kmax: int) -> complex:
    """
    Cached SEMIANA2 using:
      exp_table[kmax+k] = exp(lmds*float(k)) for k in [-kmax, +kmax]
      cA = (c*aux1)/lm2ds
      cB = c/lm2ds
    """
    # exp(-lmds) is k=-1 => index kmax-1
    conv = cB[jsend] * (-1.0 + lmds + exp_table[kmax - 1]) * exp_table[kmax + (js - jsend)]

    L = js - jsend - 1
    if L > 0:
        # j = jsend+1..js-1 increasing
        # k = js-j = L, L-1, ..., 1  => indices kmax+L, ..., kmax+1
        exps = exp_table[kmax + L : kmax : -1]
        terms = cA[jsend + 1:js] * exps
        conv = conv + np.add.accumulate(terms)[-1]

    # exp(+lmds) is k=+1 => index kmax+1
    conv = conv - cB[js] * (1.0 + lmds - exp_table[kmax + 1])
    return conv


def int_semiana1(c: np.ndarray, lm: complex, deltas: float, js: int, jsend: int) -> complex:
    """
    Port of int_semiana1.m (expects 1-based indexing).
    """
    lmds = lm*deltas
    lm2ds = lm*lmds
    aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)

    s = c[js] / lm2ds * (np.exp(-lmds) + lmds - 1.0)

    if jsend - js > 1:
        j_values = np.arange(js+1, jsend, dtype=np.int64)
        exp_values = np.exp(lmds * (js - j_values).astype(np.float64))
        coeff_values = c[j_values] * aux1 / lm2ds
        s = s + np.sum(coeff_values * exp_values)

    s = s - c[jsend] / lm2ds * (1.0 + lmds - np.exp(lmds)) * np.exp(lmds * float(js - jsend))
    return s

def int_semiana2(c: np.ndarray, lm: complex, deltas: float, js: int, jsend: int) -> complex:
    """
    Port of int_semiana2.m (expects 1-based indexing).
    """
    lmds = lm*deltas
    lm2ds = lm*lmds
    aux1 = -2.0 + np.exp(lmds) + np.exp(-lmds)

    s = c[jsend] / lm2ds * (-1.0 + lmds + np.exp(-lmds)) * np.exp(lmds * float(js - jsend))

    if js - jsend > 1:
        j_values = np.arange(jsend+1, js, dtype=np.int64)
        exp_values = np.exp(lmds * (js - j_values).astype(np.float64))
        coeff_values = c[j_values] * aux1 / lm2ds
        s = s + np.sum(coeff_values * exp_values)

    s = s - c[js] / lm2ds * (1.0 + lmds - np.exp(lmds))
    return s

# Vectorized variants used in the MATLAB parfor branch (optional)
def semiana1_v(js: int, N: int, c: np.ndarray, jsend: int, deltas: float, lm: complex) -> complex:
    return semiana1(js, N, c, jsend, deltas, lm)

def semiana2_v(js: int, N: int, c: np.ndarray, jsend: int, deltas: float, lm: complex) -> complex:
    return semiana2(js, N, c, jsend, deltas, lm)
