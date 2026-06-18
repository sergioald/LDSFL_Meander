
from __future__ import annotations
import warnings
import numpy as np


def dxdy2(ERT: float, U: np.ndarray, x: np.ndarray, y: np.ndarray,
          theta: np.ndarray, deltas: float, Nsold: int, Ns: int, jt: int, cstab: float = 0.01):
    """
    Port of dxdy2.m.

    Adds a small safety guard for cases where the migration velocity becomes
    exactly zero or numerically tiny, which would otherwise produce an infinite
    or excessively large timestep.
    """
    # Allocation-light version (important when called every timestep).
    # Keep the original math, but avoid creating multiple temporary arrays.
    theta = np.asarray(theta, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    ERT = float(ERT)
    cstab = float(cstab)

    if not np.isfinite(ERT):
        raise FloatingPointError(
            f"Non-finite migration speed encountered in dxdy2: ERT={ERT!r} at jt={jt}"
        )
    if not np.all(np.isfinite(U)):
        raise FloatingPointError(
            f"Non-finite migration speed encountered in dxdy2: U contains non-finite values at jt={jt}"
        )
    if not np.all(np.isfinite(theta)):
        raise FloatingPointError(
            f"Non-finite migration speed encountered in dxdy2: theta contains non-finite values at jt={jt}"
        )
    if not np.isfinite(cstab):
        raise FloatingPointError(
            f"Non-finite timestep stability coefficient encountered in dxdy2: cstab={cstab!r} at jt={jt}"
        )

    dxdtl = np.empty_like(U)
    dydtl = np.empty_like(U)
    np.sin(theta, out=dxdtl)
    np.cos(theta, out=dydtl)
    dxdtl *= U
    dydtl *= U
    dxdtl *= ERT
    dydtl *= ERT

    # max(max(dxdtl), max(dydtl)) == max(maximum(dxdtl, dydtl))
    maxCSI = float(max(np.abs(dxdtl).max(), np.abs(dydtl).max()))
    tiny = 1.0e-14

    if not np.isfinite(maxCSI):
        raise FloatingPointError(f"Non-finite migration speed encountered in dxdy2: maxCSI={maxCSI!r}")

    if maxCSI <= tiny:
        warnings.warn(
            f"maxCSI={maxCSI:.3e} is zero or very small at jt={jt}; using fallback dt={cstab:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )
        dt = cstab
    else:
        dt = cstab / maxCSI

    return dxdtl, dydtl, float(dt)


def update_parameters(Cf0_old: float, Cf0: float, wave_l_old: float, wave_l: float,
                      valle_l_old: float, valle_l: float,
                      beta_old: float, ds_old: float, theta0_old: float):
    """
    Port of Update_Parameters.m
    """
    dCf = float(Cf0_old) / float(Cf0)
    dLs = (float(valle_l) / float(wave_l)) / (float(valle_l_old) / float(wave_l_old))

    one_third = 1.0 / 3.0
    beta   = float(beta_old)   * (dLs ** one_third) * (dCf ** one_third)
    theta0 = float(theta0_old) * (dLs ** (2.0*one_third)) * (dCf ** (-one_third))
    ds     = float(ds_old)     * (dLs ** one_third) * (dCf ** one_third)
    return beta, theta0, ds
