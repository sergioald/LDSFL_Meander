"""Resonance diagnostics for the linearised bend theory.

The reduced model has a resonant aspect ratio ``beta_R`` at which the
fundamental mode's streamwise decay rate changes sign. Below it the flow
response is downstream-dominated (sub-resonant); above it, upstream-dominated
(super-resonant). Which side a run sits on controls the qualitative behaviour
of bend migration, so it is worth reporting rather than leaving implicit.

``beta_R`` depends on the other physical inputs. For flagbed = 2, r = 0.5:

    theta0 = 0.20            beta_R ~  8.97
    theta0 = 0.30            beta_R ~  9.68
    theta0 = 0.50            beta_R ~  9.50
    ds     = 0.002           beta_R ~ 11.34
    ds     = 0.005           beta_R ~  9.68
    ds     = 0.020           beta_R ~  7.19
"""

from __future__ import annotations

import numpy as np

from .flowfield import _precompute_modes
from .resistance import resistance_function_flagbed

#: Relative distance from beta_R within which a run is reported as near-resonant.
NEAR_RESONANCE_BAND = 0.02


def fundamental_decay_rate(
    beta: float,
    theta0: float,
    ds: float,
    rpic_0: float,
    flagbed: int = 2,
    Mdat: int = 6,
) -> float:
    """Return ``Re(lambda2)`` for the fundamental lateral mode.

    Negative values are sub-resonant; positive values are super-resonant.
    """
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(
        int(flagbed), float(theta0), float(ds), float(rpic_0)
    )
    result = _precompute_modes(
        cf0,
        ct,
        cd,
        phit,
        phid,
        float(beta),
        rpic,
        float(theta0),
        f0,
        int(Mdat),
    )
    lamb2 = result[2]
    if len(lamb2) == 0:
        return float("nan")
    return float(np.real(lamb2[0]))


def resonant_aspect_ratio(
    theta0: float,
    ds: float,
    rpic_0: float,
    flagbed: int = 2,
    Mdat: int = 6,
    bracket: tuple[float, float] = (2.0, 200.0),
    tolerance: float = 1.0e-6,
) -> float | None:
    """Bisect for the ``beta`` at which the fundamental decay rate vanishes.

    Returns ``None`` when the bracket does not contain a sign change.
    """
    lo, hi = float(bracket[0]), float(bracket[1])
    args = (theta0, ds, rpic_0, flagbed, Mdat)
    f_lo = fundamental_decay_rate(lo, *args)
    f_hi = fundamental_decay_rate(hi, *args)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0:
        return None

    while (hi - lo) > tolerance * max(1.0, lo):
        mid = 0.5 * (lo + hi)
        f_mid = fundamental_decay_rate(mid, *args)
        if not np.isfinite(f_mid):
            return None
        if f_mid * f_lo > 0.0:
            lo = mid
            f_lo = f_mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def resonance_report(
    beta: float,
    theta0: float,
    ds: float,
    rpic_0: float,
    flagbed: int = 2,
    Mdat: int = 6,
) -> dict:
    """Summarise where a run sits relative to resonance."""
    beta = float(beta)
    args = (theta0, ds, rpic_0, flagbed, Mdat)
    decay = fundamental_decay_rate(beta, *args)
    beta_r = resonant_aspect_ratio(*args)

    if decay < 0.0:
        state = "sub-resonant"
    elif decay > 0.0:
        state = "super-resonant"
    else:
        state = "resonant"

    distance = None
    if beta_r is not None and beta_r > 0.0:
        distance = beta / beta_r - 1.0
        if abs(distance) <= NEAR_RESONANCE_BAND:
            state = "near-resonant"

    if decay == 0.0:
        influence_length = float("inf")
    elif np.isfinite(decay):
        influence_length = 1.0 / abs(decay)
    else:
        influence_length = float("nan")

    return {
        "state": state,
        "flag": 1 if decay < 0.0 else -1,
        "beta": beta,
        "resonant_beta": beta_r,
        "relative_distance_to_resonance": distance,
        "fundamental_decay_rate": decay,
        "influence_length_half_widths": float(influence_length),
    }
