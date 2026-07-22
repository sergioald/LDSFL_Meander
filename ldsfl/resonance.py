"""Resonance diagnostics for the linearised bend theory.

The reduced model has a resonant aspect ratio ``beta_R`` at which the
fundamental mode's streamwise decay rate changes sign. Below it the flow
response is downstream-dominated (sub-resonant); above it, upstream-dominated
(super-resonant). Which side a run sits on controls the qualitative behaviour
of bend migration, so it is worth reporting rather than leaving implicit.

The beta bisection is intentionally implemented with a single-mode helper that
reuses the fixed resistance/vertical coefficients. Calling the full
``_precompute_modes`` path inside every bisection step would repeatedly run the
expensive ``k0123`` vertical integration even though ``Cf0`` is fixed for the
diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flowfield import _roots_companion_matlab, _sort_roots_like_matlab_swaps
from .resistance import resistance_function_flagbed
from .vertical import k0123

#: Relative distance from beta_R within which a run is reported as near-resonant.
NEAR_RESONANCE_BAND = 0.02


@dataclass(frozen=True)
class _ResonanceBase:
    rpic: float
    cf0: float
    ct: float
    cd: float
    phit: float
    phid: float
    f0: float
    k0: float
    k1: float
    k2: float
    k3: float


def _resonance_base(theta0: float, ds: float, rpic_0: float, flagbed: int) -> _ResonanceBase:
    """Compute beta-independent coefficients once for a resonance diagnostic."""
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(
        int(flagbed), float(theta0), float(ds), float(rpic_0)
    )
    k0, k1, k2, k3, *_ = k0123(cf0)
    return _ResonanceBase(
        rpic=float(rpic),
        cf0=float(cf0),
        ct=float(ct),
        cd=float(cd),
        phit=float(phit),
        phid=float(phid),
        f0=float(f0),
        k0=float(k0),
        k1=float(k1),
        k2=float(k2),
        k3=float(k3),
    )


def _fundamental_decay_rate_from_base(beta: float, theta0: float, base: _ResonanceBase) -> float:
    """Return ``Re(lambda2)`` for mode 1 using precomputed vertical coefficients.

    This mirrors the first-mode algebra in ``flowfield._precompute_modes`` but
    avoids recomputing ``k0123(Cf0)`` and avoids building unused higher modes.
    """
    beta = float(beta)
    theta0 = float(theta0)
    cf0 = base.cf0
    ct = base.ct
    cd = base.cd
    phit = base.phit
    phid = base.phid
    rpic = base.rpic
    f0 = base.f0
    k0 = base.k0
    k1 = base.k1
    k2 = base.k2
    k3 = base.k3

    s1 = 2.0 / (1.0 - ct)
    s2 = cd / (1.0 - ct)
    f1 = 2.0 * phit / (1.0 - ct)
    f2 = phid + cd * phit / (1.0 - ct)

    a1 = beta * cf0 * s1
    a2 = beta * cf0 * (s2 - 1.0)
    a3 = beta * cf0
    a4 = f1
    a5 = f2
    a6 = rpic / (beta * (theta0**0.5))

    b1 = -beta * cf0
    b2 = 1.0 - (cf0**0.5) * k2
    b3 = -k0 / (beta * (cf0**0.5)) - k3 / beta
    b4 = -k1 / (cf0 * (beta**2))
    b5 = k2 * (theta0**0.5) / (rpic * (cf0**0.5))
    b6 = k3 * (theta0**0.5) / (beta * cf0 * rpic)

    h1bar = b2
    h2bar = b3
    h3bar = b4
    d1bar = (f0**2) * h1bar - b5
    d2bar = (f0**2) * h2bar - b6
    d3bar = (f0**2) * h3bar

    alf0 = a2
    alf1 = 1.0 / (f0**2)

    bet2 = a1
    bet3 = 1.0

    gam2 = b1 - a2 * d1bar
    gam3 = -h1bar - a2 * d2bar

    del1 = a5 - 1.0 - (f0**2) * a3 * a6
    del2 = -(f0**2) * a6

    eps3 = a4 - 1.0 - (f0**2) * a3 * a6
    eps4 = del2

    et3 = -del1 * d1bar
    et4 = -del1 * d2bar + (f0**2) * a6 * d1bar
    et5 = -del1 * d3bar + (f0**2) * a6 * d2bar

    # Fundamental lateral mode: jm = 1, so M = pi / 2.
    M = np.pi / 2.0
    alf2 = (1.0 - a5) / ((M**2) * (f0**2) * a6)
    bet4 = (1.0 - a4) / ((M**2) * (f0**2) * a6)

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

    csi1 = -Delta * Delta1 * T1
    csi2 = Delta * (-Delta1 * T2 + del2 * T1 + eps3)
    csi3 = Delta * (-Delta1 * T3 + del2 * T2 + eps4)
    csi4 = Delta * del2 * T3

    sigma0 = (Delta0 * csi1 + Delta * Delta2 * T1) / csi4
    sigma1 = (csi1 + Delta0 * csi2 + Delta * Delta2 * T2) / csi4
    sigma2 = (csi2 + Delta0 * csi3 + Delta * Delta2 * T3) / csi4
    sigma3 = (csi3 + Delta0 * csi4) / csi4
    sigma4 = 1.0

    sigm = np.array([sigma4, sigma3, sigma2, sigma1, sigma0], dtype=np.float64)
    roots4 = _roots_companion_matlab(sigm)
    roots4 = _sort_roots_like_matlab_swaps(roots4, tol=1.0e-10)
    return float(np.real(roots4[1]))


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
    ``Mdat`` is accepted for API compatibility; only the fundamental mode is
    needed for this diagnostic.
    """
    _ = Mdat
    base = _resonance_base(theta0, ds, rpic_0, flagbed)
    return _fundamental_decay_rate_from_base(beta, theta0, base)


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
    _ = Mdat
    lo, hi = float(bracket[0]), float(bracket[1])
    base = _resonance_base(theta0, ds, rpic_0, flagbed)
    f_lo = _fundamental_decay_rate_from_base(lo, theta0, base)
    f_hi = _fundamental_decay_rate_from_base(hi, theta0, base)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0:
        return None

    while (hi - lo) > tolerance * max(1.0, lo):
        mid = 0.5 * (lo + hi)
        f_mid = _fundamental_decay_rate_from_base(mid, theta0, base)
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
    _ = Mdat
    base = _resonance_base(theta0, ds, rpic_0, flagbed)
    decay = _fundamental_decay_rate_from_base(beta, theta0, base)

    lo, hi = 2.0, 200.0
    f_lo = _fundamental_decay_rate_from_base(lo, theta0, base)
    f_hi = _fundamental_decay_rate_from_base(hi, theta0, base)
    beta_r = None
    if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0.0:
        while (hi - lo) > 1.0e-6 * max(1.0, lo):
            mid = 0.5 * (lo + hi)
            f_mid = _fundamental_decay_rate_from_base(mid, theta0, base)
            if not np.isfinite(f_mid):
                beta_r = None
                break
            if f_mid * f_lo > 0.0:
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
        else:
            beta_r = 0.5 * (lo + hi)

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
