"""Transfer-function tests for the linearised free-flow response.

Why this file exists
--------------------
The existing physics tests (zero curvature -> zero velocity, linearity in
curvature amplitude, serial/threaded agreement) are all invariant under a
rescaling of the linear-theory coefficients. Multiplying every ``g10``-``g41``
returned by ``_precompute_modes`` by 7 leaves all three passing, so nothing in
the suite constrains the *magnitude* of the flow response.

A sinusoidal curvature ``c(s) = a cos(ks)`` is the natural probe: the solver is
a linear, translation-invariant operator, so the interior response must be
``|H(k)| a cos(ks + arg H(k))`` for a transfer function ``H`` that depends only
on the physical parameters and the wavenumber - not on the grid.

The assertions come in three tiers:

1. Structural properties that follow from linearity alone and need no reference
   values: spectral purity, amplitude linearity, resolution independence.
2. A physical signature: ``|H(k)|`` must show a sharp interior maximum, the
   Zolezzi-Seminara resonance.
3. Pinned reference values for ``H(k)``. These lock in the present behaviour at
   a physically meaningful observable. They are *not* an independent derivation
   from the papers, so a disagreement means "the flow response changed" and
   should be reviewed deliberately, not silently regenerated.

The domain is long on purpose. At beta = 9 the slowest mode decays over roughly
1/|Re lambda| ~ 85 half-widths, so a short domain is dominated by the inlet and
outlet conditions and the response is not the far-field one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldsfl.flowfield import parall_u_free
from ldsfl.resistance import resistance_function_flagbed

# Reference configuration. Matches the bundled example's physical parameters.
BETA = 9.0
THETA0 = 0.3
DS = 0.005
FLAGBED = 2
RPIC0 = 0.5
MDAT = 4
DOMAIN = 800.0
NPTS = 1601
AMPLITUDE = 1.0e-4

# (wavenumber, |H|, arg H) measured on the reference configuration.
GOLDEN_TRANSFER = [
    (0.05, 5.406035, -0.119663),
    (0.10, 6.071546, -0.234070),
    (0.15, 7.550981, -0.341759),
    (0.20, 11.148995, -0.457369),
    (0.30, 36.264998, -2.961787),
]


def _transfer(k, *, beta=BETA, npts=NPTS, domain=DOMAIN, amplitude=AMPLITUDE, mdat=MDAT):
    """Complex response amplitude and single-harmonic fit quality."""
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(FLAGBED, THETA0, DS, RPIC0)
    s = np.linspace(0.0, domain, npts)
    deltas = float(s[1] - s[0])
    curvature = amplitude * np.cos(k * s)

    u, _flag = parall_u_free(
        curvature, s, cf0, ct, cd, phit, phid, beta, rpic, THETA0, f0,
        mdat, 1, npts, np.array([1.0]), deltas, SL=0,
    )

    # Measure in the deep interior, away from the boundary-influence zone.
    interior = slice(int(0.35 * npts), int(0.65 * npts))
    s_in = s[interior]
    u_in = u[interior]

    basis = np.column_stack((np.cos(k * s_in), np.sin(k * s_in)))
    coeffs, *_ = np.linalg.lstsq(basis, u_in, rcond=None)
    residual = u_in - basis @ coeffs
    purity = 1.0 - float(np.sum(residual**2) / np.sum(u_in**2))
    return (coeffs[0] - 1j * coeffs[1]) / amplitude, purity


@pytest.mark.parametrize("k", [0.05, 0.15, 0.30])
def test_sinusoidal_curvature_produces_a_single_harmonic_response(k):
    """A linear translation-invariant operator cannot create new wavenumbers."""
    _h, purity = _transfer(k)
    assert purity > 0.999, f"interior response is not a clean harmonic (R^2={purity:.6f})"


@pytest.mark.parametrize("k, expected_gain, expected_phase", GOLDEN_TRANSFER)
def test_transfer_function_matches_reference_values(k, expected_gain, expected_phase):
    """Pins the response magnitude, which no other test constrains."""
    measured, _purity = _transfer(k)
    expected = expected_gain * np.exp(1j * expected_phase)
    assert abs(measured - expected) <= 1.0e-3 * expected_gain, (
        f"H({k}) = {abs(measured):.6f} exp({np.angle(measured):+.6f}j), "
        f"expected {expected_gain:.6f} exp({expected_phase:+.6f}j)"
    )


def test_transfer_function_is_independent_of_grid_resolution():
    """H(k) is a property of the physics, not of the discretisation."""
    coarse, _ = _transfer(0.15, npts=801)
    medium, _ = _transfer(0.15, npts=1601)
    fine, _ = _transfer(0.15, npts=3201)
    assert abs(coarse - fine) <= 5.0e-3 * abs(fine)
    assert abs(medium - fine) <= 5.0e-3 * abs(fine)


@pytest.mark.parametrize("amplitude", [1.0e-5, 1.0e-4, 1.0e-3])
def test_transfer_function_is_independent_of_forcing_amplitude(amplitude):
    """The solver is linear, so the gain must not depend on the input size."""
    reference, _ = _transfer(0.15, amplitude=1.0e-4)
    measured, _ = _transfer(0.15, amplitude=amplitude)
    assert abs(measured - reference) <= 1.0e-6 * abs(reference)


def test_response_shows_a_sharp_resonant_peak():
    """Zolezzi-Seminara resonance: the gain peaks at an intermediate wavenumber.

    A monotone or flat gain curve would mean the resonant mechanism the model
    exists to represent has been lost.
    """
    wavenumbers = np.arange(0.10, 0.55, 0.05)
    gains = np.array([abs(_transfer(float(k))[0]) for k in wavenumbers])

    peak = int(np.argmax(gains))
    assert 0 < peak < len(gains) - 1, "gain peak is at a domain edge, not an interior resonance"
    assert gains[peak] > 3.0 * gains[0]
    assert gains[peak] > 3.0 * gains[-1]
