"""Validate the periodic flow solver against the free-flow solver.

``docs/validation_strategy.md`` records that the periodic boundary condition
"should be treated as experimental until it has dedicated validation against a
trusted reference". A trusted reference is already in the repository: for a
spatially periodic curvature field, the periodic solution is the far-field
limit of the free solution, so the two must agree in the interior of a
sufficiently long domain.

"Sufficiently long" is the crux. At beta = 9 the slowest streamwise mode decays
as exp(Re(lambda) s) with 1/|Re(lambda)| of order 85 half-widths, so on a
40-unit or 200-unit domain the free solution is still dominated by its inlet
and outlet conditions and the comparison is meaningless. The tests below
therefore check convergence with domain length rather than agreement at a
single length.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldsfl.flowfield import parall_u_free
from ldsfl.flowfield_periodic import parall_u_periodic
from ldsfl.resistance import resistance_function_flagbed

BETA = 9.0
THETA0 = 0.3
DS = 0.005
FLAGBED = 2
RPIC0 = 0.5
MDAT = 4
AMPLITUDE = 1.0e-3


def _solve_both(domain_length: float, n_periods: int):
    """Interior free and periodic responses to the same periodic curvature."""
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(FLAGBED, THETA0, DS, RPIC0)
    npts = int(domain_length) + 1
    s = np.linspace(0.0, domain_length, npts)
    deltas = float(s[1] - s[0])
    # A whole number of periods, so the field is genuinely periodic on the domain.
    curvature = AMPLITUDE * np.sin(2.0 * np.pi * n_periods * s / domain_length)

    common = (cf0, ct, cd, phit, phid, BETA, rpic, THETA0, f0, MDAT, 1, npts, np.array([1.0]), deltas)
    u_free, _ = parall_u_free(curvature, s, *common, SL=0)
    u_periodic, _ = parall_u_periodic(curvature, s, *common)

    interior = slice(int(0.4 * npts), int(0.6 * npts))
    return u_free[interior], u_periodic[interior]


def _amplitude_ratio_and_correlation(domain_length: float, n_periods: int):
    u_free, u_periodic = _solve_both(domain_length, n_periods)
    ratio = float(np.abs(u_periodic).max() / np.abs(u_free).max())
    corr = float(np.corrcoef(u_free, u_periodic)[0, 1])
    return ratio, corr


def test_periodic_matches_free_solution_on_a_long_domain():
    """800 half-widths is roughly nine influence lengths at beta = 9."""
    ratio, corr = _amplitude_ratio_and_correlation(800.0, 40)
    assert corr > 0.995, f"interior shapes disagree (corr={corr:.5f})"
    assert abs(ratio - 1.0) < 0.02, f"interior amplitudes disagree (ratio={ratio:.4f})"


def test_agreement_improves_with_domain_length():
    """The free solution approaches the periodic one as boundaries recede."""
    short_ratio, _ = _amplitude_ratio_and_correlation(200.0, 10)
    long_ratio, _ = _amplitude_ratio_and_correlation(800.0, 40)
    assert abs(long_ratio - 1.0) < abs(short_ratio - 1.0), (
        f"agreement did not improve with domain length "
        f"(short={short_ratio:.4f}, long={long_ratio:.4f})"
    )


@pytest.mark.parametrize("domain_length, n_periods", [(400.0, 20), (800.0, 40)])
def test_both_solvers_agree_in_shape_on_adequate_domains(domain_length, n_periods):
    _ratio, corr = _amplitude_ratio_and_correlation(domain_length, n_periods)
    assert corr > 0.99


def test_periodic_solver_returns_zero_for_zero_curvature():
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(FLAGBED, THETA0, DS, RPIC0)
    npts = 201
    s = np.linspace(0.0, 200.0, npts)
    deltas = float(s[1] - s[0])
    u, _flag = parall_u_periodic(
        np.zeros(npts), s, cf0, ct, cd, phit, phid, BETA, rpic, THETA0, f0,
        MDAT, 1, npts, np.array([1.0]), deltas,
    )
    assert np.allclose(u, 0.0, atol=1.0e-12)
