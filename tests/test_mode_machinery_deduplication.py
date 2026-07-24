from __future__ import annotations

import numpy as np

from ldsfl import flowfield, flowfield_periodic, modes
from ldsfl.resistance import resistance_function_flagbed


def test_free_and_periodic_flow_paths_share_mode_precomputation():
    assert flowfield._precompute_modes is modes._precompute_modes
    assert flowfield_periodic._precompute_modes is modes._precompute_modes


def test_free_and_periodic_flow_paths_share_resonance_flag():
    assert flowfield._compute_flag is modes._compute_flag
    assert flowfield_periodic._compute_flag is modes._compute_flag


def test_shared_mode_precomputation_is_numerically_stable_for_case_one_parameters():
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(2, 0.3, 0.005, 0.5)

    result = modes._precompute_modes(
        cf0,
        ct,
        cd,
        phit,
        phid,
        9.0,
        rpic,
        0.3,
        f0,
        6,
    )

    assert len(result) == 13
    for array in result:
        assert array.shape == (6,)
        assert np.all(np.isfinite(array.real))
        assert np.all(np.isfinite(array.imag))

    lamb2 = result[2]
    assert modes._compute_flag(lamb2) in {1, -1}


def test_shared_root_ordering_matches_expected_sign_convention():
    # Polynomial with two positive and two negative real roots.
    roots = modes._roots_companion_matlab(np.array([1.0, 0.0, -5.0, 0.0, 4.0]))
    ordered = modes._sort_roots_like_matlab_swaps(roots)

    assert ordered.shape == (4,)
    assert ordered[0].real > 0.0
    assert ordered[3].real < 0.0
