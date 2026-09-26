"""Numerical and cache checks for the accelerated vertical coefficient path."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ldsfl.vertical import k0123

numba_available = importlib.util.find_spec("numba") is not None
requires_numba = pytest.mark.skipif(not numba_available, reason="numba optional extra is not installed")


@requires_numba
@pytest.mark.parametrize("cf0", [0.003, 0.01318086081547672, 0.04])
def test_numba_k0123_matches_python_reference_tightly(cf0):
    from ldsfl.vertical_numba import clear_k0123_cache, k0123_numba_cached

    clear_k0123_cache()
    reference = k0123(cf0)
    accelerated = k0123_numba_cached(cf0)

    for index in range(4):
        np.testing.assert_allclose(accelerated[index], reference[index], rtol=2e-11, atol=5e-14)
    for index in range(4, 8):
        np.testing.assert_allclose(accelerated[index], reference[index], rtol=3e-11, atol=2e-13)
    np.testing.assert_allclose(accelerated[8], reference[8], rtol=5e-15, atol=0.0)


@requires_numba
def test_numba_k0123_cache_is_exact_bounded_and_returns_safe_array_copies():
    from ldsfl.vertical_numba import (
        clear_k0123_cache,
        k0123_cache_info,
        k0123_numba_cached,
    )

    cf0 = 0.01318086081547672
    nearby = cf0 + 1.0e-12
    clear_k0123_cache()
    first = k0123_numba_cached(cf0)
    first[4][0] = 12345.0
    same = k0123_numba_cached(cf0)
    assert same[4][0] != 12345.0
    assert k0123_cache_info().hits == 1

    other = k0123_numba_cached(nearby)
    assert k0123_cache_info().misses == 2
    assert other[8] != same[8]
    assert np.max(np.abs(other[4] - same[4])) > 0.0


@requires_numba
def test_numba_k0123_cache_stays_bounded_after_more_than_16_exact_keys():
    from ldsfl.vertical_numba import clear_k0123_cache, k0123_cache_info, k0123_numba_cached

    cf_values = [0.01318086081547672 + i * 1e-10 for i in range(20)]
    assert len(set(cf_values)) == 20
    clear_k0123_cache()
    for cf0 in cf_values:
        k0123_numba_cached(cf0)

    info = k0123_cache_info()
    assert info.maxsize == 16
    assert info.currsize == 16
    assert info.misses == 20


@requires_numba
def test_numba_fgj_rejects_unsupported_integral_selector():
    from ldsfl.vertical_numba import _fgj_nb

    with pytest.raises(ValueError, match="Unsupported ig value"):
        _fgj_nb(0.1, 0.01, 0.0, 1.0, 3, 0.01, 0.41, 1.84, -1.56, 0.001, 0.0)


def test_reference_coefficients_reuse_only_exact_cf0_during_resonance_search():
    from ldsfl.resonance import fundamental_decay_rate
    from ldsfl.vertical import (
        clear_k0123_reference_cache,
        k0123_reference_cache_info,
    )

    clear_k0123_reference_cache()
    fundamental_decay_rate(6.0, 0.3, 0.005, 0.5)
    fundamental_decay_rate(12.0, 0.3, 0.005, 0.5)
    info = k0123_reference_cache_info()
    assert info.misses == 1
    assert info.hits == 1
    fundamental_decay_rate(6.0, 0.3, 0.005 + 1.0e-12, 0.5)
    assert k0123_reference_cache_info().misses == 2


@requires_numba
@pytest.mark.parametrize("beta, expected_flag", [(6.0, 1), (12.0, -1)])
def test_numba_vertical_integration_preserves_flowfield_equivalence(beta, expected_flag):
    from ldsfl.flowfield import parall_u_free
    from ldsfl.resistance import resistance_function_flagbed

    theta0, ds, rpic0 = 0.3, 0.005, 0.5
    rpic, cf0, ct, cd, phi_t, phi_d, f0 = resistance_function_flagbed(2, theta0, ds, rpic0)
    s = np.linspace(0.0, 10.0, 51, dtype=np.float64)
    c = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    common = (c, s, cf0, ct, cd, phi_t, phi_d, beta, rpic, theta0, f0, 3, 1, len(s), np.array([1.0]), s[1] - s[0])
    reference, ref_flag = parall_u_free(*common, SL=1, backend="numpy")
    accelerated, accelerated_flag = parall_u_free(*common, SL=1, backend="numba", numba_fastmath=False)
    assert ref_flag == accelerated_flag == expected_flag
    np.testing.assert_allclose(accelerated, reference, rtol=1e-9, atol=1e-11)


@requires_numba
@pytest.mark.parametrize("beta", [6.0, 12.0])
def test_numba_vertical_backend_matches_with_numpy_flow_response(beta):
    from ldsfl.flowfield import parall_u_free
    from ldsfl.resistance import resistance_function_flagbed
    from ldsfl.vertical import clear_k0123_reference_cache
    from ldsfl.vertical_numba import clear_k0123_cache

    theta0, ds, rpic0 = 0.3, 0.005, 0.5
    rpic, cf0, ct, cd, phi_t, phi_d, f0 = resistance_function_flagbed(2, theta0, ds, rpic0)
    s = np.linspace(0.0, 10.0, 51, dtype=np.float64)
    c = 1.0e-3 * np.sin(2.0 * np.pi * s / s[-1])
    common = (c, s, cf0, ct, cd, phi_t, phi_d, beta, rpic, theta0, f0, 3, 1, len(s), np.array([1.0]), s[1] - s[0])

    clear_k0123_reference_cache()
    clear_k0123_cache()
    reference, ref_flag = parall_u_free(
        *common, SL=1, backend="numpy", vertical_backend="numpy"
    )
    accelerated, accelerated_flag = parall_u_free(
        *common, SL=1, backend="numpy", vertical_backend="numba"
    )

    assert ref_flag == accelerated_flag
    np.testing.assert_allclose(accelerated, reference, rtol=1e-9, atol=1e-11)
