"""The resonance flag must track the fundamental mode, not mode 6.

``_compute_flag`` previously read ``lamb2[5]``. ``Re(lambda2)`` for mode 6 is
roughly -5 to -13 across realistic aspect ratios, so the flag reported
"sub-resonant" unconditionally. The fundamental mode is the one that changes
sign at the resonant aspect ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldsfl.flowfield import _compute_flag, _precompute_modes
from ldsfl.flowfield_periodic import _compute_flag as _compute_periodic_flag
from ldsfl.resistance import resistance_function_flagbed
from ldsfl.resonance import (
    fundamental_decay_rate,
    resonance_report,
    resonant_aspect_ratio,
)

THETA0 = 0.3
DS = 0.005
RPIC0 = 0.5


def _lamb2(beta: float, mdat: int = 6):
    rpic, cf0, ct, cd, phit, phid, f0 = resistance_function_flagbed(2, THETA0, DS, RPIC0)
    return _precompute_modes(cf0, ct, cd, phit, phid, beta, rpic, THETA0, f0, mdat)[2]


def test_free_and_periodic_flags_change_sign_across_the_resonant_aspect_ratio():
    beta_r = resonant_aspect_ratio(THETA0, DS, RPIC0)
    assert beta_r is not None

    below = _lamb2(beta_r * 0.9)
    above = _lamb2(beta_r * 1.1)

    assert _compute_flag(below) == 1, "expected sub-resonant below beta_R"
    assert _compute_flag(above) == -1, "expected super-resonant above beta_R"
    assert _compute_periodic_flag(below) == 1, "periodic flag should use the same convention"
    assert _compute_periodic_flag(above) == -1, "periodic flag should use the same convention"


def test_flag_is_not_constant_over_the_realistic_aspect_ratio_range():
    """The previous implementation returned +1 for every one of these."""
    flags = {_compute_flag(_lamb2(beta)) for beta in (4.0, 6.0, 9.0, 12.0, 20.0, 40.0)}
    assert flags == {1, -1}


def test_mode_six_decay_rate_is_uninformative():
    """Documents why indexing mode 6 could never work."""
    for beta in (4.0, 9.0, 20.0, 40.0):
        assert _lamb2(beta)[5].real < -1.0


def test_resonant_aspect_ratio_matches_the_sign_change():
    beta_r = resonant_aspect_ratio(THETA0, DS, RPIC0)
    assert beta_r is not None
    assert fundamental_decay_rate(beta_r * 0.99, THETA0, DS, RPIC0) < 0.0
    assert fundamental_decay_rate(beta_r * 1.01, THETA0, DS, RPIC0) > 0.0


def test_resonant_aspect_ratio_moves_with_grain_size():
    coarse = resonant_aspect_ratio(THETA0, 0.02, RPIC0)
    fine = resonant_aspect_ratio(THETA0, 0.002, RPIC0)
    assert coarse is not None and fine is not None
    assert fine > coarse, "beta_R should increase as relative grain size decreases"


@pytest.mark.parametrize(
    "beta, expected",
    [
        (6.0, "sub-resonant"),
        (9.0, "sub-resonant"),
        (12.0, "super-resonant"),
        (20.0, "super-resonant"),
    ],
)
def test_resonance_report_states(beta, expected):
    assert resonance_report(beta, THETA0, DS, RPIC0)["state"] == expected


def test_influence_length_grows_towards_resonance():
    """A domain must be long compared with 1/|Re(lambda)|, which diverges at beta_R."""
    beta_r = resonant_aspect_ratio(THETA0, DS, RPIC0)
    assert beta_r is not None
    far = resonance_report(6.0, THETA0, DS, RPIC0)["influence_length_half_widths"]
    near = resonance_report(beta_r * 0.999, THETA0, DS, RPIC0)["influence_length_half_widths"]
    assert np.isfinite(far)
    assert near > 10.0 * far


def test_resonance_report_is_included_in_run_summary(tmp_path):
    from ldsfl.main import run_case

    input_dir = tmp_path / "Input"
    input_dir.mkdir()
    (input_dir / "Parameter.csv").write_text(
        "Id,Beta,ds,Thetha,flagbed,r,Mdat,flagbed=1 plane; flagbed=2 dunes\n"
        "1,9.0,0.005,0.3,2,0.5,6,2\n",
        encoding="utf-8",
    )
    (input_dir / "xy.csv").write_text(
        "0,0\n1,0.01\n2,0.03\n3,0.06\n4,0.08\n5,0.09\n",
        encoding="utf-8",
    )

    result = run_case(tmp_path, 1, max_steps=2, Nprint=2, do_plots=False)

    report = result["resonance"]
    assert report["state"] in {"sub-resonant", "near-resonant", "super-resonant", "resonant"}
    assert report["flag"] in {1, -1}
    assert report["resonant_beta"] is None or report["resonant_beta"] > 0.0
