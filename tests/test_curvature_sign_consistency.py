"""Regression test for the initial-curvature sign convention."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ldsfl.geometry import geometry4
from ldsfl.main import initial_curvature
from ldsfl.profile import preprof_3


def test_first_step_curvature_matches_geometry4_convention():
    t = np.linspace(0.0, 60.0, 601)
    x = t.copy()
    y = 2.0 * np.sin(0.5 * t)

    s, xa, ya, th, *_ = preprof_3(x, y, 1.0)
    c_first = initial_curvature(th)

    with tempfile.TemporaryDirectory() as tmp:
        c_geom, s2, *_ = geometry4(
            x.copy(),
            y.copy(),
            jt=1,
            dsliminicial=1.0,
            id_files="t",
            Ntstep=10,
            cut_cnt=0,
            beta=9.0,
            base_out=Path(tmp),
            neck_cutoff_interval=0,
            smoothing_enabled=False,
            do_plots=False,
        )

    c_interp = np.interp(s2, s, c_first)
    inner = slice(5, -5)
    corr = np.corrcoef(c_interp[inner], c_geom[inner])[0, 1]
    assert corr > 0.99, (
        f"first-step curvature convention disagrees with geometry4 (corr={corr:.3f}); "
        "check the sign of theta in run_case's initial curvature."
    )


@pytest.mark.parametrize("spacing", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_curvature_magnitude_is_independent_of_sampling_resolution(spacing):
    """A circular arc of radius R has curvature 1/R at any sampling resolution.

    The correlation assertion above is scale invariant and therefore cannot
    detect an amplitude error. This test pins the magnitude against an exact
    analytic value at several resolutions: ``matlab_gradient`` differentiates
    with respect to the point index, so without dividing by the arclength
    spacing ``geometry4`` returns ``spacing / R`` and the flow response is
    silently rescaled by the grid resolution.
    """
    radius = 50.0
    n_points = 400
    arclength = np.arange(n_points, dtype=np.float64) * spacing
    phi = arclength / radius
    x = radius * np.sin(phi)
    y = radius * (1.0 - np.cos(phi))

    with tempfile.TemporaryDirectory() as tmp:
        c_geom, *_ = geometry4(
            x.copy(),
            y.copy(),
            jt=1,
            dsliminicial=spacing,
            id_files="t",
            Ntstep=10,
            cut_cnt=0,
            beta=9.0,
            base_out=Path(tmp),
            neck_cutoff_interval=0,
            smoothing_enabled=False,
            do_plots=False,
        )

    interior = np.abs(c_geom[50:-50])
    assert np.allclose(interior, 1.0 / radius, rtol=1.0e-3), (
        f"curvature at spacing={spacing} is {interior.mean():.6f}, "
        f"expected {1.0 / radius:.6f} (ratio {interior.mean() * radius:.4f})"
    )
