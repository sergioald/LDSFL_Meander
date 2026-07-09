"""Regression test for the initial-curvature sign convention."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

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
