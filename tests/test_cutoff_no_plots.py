"""Cutoff CSVs are still written with plotting disabled, but figures are not."""

from __future__ import annotations

import numpy as np

import ldsfl.geometry as geometry


def test_geometry4_does_not_plot_cutoff_when_plots_disabled(monkeypatch, tmp_path):
    calls = {"find": 0, "save_cut": 0}

    def fake_find_neck_cutoff_kdtree_with_refine(*args, **kwargs):
        calls["find"] += 1
        if calls["find"] == 1:
            return 0, 5
        return None

    def fake_plot_cut(*args, **kwargs):
        raise AssertionError("plot_cut should not be called when do_plots=False")

    def fake_save_xy_cut(*args, **kwargs):
        calls["save_cut"] += 1

    monkeypatch.setattr(geometry, "find_neck_cutoff_kdtree_with_refine", fake_find_neck_cutoff_kdtree_with_refine)
    monkeypatch.setattr(geometry, "plot_cut", fake_plot_cut)
    monkeypatch.setattr(geometry, "save_xy_cut", fake_save_xy_cut)

    x = np.linspace(0.0, 9.0, 10)
    y = np.zeros_like(x)

    geometry.geometry4(
        x,
        y,
        jt=3,
        dsliminicial=1.0,
        id_files="cutoff-test",
        Ntstep=10,
        cut_cnt=0,
        beta=1.0,
        base_out=tmp_path,
        neck_cutoff_interval=1,
        smoothing_enabled=False,
        do_plots=False,
    )

    assert calls["save_cut"] == 1
