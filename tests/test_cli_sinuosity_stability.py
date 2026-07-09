"""CLI forwarding tests for sinuosity-stability options."""

from __future__ import annotations

import sys

import run_ldsfl


def test_cli_forwards_sinuosity_stability_options(monkeypatch, tmp_path):
    captured = {}

    def fake_run_project(base_dir, **kwargs):
        captured["base_dir"] = base_dir
        captured.update(kwargs)
        return [{"id_files": "fake"}]

    monkeypatch.setattr(run_ldsfl, "run_project", fake_run_project)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ldsfl.py",
            "--base-dir",
            str(tmp_path),
            "--cases",
            "1",
            "--max-steps",
            "5",
            "--stop-on-sinuosity-stability",
            "1",
            "--sinuo-window",
            "7",
            "--sinuo-rel-tol",
            "0.001",
            "--sinuo-equiv-transient-step",
            "-1",
            "--sinuo-equiv-drift-tol",
            "0.05",
            "--sinuo-equiv-confidence",
            "0.8",
            "--sinuo-equiv-min-points",
            "4",
            "--sinuo-equiv-hac-lags",
            "0",
            "--sinuo-stability-interval",
            "3",
        ],
    )

    run_ldsfl.main()

    assert captured["stop_on_sinuosity_stability"] is True
    assert captured["sinuo_window"] == 7
    assert captured["sinuo_rel_tol"] == 0.001
    assert captured["sinuo_equiv_transient_step"] is None
    assert captured["sinuo_equiv_drift_tol"] == 0.05
    assert captured["sinuo_equiv_confidence"] == 0.8
    assert captured["sinuo_equiv_min_points"] == 4
    assert captured["sinuo_equiv_hac_lags"] == 0
    assert captured["sinuo_stability_interval"] == 3
