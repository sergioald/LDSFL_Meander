"""CLI tests for optional final equivalence-stability diagnostics."""

from __future__ import annotations

import sys

import run_ldsfl


def test_cli_forwards_return_equivalence_stability(monkeypatch, tmp_path):
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
            "--return-equivalence-stability",
            "1",
        ],
    )

    run_ldsfl.main()

    assert captured["return_equivalence_stability"] is True
