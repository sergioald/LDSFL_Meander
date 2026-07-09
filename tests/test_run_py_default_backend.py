"""Run.py should use the always-available NumPy backend by default."""

from __future__ import annotations

from pathlib import Path


def test_run_py_defaults_to_numpy_backend():
    text = (Path(__file__).resolve().parents[1] / "Run.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--backend", choices=["numpy", "numba"], default="numpy")' in text
