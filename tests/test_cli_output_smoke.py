"""Smoke test that the CLI writes expected generated-output files."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import run_ldsfl

ROOT = Path(__file__).resolve().parents[1]


def _copy_inputs(tmp_path: Path) -> None:
    input_dst = tmp_path / "Input"
    input_dst.mkdir()
    shutil.copy(ROOT / "Input" / "Parameter.csv", input_dst / "Parameter.csv")
    shutil.copy(ROOT / "Input" / "xy.csv", input_dst / "xy.csv")


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_cli_smoke_writes_expected_output_files(monkeypatch, tmp_path):
    """A tiny real CLI run should create the documented Output tree."""
    _copy_inputs(tmp_path)

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
            "1",
            "--nprint",
            "1",
            "--ntstep",
            "10",
            "--max-cut",
            "100",
            "--no-plots",
        ],
    )

    run_ldsfl.main()

    output_root = tmp_path / "Output"
    assert output_root.is_dir()

    case_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert len(case_dirs) == 1
    case_dir = case_dirs[0]

    xyu_dir = case_dir / "xyu"
    files_dir = case_dir / "files"
    plot_dir = case_dir / "plot"
    xy_cut_dir = case_dir / "xy_cut"

    assert xyu_dir.is_dir()
    assert files_dir.is_dir()
    assert plot_dir.is_dir()
    assert xy_cut_dir.is_dir()

    xyu_files = sorted(xyu_dir.glob("xyu_*.csv"))
    assert xyu_files, "CLI run did not write any geometry snapshot CSV files"
    assert _csv_header(xyu_files[-1]) == ["x", "y", "s", "th", "c", "U"]

    var_files = sorted(files_dir.glob("var_*.csv"))
    assert var_files, "CLI run did not write any variable-history CSV files"
    var_header = _csv_header(var_files[-1])
    assert {"jt", "dt", "dt_cum", "sinuo"}.issubset(var_header)

    sinuosity_files = sorted(files_dir.glob("sinuosity_history_*.csv"))
    assert len(sinuosity_files) == 1
    sinuosity_rows = _csv_rows(sinuosity_files[0])
    assert sinuosity_rows[0] == ["step", "sinuo"]
    assert len(sinuosity_rows) >= 2
