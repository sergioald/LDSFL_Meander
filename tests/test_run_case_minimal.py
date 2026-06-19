from __future__ import annotations

import shutil
from pathlib import Path

from ldsfl.main import run_case

ROOT = Path(__file__).resolve().parents[1]


def test_run_case_minimal_example_creates_expected_outputs(tmp_path):
    """Run one bundled example step in an isolated workspace.

    This is intentionally a small integration test rather than a full
    morphodynamic regression test. It verifies that the input readers,
    solver loop, stop criteria, geometry update, and output writers work
    together without writing into the repository's real Output/ directory.
    """
    input_src = ROOT / "Input"
    input_dst = tmp_path / "Input"
    input_dst.mkdir()
    shutil.copy(input_src / "Parameter.csv", input_dst / "Parameter.csv")
    shutil.copy(input_src / "xy.csv", input_dst / "xy.csv")

    result = run_case(
        tmp_path,
        case_i=1,
        Nprint=5,
        Ntstep=10,
        Max_Cut=100,
        max_steps=1,
        stop_on_steps=True,
        stop_on_time=False,
        stop_on_cutoffs=True,
        do_plots=False,
    )

    assert result["steps"] == 1
    assert result["stop_criteria_reached"] == ["max_steps"]
    assert "max_steps" in result["stop_reason"]
    assert result["id_files"]
    assert result["sinuo_final"] > 0.0
    assert isinstance(result["sinuosity_stability"], dict)

    output_root = tmp_path / "Output" / result["id_files"]
    xyu_dir = output_root / "xyu"
    files_dir = output_root / "files"

    assert xyu_dir.is_dir()
    assert files_dir.is_dir()
    assert list(xyu_dir.glob("*.csv"))
    assert (files_dir / f"sinuosity_history_{result['id_files']}.csv").is_file()
