"""Regression test for variable-history bookkeeping in run_case."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from ldsfl.main import run_case

ROOT = Path(__file__).resolve().parents[1]


def test_every_step_is_recorded_exactly_once(tmp_path):
    input_dst = tmp_path / "Input"
    input_dst.mkdir()
    shutil.copy(ROOT / "Input" / "Parameter.csv", input_dst / "Parameter.csv")
    shutil.copy(ROOT / "Input" / "xy.csv", input_dst / "xy.csv")

    max_steps = 12
    nprint = 5  # deliberately not a divisor of max_steps: forces a partial final block
    result = run_case(
        tmp_path,
        case_i=1,
        Nprint=nprint,
        Ntstep=10,
        Max_Cut=100,
        max_steps=max_steps,
        do_plots=False,
    )
    assert result["steps"] == max_steps

    files_dir = tmp_path / "Output" / result["id_files"] / "files"
    jts: list[float] = []
    for csv_path in sorted(files_dir.glob("var_*.csv")):
        df = pd.read_csv(csv_path)
        assert not df["jt"].isna().any(), f"NaN row found in {csv_path.name}"
        jts.extend(df["jt"].tolist())

    assert sorted(jts) == [float(j) for j in range(1, max_steps + 1)], (
        f"variable history incomplete or duplicated: recorded jt = {sorted(jts)}"
    )
