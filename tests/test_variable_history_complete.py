"""Regression test for variable-history bookkeeping in run_case."""

from __future__ import annotations
import pytest
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
    rows = []
    for csv_path in sorted(files_dir.glob("var_*.csv")):
        df = pd.read_csv(csv_path)
        assert not df["jt"].isna().any(), f"NaN row found in {csv_path.name}"
        rows.append(df)

    history = pd.concat(rows).sort_values("state_step")
    assert history["state_step"].tolist() == [float(j) for j in range(max_steps + 1)]
    assert history["jt"].tolist() == [float(j) for j in range(1, max_steps + 2)]
    assert history.iloc[0]["dt"] == 0.0
    assert history.iloc[0]["dt_cum"] == 0.0
    assert history.iloc[-1]["dt_cum"] == pytest.approx(
                result["dt_cum"],
                rel=1.0e-15,
                abs=1.0e-9,
            )

    sinuosity = pd.read_csv(files_dir / f"sinuosity_history_{result['id_files']}.csv")
    assert sinuosity["step"].tolist() == list(range(max_steps + 1))
