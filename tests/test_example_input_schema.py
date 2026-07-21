"""Schema and sanity checks for bundled example input files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ldsfl.inputs import dimensionless_input_table, read_parameter_table, read_xy

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "Input"
PARAMETER_CSV = INPUT_DIR / "Parameter.csv"
XY_CSV = INPUT_DIR / "xy.csv"

REQUIRED_PARAMETER_COLUMNS = {
    "Id",
    "Beta",
    "ds",
    "Thetha",
    "flagbed",
    "r",
    "Mdat",
}

NUMERIC_PARAMETER_COLUMNS = [
    "Id",
    "Beta",
    "ds",
    "Thetha",
    "flagbed",
    "r",
    "Mdat",
]


def _read_parameter_csv() -> pd.DataFrame:
    return pd.read_csv(PARAMETER_CSV, encoding="utf-8-sig")


def test_bundled_input_files_exist():
    assert PARAMETER_CSV.is_file(), "Missing bundled Input/Parameter.csv"
    assert XY_CSV.is_file(), "Missing bundled Input/xy.csv"


def test_parameter_csv_has_required_schema_and_cases():
    df = _read_parameter_csv()

    missing = REQUIRED_PARAMETER_COLUMNS.difference(df.columns)
    assert not missing, f"Input/Parameter.csv is missing columns: {sorted(missing)}"
    assert not df.empty, "Input/Parameter.csv must contain at least one example case"

    for column in NUMERIC_PARAMETER_COLUMNS:
        values = pd.to_numeric(df[column], errors="raise")
        assert np.isfinite(values.to_numpy(dtype=float)).all(), f"Non-finite values in column {column!r}"

    case_ids = pd.to_numeric(df["Id"], errors="raise")
    assert (case_ids > 0).all(), "Case IDs must be positive"
    assert (case_ids % 1 == 0).all(), "Case IDs must be integer-like"
    assert case_ids.is_unique, "Case IDs must be unique"

    assert (pd.to_numeric(df["Beta"]) > 0).all(), "Beta must be positive"
    assert (pd.to_numeric(df["ds"]) > 0).all(), "ds must be positive"
    assert (pd.to_numeric(df["Mdat"]) > 0).all(), "Mdat must be positive"
    assert set(pd.to_numeric(df["flagbed"]).astype(int)).issubset({1, 2}), "flagbed must use known values 1 or 2"


def test_xy_csv_has_coordinate_schema():
    df = pd.read_csv(XY_CSV, header=None, encoding="utf-8-sig")

    assert df.shape[0] >= 4, "Input/xy.csv must contain enough centreline points for a smoke run"
    assert df.shape[1] >= 2, "Input/xy.csv must have at least two coordinate columns"

    xy = df.iloc[:, :2].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    assert np.isfinite(xy).all(), "Input/xy.csv contains non-finite coordinates"
    assert np.unique(xy[:, 0]).size > 1, "x coordinates should not all be identical"
    assert np.unique(xy[:, 1]).size > 1, "y coordinates should not all be identical"


def test_bundled_inputs_are_accepted_by_project_readers():
    params = read_parameter_table(PARAMETER_CSV)
    x, y = read_xy(XY_CSV)

    assert len(x) == len(y)
    assert len(x) >= 4
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    first_case_id = int(pd.to_numeric(params["Id"], errors="raise").iloc[0])
    beta, ds, theta0, flagbed, rpic_0, mdat = dimensionless_input_table(params, first_case_id)

    assert np.isfinite([beta, ds, theta0, rpic_0]).all()
    assert beta > 0
    assert ds > 0
    assert flagbed in {1, 2}
    assert mdat > 0
