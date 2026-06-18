from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ldsfl.inputs import dimensionless_input_table, read_parameter_table, read_xy


def test_read_parameter_table_preserves_expected_columns(tmp_path):
    parameter_csv = tmp_path / "Parameter.csv"
    parameter_csv.write_text(
        "Id,Beta,ds,Thetha,flagbed,r,Mdat\n"
        "1,9.0,0.005,0.3,2,0.5,6\n",
        encoding="utf-8",
    )

    df = read_parameter_table(parameter_csv)

    assert list(df.columns) == ["Id", "Beta", "ds", "Thetha", "flagbed", "r", "Mdat"]
    assert df.loc[0, "Beta"] == pytest.approx(9.0)


def test_read_xy_handles_utf8_bom_and_returns_float_arrays(tmp_path):
    xy_csv = tmp_path / "xy.csv"
    xy_csv.write_text("\ufeff0.0,1.0\n2.5,3.5\n", encoding="utf-8")

    x, y = read_xy(xy_csv)

    np.testing.assert_allclose(x, [0.0, 2.5])
    np.testing.assert_allclose(y, [1.0, 3.5])
    assert x.dtype == np.float64
    assert y.dtype == np.float64


def test_dimensionless_input_table_selects_row_by_case_id():
    df = pd.DataFrame(
        {
            "Id": [10, 20],
            "Beta": [9.0, 12.0],
            "ds": [0.005, 0.01],
            "Thetha": [0.3, 0.4],
            "flagbed": [2, 1],
            "r": [0.5, 0.7],
            "Mdat": [6, 8],
        }
    )

    assert dimensionless_input_table(df, 20) == (12.0, 0.01, 0.4, 1, 0.7, 8)


def test_dimensionless_input_table_falls_back_to_one_based_row_index_when_id_is_missing():
    df = pd.DataFrame(
        {
            "Id": [10, 20],
            "Beta": [9.0, 12.0],
            "ds": [0.005, 0.01],
            "Thetha": [0.3, 0.4],
            "flagbed": [2, 1],
            "r": [0.5, 0.7],
            "Mdat": [6, 8],
        }
    )

    assert dimensionless_input_table(df, 2) == (12.0, 0.01, 0.4, 1, 0.7, 8)
