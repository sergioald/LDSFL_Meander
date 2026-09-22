from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ldsfl.gui_utils import DimensionlessInputs, validate_dimensionless
from ldsfl.inputs import dimensionless_input_table, read_parameter_table, read_xy


def _valid_parameter_frame():
    return pd.DataFrame(
        {"Id": [1], "Beta": [9.0], "ds": [0.005], "Thetha": [0.3],
         "flagbed": [2], "r": [0.5], "Mdat": [6]}
    )


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


def test_dimensionless_input_table_rejects_missing_id_instead_of_selecting_a_row():
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

    with pytest.raises(ValueError, match='case ID 2'):
        dimensionless_input_table(df, 2)


@pytest.mark.parametrize(
    "column,value",
    [
        ("Id", np.nan), ("Id", np.inf), ("Id", 0), ("Id", -1), ("Id", 1.5),
        ("Beta", np.nan), ("Beta", np.inf), ("Beta", -np.inf), ("Beta", 0), ("Beta", -1),
        ("ds", np.nan), ("ds", np.inf), ("ds", 0), ("ds", -1),
        ("Thetha", np.nan), ("Thetha", np.inf), ("Thetha", 0), ("Thetha", -1),
        ("flagbed", np.nan), ("flagbed", np.inf), ("flagbed", 0), ("flagbed", -1),
        ("flagbed", 2.5), ("flagbed", 3),
        ("r", np.nan), ("r", np.inf), ("r", 0), ("r", -1),
        ("Mdat", np.nan), ("Mdat", np.inf), ("Mdat", 0), ("Mdat", -1), ("Mdat", 6.5),
    ],
)
def test_parameter_table_rejects_invalid_scientific_values(tmp_path, column, value):
    frame = _valid_parameter_frame()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    path = tmp_path / "Parameter.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        read_parameter_table(path)


def test_parameter_table_rejects_duplicate_ids(tmp_path):
    frame = pd.concat([_valid_parameter_frame(), _valid_parameter_frame()], ignore_index=True)
    path = tmp_path / "Parameter.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_parameter_table(path)


def test_parameter_table_reports_missing_required_columns(tmp_path):
    frame = _valid_parameter_frame().drop(columns=["Thetha", "Mdat"])
    path = tmp_path / "Parameter.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns.*Thetha.*Mdat"):
        read_parameter_table(path)


@pytest.mark.parametrize("field,value", [("Mdat", 6.5), ("flagbed", 2.5)])
def test_gui_scientific_validation_does_not_truncate_fractional_integers(field, value):
    values = dict(beta=9.0, ds=0.005, theta0=0.3, flagbed=2, rpic_0=0.5, Mdat=6)
    values[field] = value
    with pytest.raises(ValueError, match=field):
        validate_dimensionless(DimensionlessInputs(**values))
