
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .validation import validate_parameter_table


def read_parameter_table(param_csv: Path) -> pd.DataFrame:
    # MATLAB readtable with "preserve" keeps original names; pandas preserves headers as-is.
    return validate_parameter_table(pd.read_csv(param_csv))

def read_xy(xy_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(xy_csv, header=None, encoding='utf-8-sig')
    arr = df.values.astype(np.float64)
    # MATLAB stores row vectors; Main transposes with '
    xap = arr[:, 0].astype(np.float64)
    yap = arr[:, 1].astype(np.float64)
    return xap, yap

def dimensionless_input_table(df: pd.DataFrame, i: int):
    """
    Port of Dimensionless_Input_Table.m
    """
    df = validate_parameter_table(df)
    rows = df[df['Id'] == i]
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one row for case ID {i}; found {len(rows)}.')
    row = rows.iloc[0]
    beta = float(row['Beta'])
    ds = float(row['ds'])
    theta0 = float(row['Thetha'])
    flagbed = int(row['flagbed'])
    rpic_0 = float(row['r'])
    Mdat = int(row['Mdat'])
    return beta, ds, theta0, flagbed, rpic_0, Mdat
