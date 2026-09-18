
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_parameter_table(param_csv: Path) -> pd.DataFrame:
    # MATLAB readtable with "preserve" keeps original names; pandas preserves headers as-is.
    df = pd.read_csv(param_csv)
    ids = pd.to_numeric(df['Id'], errors='raise')
    if df.empty or not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError('Case IDs must be positive integers and the parameter table must not be empty.')
    if not ids.is_unique:
        raise ValueError('Case IDs must be unique.')
    df['Id'] = ids.astype('int64')
    return df

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
