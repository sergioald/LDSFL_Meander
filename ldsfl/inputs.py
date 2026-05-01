
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

def read_parameter_table(param_csv: Path) -> pd.DataFrame:
    # MATLAB readtable with "preserve" keeps original names; pandas preserves headers as-is.
    return pd.read_csv(param_csv)

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
    # MATLAB: index = find(Input_param.Id==i);
    idx = df.index[df['Id'] == i]
    if len(idx) == 0:
        # fallback: treat i as 1-based row index
        idx0 = i - 1
    else:
        idx0 = int(idx[0])

    beta = float(df.loc[idx0, 'Beta'])
    ds = float(df.loc[idx0, 'ds'])
    theta0 = float(df.loc[idx0, 'Thetha'])
    flagbed = int(df.loc[idx0, 'flagbed'])
    rpic_0 = float(df.loc[idx0, 'r'])
    Mdat = int(df.loc[idx0, 'Mdat'])
    return beta, ds, theta0, flagbed, rpic_0, Mdat
