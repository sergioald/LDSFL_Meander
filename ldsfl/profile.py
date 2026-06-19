
from __future__ import annotations

import numpy as np

from .mathutils import matlab_gradient, unwrap_angles_like_matlab


def preprof_3(xa: np.ndarray, ya: np.ndarray, dsliminicial: float):
    """
    Port of PreProf_3.m
    Returns:
      s, x, y, th, Ns, deltas, wave_l, valle_l, sinuo
    """
    xa = np.asarray(xa, dtype=np.float64).copy()
    ya = np.asarray(ya, dtype=np.float64).copy()
    Na = xa.size

    delta_x = np.diff(xa)
    delta_y = np.diff(ya)
    sa = np.concatenate(([0.0], np.cumsum(np.sqrt(delta_x**2 + delta_y**2))))

    # gradients and angles with custom unwrap loop
    dxg = matlab_gradient(xa)
    dyg = matlab_gradient(ya)
    theta_raw = np.arctan2(dyg, dxg)
    theta = unwrap_angles_like_matlab(theta_raw)
    theta = -1.0 * theta

    deltas = float(sa[-1] / (Na - 1))
    wave_l = float(sa[-1])
    valle_l = float(np.sqrt((xa[0] - xa[-1])**2 + (ya[0] - ya[-1])**2))
    sinuo = float(wave_l / valle_l) if valle_l != 0 else np.inf

    Ns = int(sa.size)
    return sa, xa, ya, theta, Ns, deltas, wave_l, valle_l, sinuo
