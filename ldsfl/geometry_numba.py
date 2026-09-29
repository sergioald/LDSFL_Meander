"""Optional Numba implementation of the exact MATLAB-style angle unwrap.

This module is imported only when the explicit geometry unwrap backend is
selected. Keep the kernel loop and arithmetic aligned with the Python
reference in :mod:`ldsfl.mathutils`.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=False, fastmath=False)
def unwrap_angles_like_matlab_numba(theta: np.ndarray) -> np.ndarray:
    """Unwrap a 1-D float64 angle array using the reference sequential loop."""
    out = np.empty_like(theta)
    last = 0.0
    for i in range(theta.size):
        angle = theta[i]
        while angle < last - np.pi:
            angle += 2.0 * np.pi
        while angle > last + np.pi:
            angle -= 2.0 * np.pi
        out[i] = angle
        last = angle
    return out
