"""Optional exact Numba spatial-grid point-pair detector for neck cutoffs."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=False, fastmath=False, parallel=False)
def _spatial_grid_first_hit_kernel(x, y, ss, radius, cell_width):
    """Return exact first-hit indices, or (-2, -2) for unsafe coordinates."""
    n = x.size
    end = n - ss - 1
    if n < 4 or ss <= 0 or end <= 1:
        return -1, -1
    radius2 = radius * radius
    if not math.isfinite(radius2):
        return -2, -2

    cell_x = np.empty(n, dtype=np.int64)
    cell_y = np.empty(n, dtype=np.int64)
    lower_cell_bound = -9.223372036854774e18
    upper_cell_bound = 9.223372036854774e18
    for point in range(n):
        qx = x[point] / cell_width
        qy = y[point] / cell_width
        if (
            not math.isfinite(qx)
            or not math.isfinite(qy)
            or qx <= lower_cell_bound
            or qx >= upper_cell_bound
            or qy <= lower_cell_bound
            or qy >= upper_cell_bound
        ):
            return -2, -2
        cell_x[point] = int(math.floor(qx))
        cell_y[point] = int(math.floor(qy))

    table_size = 1
    while table_size < 2 * n:
        table_size *= 2
    mask = table_size - 1
    used = np.zeros(table_size, dtype=np.bool_)
    key_x = np.empty(table_size, dtype=np.int64)
    key_y = np.empty(table_size, dtype=np.int64)
    head = np.full(table_size, -1, dtype=np.int64)
    next_point = np.empty(n, dtype=np.int64)

    for point in range(n):
        cx = cell_x[point]
        cy = cell_y[point]
        slot = int((cx * 73856093) ^ (cy * 19349663)) & mask
        while used[slot] and (key_x[slot] != cx or key_y[slot] != cy):
            slot = (slot + 1) & mask
        if not used[slot]:
            used[slot] = True
            key_x[slot] = cx
            key_y[slot] = cy
        next_point[point] = head[slot]
        head[slot] = point

    for i in range(end + 1):
        best_j = -1
        best_d2 = radius2
        for offset_x in range(-1, 2):
            target_x = cell_x[i] + offset_x
            for offset_y in range(-1, 2):
                target_y = cell_y[i] + offset_y
                slot = int((target_x * 73856093) ^ (target_y * 19349663)) & mask
                while used[slot]:
                    if key_x[slot] == target_x and key_y[slot] == target_y:
                        point = head[slot]
                        while point >= 0:
                            if i + ss <= point <= end:
                                dx = x[i] - x[point]
                                dy = y[i] - y[point]
                                d2 = dx * dx + dy * dy
                                if d2 < best_d2 or (d2 == best_d2 and point < best_j):
                                    best_d2 = d2
                                    best_j = point
                            point = next_point[point]
                        break
                    slot = (slot + 1) & mask
        if best_j >= 0:
            return i, best_j
    return -1, -1


def spatial_grid_first_hit_point_pair(x, y, ss, dslim3):
    """Return the exact first eligible pair, or ``None`` when there is no hit."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    radius = float(dslim3)
    if xa.ndim != 1 or ya.ndim != 1 or xa.size != ya.size:
        from .mathutils import _kdtree_first_hit_point_pair

        return _kdtree_first_hit_point_pair(xa, ya, ss, radius)
    if radius <= 0.0 or not math.isfinite(radius):
        from .mathutils import _kdtree_first_hit_point_pair

        return _kdtree_first_hit_point_pair(xa, ya, ss, radius)
    cell_width = radius * (1.0 + 16.0 * np.finfo(np.float64).eps)
    if not math.isfinite(cell_width) or cell_width <= radius:
        cell_width = float(np.nextafter(radius, np.inf))
    result = _spatial_grid_first_hit_kernel(xa, ya, int(ss), radius, cell_width)
    if result[0] == -2:
        from .mathutils import _kdtree_first_hit_point_pair

        return _kdtree_first_hit_point_pair(xa, ya, ss, radius)
    if result[0] < 0:
        return None
    return int(result[0]), int(result[1])
