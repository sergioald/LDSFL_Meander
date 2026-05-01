
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


def _kdtree_first_hit_point_pair(x, y, ss, dslim3, *, workers: int = 1):
    """Return (i, j) 0-based point indices using KDTree, or None.

    This is an **early-exit** variant.
    The previous implementation used query_pairs(), which can generate a huge
    list of pairs (and allocate a lot) for large N or dense neighborhoods.
    Here we scan i from left-to-right and stop at the first valid hit.
    """
    n = x.size
    end = n - ss - 1
    if n < 4 or ss <= 0 or end <= 1:
        return None

    pts = np.empty((n, 2), dtype=np.float64)
    pts[:, 0] = x
    pts[:, 1] = y
    tree = cKDTree(pts)
    r = float(dslim3)
    r2 = r * r

    # same domain as your MATLAB-style loop:
    # i in [0 .. n-ss-1], j in [i+ss .. n-ss-1]
    for i in range(0, end + 1):
        neigh = tree.query_ball_point(pts[i], r, workers=workers)
        if not neigh:
            continue

        # keep only candidates far enough ahead and within domain
        cand = [j for j in neigh if (j >= i + ss) and (j <= end)]
        if not cand:
            continue

        cand_j = np.asarray(cand, dtype=np.int64)
        dx = x[i] - x[cand_j]
        dy = y[i] - y[cand_j]
        d2 = dx * dx + dy * dy
        # Ensure it's truly within r (guard against eps/rounding)
        d2_min = float(d2.min())
        if d2_min >= r2:
            # should be rare, but keep correctness
            within = cand_j[d2 < r2]
            if within.size == 0:
                continue
            dx = x[i] - x[within]
            dy = y[i] - y[within]
            d2 = dx * dx + dy * dy
            d2_min = float(d2.min())
            cand_j = within

        # choose closest j; tie-break smallest j
        j = int(cand_j[d2 == d2_min].min())
        return int(i), j

    return None


def _refine_long_segments_linear(x, y, ds_target):
    """
    Insert points into segments longer than ds_target using linear interpolation.
    Returns refined (xr, yr) plus mapping arrays:
      - seg_left[k] = original left endpoint index for refined point k
      - t[k] in [0,1], position along that original segment (0 => exactly left vertex)
    """
    n = x.size
    xr = [float(x[0])]
    yr = [float(y[0])]
    seg_left = [0]
    tvals = [0.0]

    for i in range(n - 1):
        x0, y0 = float(x[i]), float(y[i])
        x1, y1 = float(x[i + 1]), float(y[i + 1])
        dx, dy = x1 - x0, y1 - y0
        L = (dx * dx + dy * dy) ** 0.5
        if L <= 1e-15:
            continue

        # number of subsegments so that each <= ds_target
        m = int(np.ceil(L / ds_target))
        # insert interior points only
        for k in range(1, m):
            tt = k / m
            xr.append(x0 + tt * dx)
            yr.append(y0 + tt * dy)
            seg_left.append(i)
            tvals.append(tt)

    xr.append(float(x[-1]))
    yr.append(float(y[-1]))
    seg_left.append(n - 1)
    tvals.append(0.0)

    return (np.asarray(xr), np.asarray(yr),
            np.asarray(seg_left, dtype=np.int64),
            np.asarray(tvals, dtype=np.float64))


def _map_refined_idx_to_original(idx_ref, seg_left, tvals):
    """Map refined point index to nearest original vertex index."""
    i = int(seg_left[idx_ref])
    t = float(tvals[idx_ref])
    # If the refined point lies inside segment i->i+1, choose nearest endpoint
    if t <= 0.5:
        return i
    else:
        return i + 1


def find_neck_cutoff_kdtree_with_refine(
    xa: np.ndarray,
    ya: np.ndarray,
    ss: int,
    dslim3: float,
    *,
    refine_trigger: float = 0.5,     # ds_max > refine_trigger * dslim3 triggers refinement
    refine_target: float = 0.5,      # refine to ds_target = refine_target * dslim3
    max_refined_points: int = 200_000
):
    """
    1) KDTree on original points.
    2) If no hit and ds_max > refine_trigger*dslim3:
         temporarily refine long segments to ds_target = refine_target*dslim3,
         then KDTree on refined points,
         then map found refined indices back to original vertex indices.

    Returns (i0, j0) 0-based indices into original xa,ya, or None.
    """
    xa = np.asarray(xa, dtype=np.float64)
    ya = np.asarray(ya, dtype=np.float64)
    n = xa.size
    if n < 4:
        return None

    # Fast path
    hit = _kdtree_first_hit_point_pair(xa, ya, ss, dslim3)
    if hit is not None:
        return hit

    # Check resolution
    seglen = np.sqrt(np.diff(xa) ** 2 + np.diff(ya) ** 2)
    ds_max = float(seglen.max()) if seglen.size else 0.0
    if ds_max <= refine_trigger * float(dslim3):
        return None

    # Temporary refinement
    ds_target = refine_target * float(dslim3)
    if ds_target <= 0:
        return None

    xr, yr, seg_left, tvals = _refine_long_segments_linear(xa, ya, ds_target)
    if xr.size > max_refined_points:
        # avoid blowups if dslim3 is tiny relative to spacing
        return None

    # Adjust ss to keep roughly same *physical* separation.
    # If original spacing is ~mean(seglen), refined spacing ~ds_target.
    ds_mean = float(seglen.mean()) if seglen.size else ds_target
    ss_ref = max(1, int(np.ceil(ss * (ds_mean / ds_target))))

    hit_r = _kdtree_first_hit_point_pair(xr, yr, ss_ref, dslim3)
    if hit_r is None:
        return None

    i_r, j_r = hit_r
    i0 = _map_refined_idx_to_original(i_r, seg_left, tvals)
    j0 = _map_refined_idx_to_original(j_r, seg_left, tvals)

    # Re-enforce original constraints (index and domain)
    end = n - ss - 1
    if not (0 <= i0 <= end):
        return None
    if not (i0 + ss <= j0 <= end):
        return None

    # Final distance check in original coordinates
    dx = xa[i0] - xa[j0]
    dy = ya[i0] - ya[j0]
    if (dx * dx + dy * dy) < float(dslim3) ** 2:
        return int(i0), int(j0)

    return None


def smooth_xy_via_theta(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "savgol",          # "savgol" | "gaussian" | "sos"
    periodic: bool = False,          # True if curve is closed/periodic
    enforce_closed: bool | None = None,  # for periodic curves, close endpoints after integration
    # --- Savitzky–Golay ---
    window_length: int = 21,
    polyorder: int = 3,
    # --- Gaussian ---
    sigma: float | None = None,      # sigma in *samples* (if None, sigma_s is used)
    sigma_s: float | None = None,    # sigma in arclength units
    # --- SOS low-pass ---
    sos_order: int = 3,
    cutoff: float | None = None,     # cutoff in cycles-per-arclength-unit (e.g. 0.1 means wavelength ~10)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth x,y such that curvature becomes smooth by smoothing tangent angle theta(s).

    Notes:
    - Works best if x,y are ordered along the curve.
    - Strongly recommended to use periodic=True for closed curves.
    - cutoff is in cycles per unit arclength: wavelength ≈ 1/cutoff.
      Example: cutoff=0.1 -> remove features smaller than ~10 units of arclength (tune!).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1D arrays of the same shape.")
    n = x.size
    if n < 5:
        return x.copy(), y.copy()

    if enforce_closed is None:
        enforce_closed = periodic

    # --- 1) arc-length parameter s (original) ---
    dx0 = np.diff(x)
    dy0 = np.diff(y)
    ds0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
    # avoid zero-length segments (can break resampling)
    ds0 = np.maximum(ds0, 1e-15)
    s = np.empty(n, dtype=np.float64)
    s[0] = 0.0
    s[1:] = np.cumsum(ds0)
    L = s[-1]
    if not np.isfinite(L) or L <= 0:
        return x.copy(), y.copy()

    # --- 2) resample to uniform arclength ---
    su = np.linspace(0.0, L, n, dtype=np.float64)
    xu = np.interp(su, s, x)
    yu = np.interp(su, s, y)
    dsu = su[1] - su[0]

    # --- 3) tangent angle theta(s) ---
    # derivatives w.r.t arclength; uniform step -> use spacing dsu
    dx = np.gradient(xu, dsu, edge_order=2)
    dy = np.gradient(yu, dsu, edge_order=2)
    theta = np.unwrap(np.arctan2(dy, dx))

    # --- 4) smooth theta ---
    method_l = method.strip().lower()
    theta_s = _smooth_1d(theta, method_l,
                         periodic=periodic,
                         window_length=window_length,
                         polyorder=polyorder,
                         sigma=sigma,
                         sigma_s=sigma_s,
                         dsu=dsu,
                         sos_order=sos_order,
                         cutoff=cutoff)

    # --- 5) reconstruct x,y by integrating cos/sin(theta) ---
    c = np.cos(theta_s)
    snt = np.sin(theta_s)

    # trapezoidal integration for better accuracy than simple cumulative sum
    dx_inc = 0.5 * (c[1:] + c[:-1]) * dsu
    dy_inc = 0.5 * (snt[1:] + snt[:-1]) * dsu

    xrec = np.empty_like(xu)
    yrec = np.empty_like(yu)
    xrec[0] = xu[0]
    yrec[0] = yu[0]
    xrec[1:] = xu[0] + np.cumsum(dx_inc)
    yrec[1:] = yu[0] + np.cumsum(dy_inc)

    # Optionally enforce closure for periodic curves
    if enforce_closed:
        ex = xrec[-1] - xrec[0]
        ey = yrec[-1] - yrec[0]
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        xrec = xrec - t * ex
        yrec = yrec - t * ey

    # --- 6) interpolate back to original s positions ---
    xs = np.interp(s, su, xrec)
    ys = np.interp(s, su, yrec)
    return xs, ys


def _smooth_1d(
    a: np.ndarray,
    method: str,
    *,
    periodic: bool,
    window_length: int,
    polyorder: int,
    sigma: float | None,
    sigma_s: float | None,
    dsu: float,
    sos_order: int,
    cutoff: float | None,
) -> np.ndarray:
    if method == "savgol":
        from scipy.signal import savgol_filter

        wl = int(window_length)
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, polyorder + 2 + ((polyorder + 2) % 2 == 0))  # ensure >= polyorder+2 and odd
        mode = "wrap" if periodic else "interp"
        return savgol_filter(a, window_length=wl, polyorder=int(polyorder), mode=mode)

    if method == "gaussian":
        from scipy.ndimage import gaussian_filter1d

        if sigma is None:
            if sigma_s is None:
                raise ValueError("For method='gaussian', provide sigma (samples) or sigma_s (arclength units).")
            sigma = float(sigma_s) / float(dsu)
        mode = "wrap" if periodic else "nearest"
        return gaussian_filter1d(a, sigma=float(sigma), mode=mode)

    if method == "sos":
        from scipy.signal import butter, sosfiltfilt

        if cutoff is None:
            raise ValueError("For method='sos', provide cutoff in cycles-per-arclength-unit (e.g. 0.1).")
        # sampling frequency in samples per arclength unit:
        fs = 1.0 / float(dsu)
        # cutoff in same units as fs (cycles per arclength unit)
        sos = butter(int(sos_order), float(cutoff), btype="low", fs=fs, output="sos")

        if periodic:
            # sosfiltfilt is not periodic; emulate by wrapping padding
            pad = max(3 * int(sos_order), 32)
            pad = min(pad, a.size - 1)
            ext = np.concatenate([a[-pad:], a, a[:pad]])
            out = sosfiltfilt(sos, ext)
            return out[pad:-pad]
        else:
            return sosfiltfilt(sos, a)

    raise ValueError("method must be one of: 'savgol', 'gaussian', 'sos'")



def matlab_gradient(x: np.ndarray) -> np.ndarray:
    """
    Approximate MATLAB's gradient(x) for 1D vectors with unit spacing.
    MATLAB uses first-order at boundaries and central differences inside.
    numpy.gradient with edge_order=1 matches this reasonably well.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.gradient(x, edge_order=1)

def matlab_spline(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    MATLAB spline(x,y,xq): not-a-knot cubic spline interpolation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    cs = CubicSpline(x, y, bc_type='not-a-knot', extrapolate=True)
    return cs(xq)

def smooth_sgolay(x: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """
    Approximate MATLAB smooth(x, window_length, 'sgolay', polyorder).
    Uses savgol_filter with interpolation at edges.
    """
    x = np.asarray(x, dtype=np.float64)
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, x.size if x.size % 2 == 1 else x.size - 1)
    wl = max(wl, polyorder + 2 + (polyorder + 2) % 2)  # ensure wl > polyorder, odd
    if wl > x.size:
        # Not enough points; return unchanged
        return x.copy()
    return savgol_filter(x, window_length=wl, polyorder=int(polyorder), mode='interp')

def unwrap_angles_like_matlab(theta: np.ndarray) -> np.ndarray:
    """
    Replicates the specific unwrapping loop used in PreProf_3/Geometry4.
    """
    theta = np.asarray(theta, dtype=np.float64)
    out = np.empty_like(theta)
    last = 0.0
    for i, angle in enumerate(theta):
        while angle < last - np.pi:
            angle += 2*np.pi
        while angle > last + np.pi:
            angle -= 2*np.pi
        last = angle
        out[i] = angle
    return out

def jt_p_string(Ntstep: int, cut_cnt: int) -> str:
    """
    Replicates:
      jt_p=strcat(num2str(Ntstep*10,'%.0f'),num2str(cut_cnt,'%.0f'));
      jt_p=jt_p(end-length(num2str(Ntstep)):end);
    """
    a = f"{int(round(Ntstep*10))}{int(round(cut_cnt))}"
    # MATLAB: length(num2str(Ntstep)) characters from the end, inclusive
    L = len(str(int(round(Ntstep))))
    return a[-L:]
