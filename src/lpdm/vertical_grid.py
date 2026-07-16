"""Terrain-following (hybrid) vertical coordinate: resample ERA5 pressure-level
meteorology onto a fixed above-ground-level (AGL) height grid, once per met window.

GLIDE streams ARCO ERA5 *pressure* levels, which are quasi-horizontal and slice
through mountains — so below-ground levels exist and, left alone, poison the
near-surface fields and leave the surface footprint empty over high terrain
(dev/CHECKPOINT.md Finding 7). Native-model-level LPDMs (FLEXPART) never hit this
because their levels already follow the terrain. Here we do what FLEXPART's
`verttransform` does: regrid onto a fixed terrain-following AGL grid and
slope-correct the vertical velocity, once per window (amortised over ~60 steps).

Pure NumPy, no torch — the regrid runs in the met reader (CPU, per window), not in
the per-step hot path. Fields are returned surface-first (ascending AGL).
"""

from __future__ import annotations

import numpy as np

# Default fixed AGL grid: fine near the surface (to resolve the 0-40 m and
# 40-1000 m footprint bins), geometric aloft. A modelling choice — FLEXPART makes
# the equivalent grid configurable; promote to run config later.
_DEFAULT_AGL_LEVELS_M: tuple[float, ...] = (
    0.0, 10.0, 20.0, 40.0, 80.0, 120.0, 200.0, 300.0, 500.0, 750.0, 1000.0,
    1250.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0,
    10000.0, 12000.0, 15000.0,
)

# Local metres per degree (spherical-earth approximation, matches gpu_engine).
_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LON_EQ = 111320.0


def default_agl_levels(alt_max_m: float) -> np.ndarray:
    """Fixed ascending AGL height grid (m) covering ``[0, alt_max_m]``."""
    levels = np.array([h for h in _DEFAULT_AGL_LEVELS_M if h <= alt_max_m], dtype="float64")
    if levels.size < 2:
        raise ValueError(f"alt_max_m={alt_max_m} too small for a vertical grid")
    if levels[-1] < alt_max_m:
        levels = np.append(levels, float(alt_max_m))
    return levels


def regrid_columns_to_agl(
    field_p: np.ndarray,
    height_agl_p: np.ndarray,
    agl_levels: np.ndarray,
) -> np.ndarray:
    """Per-column linear interpolation of pressure-level fields onto an AGL grid.

    Args:
        field_p: ``[C, Zp, Y, X]`` fields on the pressure-level grid.
        height_agl_p: ``[Zp, Y, X]`` geometric height above ground of each pressure
            level, per column. NOT clamped at 0 — sub-surface levels legitimately
            carry negative AGL and are excluded here (see below).
        agl_levels: ``[Za]`` strictly-ascending target AGL heights (>= 0).

    Returns:
        ``[C, Za, Y, X]`` fields interpolated onto ``agl_levels`` (surface first).

    Sub-surface pressure levels (negative AGL, ERA5's fictitious below-ground
    extrapolation) are excluded: the lower bracket index is clamped to each
    column's first above-ground level, so a target height below it constant-
    extrapolates from that lowest real level rather than blending in below-ground
    values. Targets above the top level constant-extrapolate from the top.
    """
    if field_p.ndim != 4:
        raise ValueError(f"field_p must be [C, Zp, Y, X]; got {field_p.shape}")
    if height_agl_p.shape != field_p.shape[1:]:
        raise ValueError(
            f"height_agl_p {height_agl_p.shape} must match field_p[1:] {field_p.shape[1:]}"
        )
    agl_levels = np.asarray(agl_levels, dtype="float64")
    if agl_levels.ndim != 1 or not np.all(np.diff(agl_levels) > 0):
        raise ValueError("agl_levels must be 1-D strictly ascending")

    C, Zp, Y, X = field_p.shape
    Za = agl_levels.size

    # Work surface-up. Pressure levels are stored top-of-atmosphere first (AGL
    # descending); flip both to ascending AGL so bracketing is monotonic.
    mean_profile = np.nanmean(height_agl_p, axis=(1, 2))
    if mean_profile[0] > mean_profile[-1]:
        h = np.ascontiguousarray(height_agl_p[::-1])
        f = np.ascontiguousarray(field_p[:, ::-1])
    else:
        h = np.ascontiguousarray(height_agl_p)
        f = np.ascontiguousarray(field_p)

    # First above-ground level per column (excludes sub-surface levels below it).
    k0 = np.argmax(h >= 0.0, axis=0)  # [Y, X]

    out = np.empty((C, Za, Y, X), dtype=field_p.dtype)
    for k in range(Za):
        z_t = float(agl_levels[k])
        lower = np.clip(np.sum(h < z_t, axis=0) - 1, k0, Zp - 2)  # [Y, X]
        upper = lower + 1
        h_lo = np.take_along_axis(h, lower[None], axis=0)[0]  # [Y, X]
        h_hi = np.take_along_axis(h, upper[None], axis=0)[0]
        w = np.clip((z_t - h_lo) / np.clip(h_hi - h_lo, 1e-6, None), 0.0, 1.0)  # [Y, X]
        li = np.broadcast_to(lower[None, None], (C, 1, Y, X))
        ui = np.broadcast_to(upper[None, None], (C, 1, Y, X))
        f_lo = np.take_along_axis(f, li, axis=1)[:, 0]  # [C, Y, X]
        f_hi = np.take_along_axis(f, ui, axis=1)[:, 0]
        out[:, k] = f_lo * (1.0 - w) + f_hi * w
    return out


def terrain_gradient(
    terrain_m: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Terrain slope ``(dh/dx_east, dh/dy_north)`` in metres per metre, ``[Y, X]``.

    Uses the actual coordinate arrays, so descending latitude is handled correctly.
    """
    dh_dlat, dh_dlon = np.gradient(terrain_m, lat_deg, lon_deg)  # per degree
    m_per_deg_lon = np.clip(
        _M_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat_deg)), 1.0, None
    )  # [Y]
    dhdx = dh_dlon / m_per_deg_lon[:, None]
    dhdy = dh_dlat / _M_PER_DEG_LAT
    return dhdx, dhdy


def slope_correct_w(
    w_agl: np.ndarray,
    u_agl: np.ndarray,
    v_agl: np.ndarray,
    dhdx: np.ndarray,
    dhdy: np.ndarray,
    agl_levels: np.ndarray,
    z_top_m: float,
) -> np.ndarray:
    """Transform the vertical velocity into the terrain-following AGL frame.

    A particle riding the horizontal wind over sloping terrain must move vertically
    at ``u·∂h/∂x + v·∂h/∂y`` just to hold its height above ground; the AGL-frame
    vertical velocity is ``w_agl = w − (u·∂h/∂x + v·∂h/∂y)``. The correction is
    tapered ``∝ (1 − z/z_top)`` so the coordinate hugs the terrain near the ground
    and relaxes to flat (uncorrected ``w``) near the model top — otherwise mountains
    would wobble the stratosphere. All arrays ``[Za, Y, X]``; slopes ``[Y, X]``.
    """
    taper = np.clip(1.0 - agl_levels / max(float(z_top_m), 1.0), 0.0, 1.0)  # [Za]
    slope_w = u_agl * dhdx[None] + v_agl * dhdy[None]  # [Za, Y, X]
    return w_agl - taper[:, None, None] * slope_w
