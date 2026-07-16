"""Terrain-following (hybrid) vertical coordinate: resample ERA5 pressure-level
meteorology onto a fixed above-ground-level (AGL) height grid, once per met window.

GLIDE streams ARCO ERA5 *pressure* levels, which are quasi-horizontal and slice
through mountains — so below-ground levels exist and, left alone, poison the
near-surface fields and leave the surface footprint empty over high terrain
(dev/CHECKPOINT.md Finding 7). Native-model-level LPDMs (FLEXPART) never hit this
because their levels already follow the terrain. Here we do what FLEXPART's
`verttransform` does: regrid onto a fixed terrain-following AGL grid and
slope-correct the vertical velocity, once per window (amortised over ~60 steps).

NumPy in / NumPy out; internals use torch (batched `searchsorted` + multithreaded
`gather` — ~10x faster than the equivalent NumPy, measured on a real ERA5 hour).
The regrid runs in the met reader (per window, on the prefetch thread), not in the
per-step hot path. Fields are returned surface-first (ascending AGL).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

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


@dataclass(frozen=True)
class AglRegridWeights:
    """Precomputed per-column bracketing for a pressure->AGL regrid.

    The bracketing depends only on the level heights and the target grid, not on
    the field being regridded — so it is computed ONCE per met window and shared
    by every tensor regridded onto the same AGL grid (hour_start, hour_end, the
    pressure profile). Splitting this out cut the per-window regrid CPU cost ~
    an order of magnitude vs recomputing per tensor per level (perf review
    2026-07-16 #3): the regrid runs on the met prefetch thread, and at full-domain
    scale the unsplit version could no longer be hidden behind GPU compute.
    """

    lower: np.ndarray  # [Za, Y, X] intp — bracket index into the ASCENDING level axis
    frac: np.ndarray   # [Za, Y, X] float64 — interpolation fraction in [0, 1]
    flipped: bool      # source arrived TOA-first and was flipped to ascending
    n_source_levels: int


def compute_agl_regrid_weights(
    height_agl_p: np.ndarray,
    agl_levels: np.ndarray,
) -> AglRegridWeights:
    """Bracket indices + fractions for interpolating onto ``agl_levels``.

    Args:
        height_agl_p: ``[Zp, Y, X]`` geometric height above ground of each pressure
            level, per column. NOT clamped at 0 — sub-surface levels legitimately
            carry negative AGL and are excluded here (see below).
        agl_levels: ``[Za]`` strictly-ascending target AGL heights (>= 0).

    Sub-surface pressure levels (negative AGL, ERA5's fictitious below-ground
    extrapolation) are excluded: the lower bracket index is clamped to each
    column's first above-ground level, so a target height below it constant-
    extrapolates from that lowest real level rather than blending in below-ground
    values. Targets above the top level constant-extrapolate from the top.
    """
    if height_agl_p.ndim != 3:
        raise ValueError(f"height_agl_p must be [Zp, Y, X]; got {height_agl_p.shape}")
    agl_levels = np.asarray(agl_levels, dtype="float64")
    if agl_levels.ndim != 1 or not np.all(np.diff(agl_levels) > 0):
        raise ValueError("agl_levels must be 1-D strictly ascending")

    Zp, Y, X = height_agl_p.shape
    Za = agl_levels.size

    # Work surface-up. Pressure levels are stored top-of-atmosphere first (AGL
    # descending); flip to ascending AGL so bracketing is monotonic.
    mean_profile = np.nanmean(height_agl_p, axis=(1, 2))
    flipped = bool(mean_profile[0] > mean_profile[-1])

    h_t = torch.as_tensor(np.ascontiguousarray(height_agl_p), dtype=torch.float64)
    if flipped:
        h_t = h_t.flip(0)
    # Rows of ascending column heights for ONE batched searchsorted over all columns
    # (replaces a Python loop of Za full-grid passes).
    h_rows = h_t.permute(1, 2, 0).reshape(-1, Zp).contiguous()  # [Y*X, Zp]
    targets = torch.as_tensor(agl_levels, dtype=torch.float64)
    target_rows = targets.unsqueeze(0).expand(h_rows.shape[0], -1).contiguous()

    # Insertion index = count of levels strictly below each target (== the old
    # per-level `np.sum(h < z_t, axis=0)`).
    idx = torch.searchsorted(h_rows, target_rows)  # [Y*X, Za]
    # First above-ground level per column: ascending order puts all sub-surface
    # (negative-AGL) levels first, so their count is the first real index.
    k0 = (h_rows < 0.0).sum(dim=1, keepdim=True)  # [Y*X, 1]
    lower = (idx - 1).clamp(min=0).maximum(k0).clamp(max=Zp - 2)  # [Y*X, Za]

    h_lo = torch.gather(h_rows, 1, lower)
    h_hi = torch.gather(h_rows, 1, lower + 1)
    frac = ((target_rows - h_lo) / (h_hi - h_lo).clamp(min=1e-6)).clamp(0.0, 1.0)

    lower_zyx = lower.reshape(Y, X, Za).permute(2, 0, 1).contiguous().numpy().astype(np.intp)
    frac_zyx = frac.reshape(Y, X, Za).permute(2, 0, 1).contiguous().numpy()
    return AglRegridWeights(
        lower=lower_zyx, frac=frac_zyx, flipped=flipped, n_source_levels=Zp
    )


def apply_agl_regrid(field_p: np.ndarray, weights: AglRegridWeights) -> np.ndarray:
    """Regrid ``[C, Zp, Y, X]`` pressure-level fields onto the weights' AGL grid.

    Returns ``[C, Za, Y, X]`` (surface first). One gather pair over all target
    levels at once, sharing the precomputed bracketing across channels.
    """
    if field_p.ndim != 4:
        raise ValueError(f"field_p must be [C, Zp, Y, X]; got {field_p.shape}")
    if field_p.shape[1] != weights.n_source_levels or field_p.shape[2:] != weights.lower.shape[1:]:
        raise ValueError(
            f"field_p {field_p.shape} does not match weights "
            f"(Zp={weights.n_source_levels}, YX={weights.lower.shape[1:]})"
        )
    C = field_p.shape[0]
    Zp = weights.n_source_levels
    # ascontiguousarray: torch cannot wrap negative-/zero-stride numpy views
    # (copies only when the input actually is such a view).
    f_t = torch.as_tensor(np.ascontiguousarray(field_p))

    # Gather in the ORIGINAL level orientation (index arithmetic instead of
    # flipping the big field tensor): ascending index i maps to original Zp-1-i.
    lower_t = torch.as_tensor(weights.lower)
    if weights.flipped:
        idx_lo = (Zp - 1) - lower_t
        idx_hi = idx_lo - 1
    else:
        idx_lo = lower_t
        idx_hi = idx_lo + 1
    expand = (C, -1, -1, -1)
    f_lo = torch.gather(f_t, 1, idx_lo.unsqueeze(0).expand(*expand))  # [C, Za, Y, X]
    f_hi = torch.gather(f_t, 1, idx_hi.unsqueeze(0).expand(*expand))
    frac_t = torch.as_tensor(weights.frac).to(dtype=f_t.dtype)
    out = torch.lerp(f_lo, f_hi, frac_t)
    return out.numpy() if isinstance(field_p, np.ndarray) else out


def regrid_columns_to_agl(
    field_p: np.ndarray,
    height_agl_p: np.ndarray,
    agl_levels: np.ndarray,
) -> np.ndarray:
    """Per-column linear interpolation of pressure-level fields onto an AGL grid.

    Convenience wrapper: ``apply_agl_regrid(field_p, compute_agl_regrid_weights(...))``.
    When regridding several tensors that share level heights (hour_start, hour_end,
    pressure), compute the weights once and call ``apply_agl_regrid`` per tensor.
    """
    if field_p.ndim != 4:
        raise ValueError(f"field_p must be [C, Zp, Y, X]; got {field_p.shape}")
    if height_agl_p.shape != field_p.shape[1:]:
        raise ValueError(
            f"height_agl_p {height_agl_p.shape} must match field_p[1:] {field_p.shape[1:]}"
        )
    return apply_agl_regrid(field_p, compute_agl_regrid_weights(height_agl_p, agl_levels))


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
