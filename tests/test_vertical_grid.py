"""Unit tests for the terrain-following (hybrid) vertical regrid (Finding 7)."""

from __future__ import annotations

import numpy as np
import pytest

from lpdm.vertical_grid import (
    default_agl_levels,
    regrid_columns_to_agl,
    slope_correct_w,
    terrain_gradient,
)


def test_default_agl_levels_ascending_and_covers_alt_max():
    lv = default_agl_levels(15000.0)
    assert lv[0] == 0.0
    assert np.all(np.diff(lv) > 0)
    assert lv[-1] >= 15000.0
    # A non-standard cap is appended, not dropped.
    lv2 = default_agl_levels(3300.0)
    assert lv2[-1] == pytest.approx(3300.0)


def _descending_pressure_column(terrain_m: float, agl_top: float = 12000.0):
    """One column of pressure-level AGL heights (store order: TOA first, so AGL
    descending), spanning sub-surface to high aloft for the given terrain."""
    asl = np.array([12000.0, 6000.0, 3000.0, 1500.0, 800.0, 400.0, 150.0, 50.0])
    return asl - terrain_m  # AGL, descending along axis 0


def test_linear_field_recovered_exactly():
    # A field that is exactly linear in AGL height must be reproduced at the targets.
    h = _descending_pressure_column(0.0)[:, None, None]  # [Zp,1,1]
    field = (3.0 * h + 7.0).astype("float64")[None]  # [1,Zp,1,1], linear in h
    targets = np.array([50.0, 400.0, 1500.0, 6000.0])
    out = regrid_columns_to_agl(field, h, targets)
    expected = 3.0 * targets + 7.0
    assert np.allclose(out[0, :, 0, 0], expected, atol=1e-6)


def test_subsurface_levels_excluded():
    # Terrain 1000 m: the bottom two levels (50 m, 150 m ASL) are below ground.
    # Their values are poisoned; near-surface targets must use the lowest
    # above-ground level, never the sub-surface ones.
    terrain = 1000.0
    h = _descending_pressure_column(terrain)  # [Zp], AGL; some entries negative
    assert (h < 0).sum() >= 1  # terrain pushes lower levels below ground
    field = np.full((1, h.size, 1, 1), 999.0)  # sub-surface sentinel
    lowest_above = np.where(h >= 0)[0]
    lowest_above_idx = lowest_above[np.argmax(h[lowest_above] * -1)]  # smallest positive AGL
    # give the lowest above-ground level a distinct, physical value
    field[0, :, 0, 0] = np.where(h >= 0, np.arange(h.size) + 10.0, 999.0)
    real_low = field[0, lowest_above_idx, 0, 0]
    out = regrid_columns_to_agl(field, h[:, None, None], np.array([0.0, 20.0, 40.0]))
    # All below the lowest real level -> constant-extrapolated from it, never 999.
    assert np.allclose(out[0, :, 0, 0], real_low)
    assert not np.any(out == 999.0)


def test_constant_extrapolation_above_top():
    h = _descending_pressure_column(0.0)
    field = np.arange(h.size, dtype="float64")[None, :, None, None]
    top_val = field[0, np.argmax(h), 0, 0]  # value at the highest level
    out = regrid_columns_to_agl(field, h[:, None, None], np.array([50000.0]))
    assert out[0, 0, 0, 0] == pytest.approx(top_val)


def test_orientation_invariance():
    # Ascending vs descending input ordering must give identical results.
    h_desc = _descending_pressure_column(200.0)
    f_desc = (2.0 * h_desc + 1.0)[None, :, None, None]
    targets = np.array([0.0, 100.0, 500.0, 2000.0])
    out_desc = regrid_columns_to_agl(f_desc, h_desc[:, None, None], targets)
    out_asc = regrid_columns_to_agl(
        f_desc[:, ::-1], h_desc[::-1, None, None], targets
    )
    assert np.allclose(out_desc, out_asc)


def test_terrain_gradient_planar_slope():
    # A plane sloping 100 m per degree east at 45N.
    lat = np.array([46.0, 45.0, 44.0])  # descending, as ERA5
    lon = np.array([0.0, 1.0, 2.0])
    terrain = 100.0 * lon[None, :] * np.ones((3, 3))
    dhdx, dhdy = terrain_gradient(terrain, lat, lon)
    # dh/dx scales with cos(lat), so check row-by-row against each latitude.
    expected = 100.0 / (111320.0 * np.cos(np.deg2rad(lat)))  # [Y]
    assert np.allclose(dhdx, expected[:, None], atol=1e-9)
    assert np.allclose(dhdy, 0.0, atol=1e-12)


def test_slope_correct_w_taper_and_flat_terrain():
    agl = np.array([0.0, 1000.0, 10000.0])
    shape = (3, 2, 2)
    w = np.zeros(shape)
    u = np.ones(shape)
    v = np.zeros(shape)
    dhdx = np.full((2, 2), 0.1)  # steep 0.1 m/m slope
    dhdy = np.zeros((2, 2))
    out = slope_correct_w(w, u, v, dhdx, dhdy, agl, z_top_m=10000.0)
    # Surface: full correction w - u*dhdx = -0.1
    assert np.allclose(out[0], -0.1)
    # Model top: taper -> 0, w unchanged
    assert np.allclose(out[2], 0.0)
    # Flat terrain leaves w untouched at all levels.
    flat = slope_correct_w(w, u, v, np.zeros((2, 2)), np.zeros((2, 2)), agl, 10000.0)
    assert np.allclose(flat, w)


def test_regrid_rejects_bad_shapes():
    with pytest.raises(ValueError):
        regrid_columns_to_agl(np.zeros((2, 3)), np.zeros((3, 1, 1)), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        regrid_columns_to_agl(np.zeros((1, 3, 2, 2)), np.zeros((3, 2, 2)), np.array([1.0, 0.0]))
