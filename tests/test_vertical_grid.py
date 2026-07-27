"""Unit tests for the terrain-following (hybrid) vertical regrid (Finding 7)."""

from __future__ import annotations

import numpy as np
import pytest

from lpdm.vertical_grid import (
    default_agl_levels,
    model_level_pressure_pa,
    regrid_columns_to_agl,
    slope_correct_w,
    stretched_agl_levels,
    terrain_gradient,
)

_GRAVITY = 9.80665
_RD = 287.05


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
    out_asc = regrid_columns_to_agl(f_desc[:, ::-1], h_desc[::-1, None, None], targets)
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


def test_weights_split_matches_wrapper_and_is_shareable():
    # compute-once/apply-many must equal the one-shot wrapper, for multiple fields
    # sharing the same level heights (the met-window usage pattern).
    from lpdm.vertical_grid import apply_agl_regrid, compute_agl_regrid_weights

    rng = np.random.default_rng(7)
    h = _descending_pressure_column(300.0)[:, None, None] + rng.normal(
        0.0, 5.0, size=(8, 3, 4)
    )  # [Zp, Y, X], descending with jitter
    targets = np.array([0.0, 40.0, 500.0, 2000.0, 8000.0])
    w = compute_agl_regrid_weights(h, targets)
    for seed in (0, 1):
        f = np.random.default_rng(seed).normal(size=(3, 8, 3, 4))
        assert np.allclose(apply_agl_regrid(f, w), regrid_columns_to_agl(f, h, targets))


# --- model_level_pressure_pa (hydrostatic reconstruction) --------------------


def _isothermal_column(z_agl_m, terrain_m=0.0, t_k=250.0, ps_pa=101325.0):
    """Build a single-column model-level input for an isothermal, dry atmosphere.

    Returns (geopotential [Z,1,1], surface_geopotential [1,1], surface_pressure
    [1,1], temperature [Z,1,1]).
    """
    z_agl_m = np.asarray(z_agl_m, dtype=np.float64)
    phi_s = np.full((1, 1), terrain_m * _GRAVITY)
    phi = (terrain_m + z_agl_m)[:, None, None] * _GRAVITY
    temp = np.full_like(phi, t_k)
    ps = np.full((1, 1), ps_pa)
    return phi, phi_s, ps, temp


def test_model_level_pressure_matches_isothermal_analytic():
    # Isothermal, dry: p(z) = ps * exp(-(phi - phi_s)/(Rd*T)) exactly.
    z = np.array([10.0, 100.0, 1000.0, 5000.0])
    phi, phi_s, ps, temp = _isothermal_column(z, terrain_m=200.0, t_k=250.0)
    p = model_level_pressure_pa(phi, phi_s, ps, temp, specific_humidity=None)
    expected = 101325.0 * np.exp(-(phi[:, 0, 0] - phi_s[0, 0]) / (_RD * 250.0))
    assert np.allclose(p[:, 0, 0], expected, rtol=1e-10)


def test_model_level_pressure_lowest_level_near_surface():
    # The lowest level (~10 m AGL) sits just below surface pressure.
    z = np.array([10.0, 200.0, 2000.0])
    phi, phi_s, ps, temp = _isothermal_column(z, t_k=280.0, ps_pa=100000.0)
    p = model_level_pressure_pa(phi, phi_s, ps, temp)
    assert p[0, 0, 0] < 100000.0
    assert p[0, 0, 0] > 0.998 * 100000.0  # within ~0.2% of ps at 10 m


def test_model_level_pressure_moisture_raises_pressure_aloft():
    # Virtual temperature > T slows the pressure decay, so a moist column has
    # HIGHER pressure at a given geopotential than a dry one.
    z = np.array([10.0, 1000.0, 5000.0])
    phi, phi_s, ps, temp = _isothermal_column(z, t_k=290.0)
    q = np.full_like(phi, 0.015)  # 15 g/kg
    p_dry = model_level_pressure_pa(phi, phi_s, ps, temp, None)
    p_moist = model_level_pressure_pa(phi, phi_s, ps, temp, q)
    assert np.all(p_moist[1:] > p_dry[1:])


def test_model_level_pressure_order_invariant():
    # Result is returned in input order regardless of level ordering.
    z = np.array([10.0, 100.0, 1000.0, 5000.0])
    phi, phi_s, ps, temp = _isothermal_column(z, t_k=260.0)
    p_up = model_level_pressure_pa(phi, phi_s, ps, temp)  # surface-first
    rev = slice(None, None, -1)
    p_dn = model_level_pressure_pa(phi[rev], phi_s, ps, temp[rev])  # top-first
    assert np.allclose(p_up, p_dn[rev], rtol=1e-12)


def test_model_level_pressure_decreases_with_height():
    z = np.array([10.0, 500.0, 3000.0, 12000.0])
    phi, phi_s, ps, temp = _isothermal_column(z, t_k=245.0)
    p = model_level_pressure_pa(phi, phi_s, ps, temp)
    assert np.all(np.diff(p[:, 0, 0]) < 0)


# --- stretched_agl_levels (configurable vertical grid) -----------------------


@pytest.mark.parametrize("n_levels", [2, 5, 23, 40, 60, 137])
def test_stretched_grid_spans_domain_with_exact_level_count(n_levels):
    lv = stretched_agl_levels(n_levels, 15000.0)
    assert lv.size == n_levels
    assert lv[0] == 0.0
    assert lv[-1] == 15000.0
    assert np.all(np.diff(lv) > 0)


def test_stretched_grid_honours_first_layer_thickness():
    for dz0 in (5.0, 10.0, 25.0):
        lv = stretched_agl_levels(40, 15000.0, first_layer_m=dz0)
        assert lv[1] == pytest.approx(dz0, rel=1e-9)


def test_stretched_grid_layers_grow_monotonically():
    """Geometric stretch: every layer is thicker than the one below it."""
    lv = stretched_agl_levels(50, 15000.0)
    thicknesses = np.diff(lv)
    assert np.all(np.diff(thicknesses) > 0)
    # constant ratio (the defining property), except the top layer which is
    # pinned to alt_max exactly.
    ratios = thicknesses[1:-1] / thicknesses[:-2]
    assert np.allclose(ratios, ratios[0], rtol=1e-6)


def test_more_levels_resolves_more_of_the_boundary_layer():
    """The point of the knob: raising n_levels buys near-surface resolution."""
    below_1500 = [int((stretched_agl_levels(n, 15000.0) <= 1500.0).sum()) for n in (23, 40, 60)]
    assert below_1500 == sorted(below_1500)
    assert below_1500[0] < below_1500[-1]
    # 23 stretched levels reproduce the hand-tuned default ladder's character
    assert abs(below_1500[0] - int((default_agl_levels(15000.0) <= 1500.0).sum())) <= 2


def test_stretched_grid_rejects_impossible_geometry():
    with pytest.raises(ValueError, match="n_levels must be >= 2"):
        stretched_agl_levels(1, 15000.0)
    with pytest.raises(ValueError, match="alt_max_m must be > 0"):
        stretched_agl_levels(10, 0.0)
    with pytest.raises(ValueError, match="first_layer_m must be > 0"):
        stretched_agl_levels(10, 15000.0, first_layer_m=0.0)
    # 200 uniform 10 m layers already overshoot a 1000 m top
    with pytest.raises(ValueError, match="too thick for n_levels"):
        stretched_agl_levels(200, 1000.0, first_layer_m=10.0)
