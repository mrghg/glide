"""Unit tests for the footprint comparison utilities.

Covers `to_stilt_surface_footprint` (unit conversion + surface-layer integration)
and `regrid_conservative` (area-weighted mass-conservative regridding).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from lpdm.comparison import (
    DRY_AIR_M_KG_MOL,
    R_DRY_AIR_J_KG_K,
    regrid_conservative,
    surface_air_density_from_met,
    to_stilt_surface_footprint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_footprint(
    *,
    z_edges_m: np.ndarray,
    n_time: int = 3,
    n_lat: int = 4,
    n_lon: int = 4,
    fill: float = 0.0,
) -> xr.DataArray:
    """Build a raw footprint DataArray matching what main.py persists."""

    n_z = z_edges_m.size - 1
    z_centres = 0.5 * (z_edges_m[:-1] + z_edges_m[1:])
    arr = np.full((n_time, n_z, n_lat, n_lon), fill, dtype=np.float64)
    da = xr.DataArray(
        arr,
        dims=("time_ago", "z_bin", "latitude", "longitude"),
        coords={
            "time_ago": np.arange(n_time, dtype=np.int64),
            "z_bin": z_centres,
            "z_bottom_m": ("z_bin", z_edges_m[:-1]),
            "z_top_m": ("z_bin", z_edges_m[1:]),
            "latitude": np.linspace(-1.0, 1.0, n_lat),
            "longitude": np.linspace(-1.0, 1.0, n_lon),
        },
        name="footprint",
    )
    return da


# ---------------------------------------------------------------------------
# (a) STILT-style unit conversion
# ---------------------------------------------------------------------------


def test_to_stilt_full_overlap_thin_surface_bin() -> None:
    """When a single z-bin matches the surface layer exactly, conversion is exact and direct."""

    raw = _make_raw_footprint(z_edges_m=np.array([0.0, 40.0]), n_time=2)
    raw.values[:] = 0.0
    raw.values[1, 0, 2, 1] = 5.0  # one cell with 5 s residence

    stilt = to_stilt_surface_footprint(
        raw,
        surface_layer_depth_m=40.0,
        air_density_kg_m3=1.225,
        integrate_time=True,
    )

    assert set(stilt.dims) == {"latitude", "longitude"}
    expected = 5.0 * (DRY_AIR_M_KG_MOL / (40.0 * 1.225))
    assert np.isclose(stilt.values[2, 1], expected, rtol=1e-12)
    assert np.isclose(stilt.values.sum(), expected)


def test_to_stilt_partial_overlap_with_thicker_bin() -> None:
    """Bins that span beyond the surface layer should be credited only by the overlap fraction."""

    raw = _make_raw_footprint(z_edges_m=np.array([0.0, 1000.0]))  # one big bin
    raw.values[0, 0, 0, 0] = 100.0  # 100 s residence over 0-1000 m

    stilt = to_stilt_surface_footprint(
        raw,
        surface_layer_depth_m=40.0,
        air_density_kg_m3=1.225,
        integrate_time=True,
    )

    # Overlap fraction = 40 / 1000 = 0.04 → effective surface residence = 4 s
    expected = (100.0 * 0.04) * (DRY_AIR_M_KG_MOL / (40.0 * 1.225))
    assert np.isclose(stilt.values[0, 0], expected, rtol=1e-12)


def test_to_stilt_time_integration_flag() -> None:
    """integrate_time=False should keep the time axis."""

    raw = _make_raw_footprint(z_edges_m=np.array([0.0, 40.0]), n_time=3)
    raw.values[0, 0, 0, 0] = 1.0
    raw.values[1, 0, 0, 0] = 2.0
    raw.values[2, 0, 0, 0] = 4.0

    integrated = to_stilt_surface_footprint(
        raw, surface_layer_depth_m=40.0, air_density_kg_m3=1.0, integrate_time=True,
    )
    by_time = to_stilt_surface_footprint(
        raw, surface_layer_depth_m=40.0, air_density_kg_m3=1.0, integrate_time=False,
    )

    assert set(by_time.dims) == {"time_ago", "latitude", "longitude"}
    assert np.isclose(by_time.values[:, 0, 0].sum(), integrated.values[0, 0])
    assert np.isclose(by_time.values[2, 0, 0], 4.0 * DRY_AIR_M_KG_MOL / 40.0)


def test_to_stilt_rejects_missing_z_edges_coords() -> None:
    """The converter should fail loud if z_bottom_m / z_top_m coords are missing."""

    raw = _make_raw_footprint(z_edges_m=np.array([0.0, 40.0]))
    bad = raw.drop_vars("z_bottom_m")
    with pytest.raises(ValueError, match="z_bottom_m"):
        to_stilt_surface_footprint(bad, surface_layer_depth_m=40.0, air_density_kg_m3=1.0)


def test_to_stilt_rejects_nonpositive_surface_depth() -> None:
    raw = _make_raw_footprint(z_edges_m=np.array([0.0, 40.0]))
    with pytest.raises(ValueError, match="must be > 0"):
        to_stilt_surface_footprint(raw, surface_layer_depth_m=0.0, air_density_kg_m3=1.0)


def test_to_stilt_records_conversion_metadata() -> None:
    raw = _make_raw_footprint(z_edges_m=np.array([0.0, 40.0]))
    stilt = to_stilt_surface_footprint(
        raw, surface_layer_depth_m=40.0, air_density_kg_m3=1.225,
    )
    assert stilt.attrs["units"] == "m**2 s mol**-1"
    assert stilt.attrs["stilt_surface_layer_depth_m"] == 40.0
    assert stilt.attrs["stilt_air_density_kg_m3"] == 1.225
    assert "Lin et al. 2003" in stilt.attrs["conversion_reference"]


# ---------------------------------------------------------------------------
# (c) Conservative regridder
# ---------------------------------------------------------------------------


def _make_field(lat: np.ndarray, lon: np.ndarray, fill_fn) -> xr.DataArray:
    """Build a 2D DataArray from coordinate arrays and a (lat, lon) → value function."""

    arr = np.array([[fill_fn(la, lo) for lo in lon] for la in lat], dtype=float)
    return xr.DataArray(
        arr,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="field",
    )


def test_regrid_identity_when_grids_match() -> None:
    """Regridding to the same grid should return values unchanged (within float precision)."""

    lat = np.linspace(0.0, 4.0, 5)
    lon = np.linspace(0.0, 4.0, 5)
    src = _make_field(lat, lon, lambda la, lo: la + 10.0 * lo)

    out = regrid_conservative(src, target_latitude=lat, target_longitude=lon)

    assert out.dims == ("latitude", "longitude")
    assert np.allclose(out.values, src.values, atol=1e-12)


def test_regrid_coarsen_preserves_total() -> None:
    """Regridding from fine to coarse grid (target ⊇ source) must preserve total sum."""

    src_lat = np.linspace(0.5, 3.5, 4)  # 4 cells of 1° each, centres
    src_lon = np.linspace(0.5, 3.5, 4)
    src = _make_field(src_lat, src_lon, lambda la, lo: 1.0)  # uniform 1.0

    # Coarsen to 2x2 cells of 2° each
    tgt_lat = np.array([1.0, 3.0])
    tgt_lon = np.array([1.0, 3.0])
    out = regrid_conservative(src, target_latitude=tgt_lat, target_longitude=tgt_lon)

    src_total_area = (
        np.sin(np.deg2rad(np.array([1, 2, 3, 4]))) - np.sin(np.deg2rad(np.array([0, 1, 2, 3])))
    ).sum() * np.deg2rad(4.0)  # sum lat × total lon span
    src_total = src.values.sum()  # 16 (uniform 1.0 over 4×4)

    # Total preserved over the full domain (target covers src exactly).
    assert np.isclose(out.values.sum(), src_total, rtol=1e-12)


def test_regrid_refine_preserves_total() -> None:
    """Regridding from coarse to fine grid should also preserve total sum."""

    src_lat = np.array([1.0, 3.0])  # 2 cells of 2° each
    src_lon = np.array([1.0, 3.0])
    src = _make_field(src_lat, src_lon, lambda la, lo: 5.0)  # 4 cells of value 5 → total 20

    tgt_lat = np.linspace(0.5, 3.5, 4)
    tgt_lon = np.linspace(0.5, 3.5, 4)
    out = regrid_conservative(src, target_latitude=tgt_lat, target_longitude=tgt_lon)

    assert np.isclose(out.values.sum(), src.values.sum(), rtol=1e-12)


def test_regrid_redistributes_to_overlapping_targets() -> None:
    """A localised source pulse should split between the target cells it overlaps."""

    src_lat = np.array([10.0])
    src_lon = np.array([10.0])
    src = xr.DataArray(
        np.array([[100.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": src_lat, "longitude": src_lon},
    )

    # Target cells just smaller than source, so source straddles two target cells equally
    tgt_lat = np.array([9.75, 10.25])
    tgt_lon = np.array([10.0])
    out = regrid_conservative(src, target_latitude=tgt_lat, target_longitude=tgt_lon)

    # Each half should get ~50 (proportional to lat overlap).
    assert np.isclose(out.values.sum(), 100.0, rtol=1e-9)
    # Each target gets close to half (slight asymmetry from sin(lat) curvature is tiny here).
    assert abs(out.values[0, 0] - 50.0) < 1.0
    assert abs(out.values[1, 0] - 50.0) < 1.0


def test_regrid_preserves_extra_dimensions() -> None:
    """Regridding should preserve leading dimensions like time_ago, z_bin."""

    src_lat = np.array([0.5, 1.5, 2.5, 3.5])
    src_lon = np.array([0.5, 1.5, 2.5, 3.5])
    arr = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)
    src = xr.DataArray(
        arr,
        dims=("time_ago", "z_bin", "latitude", "longitude"),
        coords={
            "time_ago": np.array([0, 1], dtype=np.int64),
            "z_bin": np.array([0.0, 1.0, 2.0]),
            "latitude": src_lat,
            "longitude": src_lon,
        },
    )

    # Coarsen 4x4 → 2x2 (target covers same extent as source: 0-4°).
    tgt_lat = np.array([1.0, 3.0])
    tgt_lon = np.array([1.0, 3.0])
    out = regrid_conservative(src, target_latitude=tgt_lat, target_longitude=tgt_lon)

    assert out.dims == ("time_ago", "z_bin", "latitude", "longitude")
    assert out.shape == (2, 3, 2, 2)
    for t in range(2):
        for z in range(3):
            assert np.isclose(out.values[t, z].sum(), src.values[t, z].sum(), rtol=1e-12)


def test_regrid_lat_cosine_area_factor() -> None:
    """Total preservation should hold even at high latitudes where cos-lat matters."""

    # Source: 4x4 cells at high latitude (~60°) where cos(lat) is meaningfully smaller than 1.
    src_lat = np.array([60.5, 61.5, 62.5, 63.5])
    src_lon = np.array([0.5, 1.5, 2.5, 3.5])
    src = _make_field(src_lat, src_lon, lambda la, lo: 1.0)

    # Coarsen 4x4 → 2x2 (target spans 60-64° in lat, 0-4° in lon, same as source).
    tgt_lat = np.array([61.0, 63.0])
    tgt_lon = np.array([1.0, 3.0])
    out = regrid_conservative(src, target_latitude=tgt_lat, target_longitude=tgt_lon)

    assert np.isclose(out.values.sum(), src.values.sum(), rtol=1e-12)


def test_regrid_zero_outside_source_extent() -> None:
    """Target cells with no source overlap should be zero."""

    src_lat = np.array([0.5])
    src_lon = np.array([0.5])
    src = xr.DataArray(
        np.array([[42.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": src_lat, "longitude": src_lon},
    )

    tgt_lat = np.array([10.0, 20.0])  # Far from source
    tgt_lon = np.array([10.0, 20.0])
    out = regrid_conservative(src, target_latitude=tgt_lat, target_longitude=tgt_lon)

    assert np.allclose(out.values, 0.0)


def test_regrid_rejects_non_ascending_centres() -> None:
    src = xr.DataArray(
        np.zeros((3, 3)),
        dims=("latitude", "longitude"),
        coords={"latitude": np.array([0.0, 1.0, 2.0]), "longitude": np.array([0.0, 1.0, 2.0])},
    )
    with pytest.raises(ValueError, match="ascending"):
        regrid_conservative(
            src,
            target_latitude=np.array([2.0, 1.0, 0.0]),
            target_longitude=np.array([0.0, 1.0, 2.0]),
        )


# ---------------------------------------------------------------------------
# (b) Surface air density helper (F10, audit 2026-05-30)
# ---------------------------------------------------------------------------


def _make_minimal_met(sp_pa: float = 101325.0, t_lowest_k: float = 280.0) -> xr.Dataset:
    """Tiny met fixture with the variables `surface_air_density_from_met` needs."""

    n_lat, n_lon, n_lev = 3, 4, 2
    lat = np.linspace(45.0, 50.0, n_lat)
    lon = np.linspace(-2.0, 2.0, n_lon)
    # Level coord in hPa, descending altitude (ARCO ERA5 convention) — so
    # argmax(level) picks the LOWEST altitude (highest pressure = surface).
    level = np.array([500.0, 1000.0])  # 500 hPa (aloft), 1000 hPa (surface)
    t = np.full((n_lev, n_lat, n_lon), 260.0)  # aloft default
    t[1, :, :] = t_lowest_k  # lowest level = surface proxy
    sp = np.full((n_lat, n_lon), sp_pa)
    return xr.Dataset(
        {
            "surface_pressure": (("latitude", "longitude"), sp, {"units": "Pa"}),
            "temperature": (("level", "latitude", "longitude"), t, {"units": "K"}),
        },
        coords={
            "level": ("level", level, {"units": "hPa"}),
            "latitude": ("latitude", lat),
            "longitude": ("longitude", lon),
        },
    )


def test_surface_air_density_matches_ideal_gas() -> None:
    """ρ = sp / (R_d · T_surface) for the lowest level."""

    sp_pa, t_k = 101325.0, 288.0
    met = _make_minimal_met(sp_pa=sp_pa, t_lowest_k=t_k)
    rho = surface_air_density_from_met(met)
    expected = sp_pa / (R_DRY_AIR_J_KG_K * t_k)
    assert rho.dims == ("latitude", "longitude")
    assert np.allclose(rho.values, expected)
    assert rho.attrs["units"] == "kg m**-3"


def test_surface_air_density_spatial_variation() -> None:
    """A spatially-varying surface pressure produces a varying ρ field."""

    met = _make_minimal_met()
    # Inject a 1% pressure gradient across longitude.
    sp = met["surface_pressure"].values.copy()
    sp[:, -1] *= 1.01  # last lon column = 1.01× the rest
    met["surface_pressure"][:] = sp

    rho = surface_air_density_from_met(met)
    base = float(rho.values[0, 0])
    edge = float(rho.values[0, -1])
    assert abs(edge - 1.01 * base) / base < 1e-6


def test_surface_air_density_rejects_missing_variable() -> None:
    """Clear error if the requested variable name isn't in the dataset."""

    met = _make_minimal_met()
    with pytest.raises(KeyError, match="surface-pressure"):
        surface_air_density_from_met(met, sp_var="not_a_variable")
    with pytest.raises(KeyError, match="temperature"):
        surface_air_density_from_met(met, t_var="not_a_variable")


def test_surface_air_density_feeds_to_stilt_surface_footprint() -> None:
    """End-to-end: the rho DataArray plugs into `to_stilt_surface_footprint`
    without complaint and is recorded in the attrs as a DataArray."""

    met = _make_minimal_met(sp_pa=101325.0, t_lowest_k=288.0)
    rho = surface_air_density_from_met(met)

    # Build a raw footprint matching rho's spatial grid (3x4).
    raw = _make_raw_footprint(
        z_edges_m=np.array([0.0, 40.0, 200.0]),
        n_lat=rho.sizes["latitude"], n_lon=rho.sizes["longitude"],
        fill=1.0,
    )
    raw = raw.assign_coords(latitude=rho["latitude"].values, longitude=rho["longitude"].values)

    stilt = to_stilt_surface_footprint(
        raw, surface_layer_depth_m=40.0, air_density_kg_m3=rho,
    )
    assert stilt.attrs["stilt_air_density_kg_m3"] == "DataArray"
    assert stilt.dims == ("latitude", "longitude")
