from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import torch
import xarray as xr

from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, SpatialBounds, TimeBounds


class _InMemoryArcoReader(ArcoEra5ZarrReader):
    def __init__(self, dataset: xr.Dataset) -> None:
        super().__init__(zarr_store="in-memory", device="cpu", dtype=torch.float64)
        self._dataset = dataset

    def _open_dataset(self) -> xr.Dataset:
        return self._dataset


def _build_mock_era5_dataset() -> xr.Dataset:
    times = np.array([
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-01T01:00:00"),
    ])
    levels = np.array([900.0, 1000.0], dtype=np.float64)  # hPa
    lat = np.array([10.0, 11.0], dtype=np.float64)
    lon = np.array([20.0, 21.0], dtype=np.float64)

    shape_4d = (times.size, levels.size, lat.size, lon.size)
    shape_3d = (times.size, lat.size, lon.size)

    u = np.full(shape_4d, 5.0, dtype=np.float64)
    v = np.full(shape_4d, -2.0, dtype=np.float64)
    omega = np.full(shape_4d, -0.10, dtype=np.float64)  # Pa/s
    t = np.full(shape_4d, 280.0, dtype=np.float64)  # K
    blh = np.full(shape_3d, 400.0, dtype=np.float64)
    sp = np.full(shape_3d, 101325.0, dtype=np.float64)
    ustar = np.full(shape_3d, 0.4, dtype=np.float64)  # m/s
    # 100 W/m^2 accumulated over 3600 s = 360_000 J/m^2
    shf_accumulated = np.full(shape_3d, 360_000.0, dtype=np.float64)

    # Build geopotential fields so the implied AGL is exactly [900, 1000] m.
    z_sfc = np.full(shape_3d, 150.0 * 9.80665, dtype=np.float64)  # 150 m AMSL in m^2 s^-2
    z = np.empty(shape_4d, dtype=np.float64)
    z[:, 0, :, :] = z_sfc + (900.0 * 9.80665)
    z[:, 1, :, :] = z_sfc + (1000.0 * 9.80665)

    ds = xr.Dataset(
        data_vars={
            "u_component_of_wind": (("time", "level", "latitude", "longitude"), u),
            "v_component_of_wind": (("time", "level", "latitude", "longitude"), v),
            "vertical_velocity": (("time", "level", "latitude", "longitude"), omega),
            "temperature": (("time", "level", "latitude", "longitude"), t),
            "boundary_layer_height": (("time", "latitude", "longitude"), blh),
            "surface_pressure": (("time", "latitude", "longitude"), sp),
            "geopotential": (("time", "level", "latitude", "longitude"), z),
            "geopotential_at_surface": (("time", "latitude", "longitude"), z_sfc),
            "friction_velocity": (("time", "latitude", "longitude"), ustar),
            "surface_sensible_heat_flux": (("time", "latitude", "longitude"), shf_accumulated),
        },
        coords={
            "time": times,
            "level": levels,
            "latitude": lat,
            "longitude": lon,
        },
    )

    ds["u_component_of_wind"].attrs["units"] = "m s**-1"
    ds["v_component_of_wind"].attrs["units"] = "m s**-1"
    ds["vertical_velocity"].attrs["units"] = "Pa s**-1"
    ds["temperature"].attrs["units"] = "K"
    ds["boundary_layer_height"].attrs["units"] = "m"
    ds["surface_pressure"].attrs["units"] = "Pa"
    ds["geopotential"].attrs["units"] = "m**2 s**-2"
    ds["geopotential_at_surface"].attrs["units"] = "m**2 s**-2"
    ds["friction_velocity"].attrs["units"] = "m s**-1"
    ds["surface_sensible_heat_flux"].attrs["units"] = "J m**-2"
    ds["level"].attrs["units"] = "hPa"

    return ds


def _build_mock_era5_dataset_360_lon() -> xr.Dataset:
    ds = _build_mock_era5_dataset()
    ds = ds.assign_coords(longitude=np.array([237.5, 238.5], dtype=np.float64))
    return ds


def test_fetch_hourly_window_includes_surface_pressure_and_converts_w() -> None:
    ds = _build_mock_era5_dataset()
    reader = _InMemoryArcoReader(ds)

    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=19.5,
            lon_max=21.5,
            lat_min=9.5,
            lat_max=11.5,
            z_min=850.0,
            z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    result = reader.fetch_hourly_window(request)

    # Channel order is [u, v, w, blh, sp].
    assert result.hour_start.shape == (5, 2, 2, 2)
    assert result.hour_end.shape == (5, 2, 2, 2)

    # Surface pressure should be broadcast over the vertical dimension.
    sp_channel = result.hour_start[4]
    assert torch.allclose(sp_channel[0], sp_channel[1])
    assert torch.allclose(sp_channel, torch.full_like(sp_channel, 101325.0))

    # Omega (Pa/s) should be converted to geometric vertical velocity (m/s).
    # w = -(R_d * T / (g * p)) * omega
    rd = 287.05
    g = 9.80665
    omega = -0.10
    expected_w_lvl0 = -(rd * 280.0 / (g * (900.0 * 100.0))) * omega
    expected_w_lvl1 = -(rd * 280.0 / (g * (1000.0 * 100.0))) * omega

    w_channel = result.hour_start[2]
    assert torch.allclose(w_channel[0], torch.full_like(w_channel[0], expected_w_lvl0), atol=1e-12)
    assert torch.allclose(w_channel[1], torch.full_like(w_channel[1], expected_w_lvl1), atol=1e-12)

    assert result.metadata.variable_units["sp"] == "Pa"
    assert np.allclose(result.metadata.level, np.array([900.0, 1000.0]))
    assert np.allclose(result.metadata.pressure_level_hpa, np.array([900.0, 1000.0]))


def test_fetch_hourly_window_rejects_unknown_w_units() -> None:
    ds = _build_mock_era5_dataset()
    ds["vertical_velocity"].attrs["units"] = "furlong/day"

    reader = _InMemoryArcoReader(ds)
    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=19.5,
            lon_max=21.5,
            lat_min=9.5,
            lat_max=11.5,
            z_min=850.0,
            z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    try:
        reader.fetch_hourly_window(request)
        raise AssertionError("Expected ValueError for unsupported vertical velocity units")
    except ValueError as exc:
        assert "Unsupported units" in str(exc)


def test_fetch_hourly_window_handles_negative_lon_request_with_360_dataset() -> None:
    ds = _build_mock_era5_dataset_360_lon()
    reader = _InMemoryArcoReader(ds)

    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=-122.6,
            lon_max=-121.4,
            lat_min=9.5,
            lat_max=11.5,
            z_min=850.0,
            z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    result = reader.fetch_hourly_window(request)
    assert result.hour_start.shape == (5, 2, 2, 2)
    assert result.hour_end.shape == (5, 2, 2, 2)


def test_get_time_coverage_returns_dataset_bounds() -> None:
    ds = _build_mock_era5_dataset()
    reader = _InMemoryArcoReader(ds)

    start, end = reader.get_time_coverage()

    assert start == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)


def test_fetch_hourly_window_rejects_partial_nan_agl_cells() -> None:
    ds = _build_mock_era5_dataset()
    geopotential = np.asarray(ds["geopotential"].values).copy()
    geopotential_at_surface = np.asarray(ds["geopotential_at_surface"].values).copy()
    geopotential[0, :, 0, 0] = np.nan
    geopotential_at_surface[0, 0, 0] = np.nan
    ds["geopotential"] = (("time", "level", "latitude", "longitude"), geopotential)
    ds["geopotential_at_surface"] = (("time", "latitude", "longitude"), geopotential_at_surface)
    ds["geopotential"].attrs["units"] = "m**2 s**-2"
    ds["geopotential_at_surface"].attrs["units"] = "m**2 s**-2"

    reader = _InMemoryArcoReader(ds)
    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=19.5,
            lon_max=21.5,
            lat_min=9.5,
            lat_max=11.5,
            z_min=850.0,
            z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    try:
        reader.fetch_hourly_window(request)
        raise AssertionError("Expected ValueError for partial-NaN AGL fields")
    except ValueError as exc:
        assert "contain non-finite values" in str(exc)


def test_fetch_hourly_window_rejects_all_nan_agl_fields() -> None:
    ds = _build_mock_era5_dataset()
    ds["geopotential"] = (("time", "level", "latitude", "longitude"), np.full_like(ds["geopotential"].values, np.nan))
    ds["geopotential_at_surface"] = (("time", "latitude", "longitude"), np.full_like(ds["geopotential_at_surface"].values, np.nan))
    ds["geopotential"].attrs["units"] = "m**2 s**-2"
    ds["geopotential_at_surface"].attrs["units"] = "m**2 s**-2"

    reader = _InMemoryArcoReader(ds)
    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=19.5,
            lon_max=21.5,
            lat_min=9.5,
            lat_max=11.5,
            z_min=850.0,
            z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    try:
        reader.fetch_hourly_window(request)
        raise AssertionError("Expected ValueError for all-NaN AGL fields")
    except ValueError as exc:
        assert "contain non-finite values" in str(exc)


def test_hourly_met_tensors_channel_accessors_match_positional() -> None:
    """channel() / channels() should agree with positional indexing and respect order."""

    ds = _build_mock_era5_dataset()
    reader = _InMemoryArcoReader(ds)
    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=19.5, lon_max=21.5, lat_min=9.5, lat_max=11.5, z_min=850.0, z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    result = reader.fetch_hourly_window(request)
    assert result.channel_names == ("u", "v", "w", "blh", "sp")

    # Single-channel access matches positional index.
    sp_start, sp_end = result.channel("sp")
    assert torch.equal(sp_start, result.hour_start[4])
    assert torch.equal(sp_end, result.hour_end[4])

    # Multi-channel access stacks in the requested order, not the underlying order.
    stacked_start, stacked_end = result.channels("w", "u", "blh")
    assert torch.equal(stacked_start[0], result.hour_start[2])
    assert torch.equal(stacked_start[1], result.hour_start[0])
    assert torch.equal(stacked_start[2], result.hour_start[3])
    assert stacked_end.shape == (3, *result.hour_end.shape[1:])


def test_fetch_hourly_window_includes_ustar_and_deaccumulates_shf() -> None:
    """Extending channel_names should bring ustar/shf into the tensor; J/m^2 -> W/m^2."""

    ds = _build_mock_era5_dataset()

    class _ExtendedReader(_InMemoryArcoReader):
        pass

    reader = _ExtendedReader(ds)
    reader.channel_names = ("u", "v", "w", "blh", "sp", "ustar", "shf")

    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=19.5, lon_max=21.5, lat_min=9.5, lat_max=11.5, z_min=850.0, z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    result = reader.fetch_hourly_window(request)

    assert result.channel_names == ("u", "v", "w", "blh", "sp", "ustar", "shf")
    assert result.hour_start.shape == (7, 2, 2, 2)

    ustar_start, _ = result.channel("ustar")
    assert torch.allclose(ustar_start, torch.full_like(ustar_start, 0.4))

    # Mock SHF is 360_000 J/m^2 accumulated over 3600 s; expect 100 W/m^2.
    shf_start, _ = result.channel("shf")
    assert torch.allclose(shf_start, torch.full_like(shf_start, 100.0))


def test_fetch_hourly_window_passes_through_instantaneous_shf() -> None:
    """If SHF is supplied as W/m^2 the de-accumulation step should be a no-op."""

    ds = _build_mock_era5_dataset()
    ds["surface_sensible_heat_flux"] = ds["surface_sensible_heat_flux"] / 3600.0
    ds["surface_sensible_heat_flux"].attrs["units"] = "W m**-2"

    class _ExtendedReader(_InMemoryArcoReader):
        pass

    reader = _ExtendedReader(ds)
    reader.channel_names = ("u", "v", "w", "blh", "sp", "ustar", "shf")

    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=19.5, lon_max=21.5, lat_min=9.5, lat_max=11.5, z_min=850.0, z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    result = reader.fetch_hourly_window(request)

    shf_start, _ = result.channel("shf")
    assert torch.allclose(shf_start, torch.full_like(shf_start, 100.0))


def test_reader_rejects_channel_names_with_no_variable_map_entry() -> None:
    try:
        ArcoEra5ZarrReader(
            zarr_store="in-memory",
            channel_names=("u", "v", "w", "blh", "sp", "totally_unknown"),
            device="cpu",
        )
        raise AssertionError("Expected KeyError for unmapped channel name")
    except KeyError as exc:
        assert "totally_unknown" in str(exc)


def test_hourly_met_tensors_channel_unknown_name_raises() -> None:
    ds = _build_mock_era5_dataset()
    reader = _InMemoryArcoReader(ds)
    request = BoundingBoxRequest(
        spatial=SpatialBounds(
            lon_min=19.5, lon_max=21.5, lat_min=9.5, lat_max=11.5, z_min=850.0, z_max=1050.0,
        ),
        time=TimeBounds(
            start=datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        ),
    )

    result = reader.fetch_hourly_window(request)

    try:
        result.channel("ustar")
        raise AssertionError("Expected KeyError for unknown channel name")
    except KeyError as exc:
        assert "'ustar'" in str(exc)
        assert "Available" in str(exc)
