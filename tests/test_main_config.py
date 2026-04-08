from __future__ import annotations

from argparse import Namespace

from lpdm.main import _build_config


def test_build_config_parses_and_validates() -> None:
    args = Namespace(
        zarr_store="gs://dummy/era5.zarr",
        start_time="2024-01-01T00:00:00Z",
        release_duration_seconds=1800,
        simulation_length_seconds=7200,
        release_seed=42,
        n_particles=100,
        release_lon=1.0,
        release_lat=2.0,
        release_alt_agl_m=400.0,
        dt_seconds=300,
        bbox_pad_lon_deg=2.0,
        bbox_pad_lat_deg=2.0,
        bbox_pad_alt_m=3000.0,
        output_uri="outputs/test-run",
        device="cpu",
    )

    cfg = _build_config(args)
    assert cfg.zarr_store == "gs://dummy/era5.zarr"
    assert cfg.n_particles == 100
    assert cfg.start_time.isoformat().endswith("+00:00")
    assert cfg.release_duration_seconds == 1800
    assert cfg.simulation_length_seconds == 7200
    assert cfg.release_seed == 42


def test_build_config_rejects_short_simulation_length() -> None:
    args = Namespace(
        zarr_store="gs://dummy/era5.zarr",
        start_time="2024-01-01T00:00:00Z",
        release_duration_seconds=3600,
        simulation_length_seconds=3600,
        release_seed=None,
        n_particles=100,
        release_lon=1.0,
        release_lat=2.0,
        release_alt_agl_m=400.0,
        dt_seconds=300,
        bbox_pad_lon_deg=2.0,
        bbox_pad_lat_deg=2.0,
        bbox_pad_alt_m=3000.0,
        output_uri="outputs/test-run",
        device="cpu",
    )

    try:
        _build_config(args)
        raise AssertionError("Expected ValueError for invalid simulation/release length relationship")
    except ValueError as exc:
        assert "simulation-length-seconds must be > release-duration-seconds" in str(exc)


def test_build_config_rejects_missing_start_time() -> None:
    args = Namespace(
        zarr_store="gs://dummy/era5.zarr",
        start_time=None,
        release_duration_seconds=1800,
        simulation_length_seconds=5400,
        release_seed=None,
        n_particles=100,
        release_lon=1.0,
        release_lat=2.0,
        release_alt_agl_m=400.0,
        dt_seconds=300,
        bbox_pad_lon_deg=2.0,
        bbox_pad_lat_deg=2.0,
        bbox_pad_alt_m=3000.0,
        output_uri="outputs/test-run",
        device="cpu",
    )

    try:
        _build_config(args)
        raise AssertionError("Expected ValueError for missing start-time")
    except ValueError as exc:
        assert "Missing --start-time" in str(exc)


def test_build_config_rejects_negative_release_seed() -> None:
    args = Namespace(
        zarr_store="gs://dummy/era5.zarr",
        start_time="2024-01-01T00:00:00Z",
        release_duration_seconds=1800,
        simulation_length_seconds=5400,
        release_seed=-1,
        n_particles=100,
        release_lon=1.0,
        release_lat=2.0,
        release_alt_agl_m=400.0,
        dt_seconds=300,
        bbox_pad_lon_deg=2.0,
        bbox_pad_lat_deg=2.0,
        bbox_pad_alt_m=3000.0,
        output_uri="outputs/test-run",
        device="cpu",
    )

    try:
        _build_config(args)
        raise AssertionError("Expected ValueError for negative release seed")
    except ValueError as exc:
        assert "release-seed must be >= 0" in str(exc)
