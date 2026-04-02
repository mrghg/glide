from __future__ import annotations

from argparse import Namespace

from lpdm.main import _build_config


def test_build_config_parses_and_validates() -> None:
    args = Namespace(
        zarr_store="gs://dummy/era5.zarr",
        start_time="2024-01-01T00:00:00Z",
        end_time="2024-01-01T03:00:00Z",
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
    assert cfg.end_time.isoformat().endswith("+00:00")


def test_build_config_rejects_bad_times() -> None:
    args = Namespace(
        zarr_store="gs://dummy/era5.zarr",
        start_time="2024-01-01T03:00:00Z",
        end_time="2024-01-01T03:00:00Z",
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
        raise AssertionError("Expected ValueError for invalid time ordering")
    except ValueError as exc:
        assert "end-time must be later" in str(exc)
