from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta, timezone

from lpdm.main import _build_config, _footprint_time_bin_index, _validate_meteorology_time_coverage


def _base_args(**overrides: object) -> Namespace:
    args = {
        "zarr_store": "gs://dummy/era5.zarr",
        "start_time": "2024-01-01T00:00:00Z",
        "release_duration_seconds": 1800,
        "simulation_length_seconds": 7200,
        "release_seed": 42,
        "n_particles": 100,
        "release_lon": 1.0,
        "release_lat": 2.0,
        "release_alt_agl_m": 400.0,
        "dt_seconds": 300,
        "bbox_pad_lon_deg": 2.0,
        "bbox_pad_lat_deg": 2.0,
        "bbox_pad_alt_m": 3000.0,
        "output_uri": "outputs/test-run",
        "device": "cpu",
        "met_cache_max_hours": 2,
        "memory_log_every_steps": 10,
        "gc_every_steps": 50,
        "memory_guard_max_rss_gib": None,
        "memory_guard_max_device_allocated_gib": None,
        "memory_guard_max_device_reserved_gib": None,
        "memory_guard_check_every_steps": 1,
    }
    args.update(overrides)
    return Namespace(**args)


def test_build_config_parses_and_validates() -> None:
    args = _base_args()

    cfg = _build_config(args)
    assert cfg.zarr_store == "gs://dummy/era5.zarr"
    assert cfg.n_particles == 100
    assert cfg.start_time.isoformat().endswith("+00:00")
    assert cfg.release_duration_seconds == 1800
    assert cfg.simulation_length_seconds == 7200
    assert cfg.release_seed == 42


def test_build_config_rejects_short_simulation_length() -> None:
    args = _base_args(
        release_duration_seconds=3600,
        simulation_length_seconds=3600,
        release_seed=None,
    )

    try:
        _build_config(args)
        raise AssertionError("Expected ValueError for invalid simulation/release length relationship")
    except ValueError as exc:
        assert "simulation-length-seconds must be > release-duration-seconds" in str(exc)


def test_build_config_rejects_missing_start_time() -> None:
    args = _base_args(
        start_time=None,
        simulation_length_seconds=5400,
        release_seed=None,
    )

    try:
        _build_config(args)
        raise AssertionError("Expected ValueError for missing start-time")
    except ValueError as exc:
        assert "Missing --start-time" in str(exc)


def test_build_config_rejects_negative_release_seed() -> None:
    args = _base_args(
        simulation_length_seconds=5400,
        release_seed=-1,
    )

    try:
        _build_config(args)
        raise AssertionError("Expected ValueError for negative release seed")
    except ValueError as exc:
        assert "release-seed must be >= 0" in str(exc)


def test_footprint_time_bin_index_advances_each_hour() -> None:
    cfg = _build_config(_base_args(simulation_length_seconds=10800, release_duration_seconds=1800))
    release_end = cfg.start_time + timedelta(seconds=cfg.release_duration_seconds)

    assert _footprint_time_bin_index(release_end, release_end, 3) == 0
    assert _footprint_time_bin_index(release_end, release_end - timedelta(minutes=55), 3) == 0
    assert _footprint_time_bin_index(release_end, release_end - timedelta(minutes=60), 3) == 1
    assert _footprint_time_bin_index(release_end, release_end - timedelta(minutes=125), 3) == 2


class _CoverageReader:
    def __init__(self, start: datetime, end: datetime) -> None:
        self._start = start
        self._end = end

    def get_time_coverage(self) -> tuple[datetime, datetime]:
        return self._start, self._end


def test_validate_meteorology_time_coverage_rejects_insufficient_history() -> None:
    cfg = _build_config(
        _base_args(
            start_time="2024-01-01T06:00:00Z",
            release_duration_seconds=3600,
            simulation_length_seconds=36000,
        )
    )
    release_end = cfg.start_time + timedelta(seconds=cfg.release_duration_seconds)
    sim_start = release_end - timedelta(seconds=cfg.simulation_length_seconds)
    reader = _CoverageReader(
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    try:
        _validate_meteorology_time_coverage(reader, cfg, release_end, sim_start)
        raise AssertionError("Expected ValueError for insufficient meteorology coverage")
    except ValueError as exc:
        assert "Meteorological dataset does not cover the requested simulation window" in str(exc)
        assert "Reduce simulation-length-seconds" in str(exc)
