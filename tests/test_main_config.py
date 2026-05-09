from __future__ import annotations

import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from lpdm.config import RunConfig
from lpdm.main import _footprint_time_bin_index, _validate_meteorology_time_coverage


def _base_dict(**section_overrides: dict[str, object]) -> dict[str, object]:
    """Build a fully-populated valid RunConfig dict; override sections with kwargs."""

    base: dict[str, dict[str, object]] = {
        "io": {"zarr_store": "gs://dummy/era5.zarr", "output_uri": "outputs/test-run"},
        "simulation": {
            "start_time": "2024-01-01T00:00:00Z",
            "length_seconds": 7200,
            "dt_seconds": 300,
            "device": "cpu",
        },
        "release": {
            "kind": "point",
            "lon": 1.0,
            "lat": 2.0,
            "alt_agl_m": 400.0,
            "duration_seconds": 1800,
            "n_particles": 100,
            "seed": 42,
        },
        "turbulence": {"scheme": "hanna_1982"},
        "output_grid": {
            "lon_bounds": [-2.0, 4.0],
            "lat_bounds": [-1.0, 5.0],
            "n_x": 24,
            "n_y": 24,
            "z_edges_m": [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "n_time_bins": 2,
        },
        "met_domain": {
            "lon_bounds": [-3.0, 5.0],
            "lat_bounds": [-2.0, 6.0],
            "alt_max_m": 10000.0,
        },
        "memory": {
            "met_cache_max_hours": 2,
            "log_every_steps": 10,
            "gc_every_steps": 50,
            "guard_check_every_steps": 1,
        },
    }
    for section, overrides in section_overrides.items():
        base[section] = {**base[section], **overrides}
    return base


def test_run_config_parses_and_validates() -> None:
    cfg = RunConfig.model_validate(_base_dict())
    assert cfg.io.zarr_store == "gs://dummy/era5.zarr"
    assert cfg.release.n_particles == 100
    assert cfg.simulation.start_time.isoformat().endswith("+00:00")
    assert cfg.release.duration_seconds == 1800
    assert cfg.simulation.length_seconds == 7200
    assert cfg.release.seed == 42


def test_run_config_rejects_release_at_least_as_long_as_simulation() -> None:
    with pytest.raises(ValidationError, match="length_seconds must be > release"):
        RunConfig.model_validate(
            _base_dict(
                simulation={"length_seconds": 3600},
                release={"duration_seconds": 3600, "seed": None},
            )
        )


def test_run_config_rejects_negative_release_seed() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_base_dict(release={"seed": -1}))


def test_run_config_rejects_z_edges_not_ascending() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        RunConfig.model_validate(
            _base_dict(output_grid={"z_edges_m": [0.0, 1000.0, 500.0]})
        )


def test_run_config_rejects_release_outside_met_domain() -> None:
    with pytest.raises(ValidationError, match="outside met_domain"):
        RunConfig.model_validate(_base_dict(release={"lon": 99.0}))


def test_run_config_rejects_unknown_field() -> None:
    cfg_dict = _base_dict()
    cfg_dict["unexpected"] = "x"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(cfg_dict)


def test_run_config_with_overrides_round_trips() -> None:
    cfg = RunConfig.model_validate(_base_dict())
    overridden = cfg.with_overrides(device="cuda", output_uri="outputs/other", start_time="2025-06-01T00:00:00Z")
    assert overridden.simulation.device == "cuda"
    assert overridden.io.output_uri == "outputs/other"
    assert overridden.simulation.start_time.year == 2025
    # Original is unchanged.
    assert cfg.simulation.device == "cpu"


def test_run_config_from_yaml(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        io:
          zarr_store: fake://x
          output_uri: outputs/test
        simulation:
          start_time: 2024-01-01T00:00:00Z
          length_seconds: 7200
          dt_seconds: 300
          device: cpu
        release:
          kind: point
          lon: 0.0
          lat: 0.0
          alt_agl_m: 100.0
          duration_seconds: 60
          n_particles: 10
          seed: 1
        turbulence:
          scheme: placeholder_constant_ou
        output_grid:
          lon_bounds: [-2.0, 2.0]
          lat_bounds: [-2.0, 2.0]
          n_x: 8
          n_y: 8
          z_edges_m: [0.0, 1000.0, 5000.0]
          n_time_bins: 2
        met_domain:
          lon_bounds: [-3.0, 3.0]
          lat_bounds: [-3.0, 3.0]
          alt_max_m: 10000.0
        """
    ).strip()
    p = tmp_path / "run.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    cfg = RunConfig.from_yaml(p)
    assert cfg.release.n_particles == 10
    assert cfg.output_grid.z_edges_m == (0.0, 1000.0, 5000.0)


def test_footprint_time_bin_index_advances_each_hour() -> None:
    cfg = RunConfig.model_validate(
        _base_dict(simulation={"length_seconds": 10800}, release={"duration_seconds": 1800})
    )
    release_end = cfg.simulation.start_time + timedelta(seconds=cfg.release.duration_seconds)

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
    cfg = RunConfig.model_validate(
        _base_dict(
            simulation={"start_time": "2024-01-01T06:00:00Z", "length_seconds": 36000},
            release={"duration_seconds": 3600},
        )
    )
    release_end = cfg.simulation.start_time + timedelta(seconds=cfg.release.duration_seconds)
    sim_start = release_end - timedelta(seconds=cfg.simulation.length_seconds)
    reader = _CoverageReader(
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="Meteorological dataset does not cover"):
        _validate_meteorology_time_coverage(reader, cfg, release_end, sim_start)
