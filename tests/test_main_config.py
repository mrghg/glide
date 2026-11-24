from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from lpdm.config import (
    ConcreteRelease,
    PeriodicPointReleaseConfig,
    PointReleaseConfig,
    PointScheduleReleaseConfig,
    RunConfig,
)
from lpdm.main import _footprint_time_bin_index, _validate_meteorology_time_coverage
from lpdm.met_reader import MetReader


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
            "n_time_bins": 3,
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
        "batch": {"max_releases_per_batch": 24},
    }
    for section, overrides in section_overrides.items():
        # The release section is polymorphic — a different `kind` brings a
        # disjoint field set, so don't merge against the point-release defaults.
        if (
            section == "release"
            and "kind" in overrides
            and overrides["kind"] != base[section].get("kind")
        ):
            base[section] = overrides
        else:
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
        RunConfig.model_validate(_base_dict(output_grid={"z_edges_m": [0.0, 1000.0, 500.0]}))


def test_run_config_rejects_release_outside_met_domain() -> None:
    with pytest.raises(ValidationError, match="outside met_domain"):
        RunConfig.model_validate(_base_dict(release={"lon": 99.0}))


def _multi_site_release(**overrides: object) -> dict[str, object]:
    rel: dict[str, object] = {
        "kind": "multi_point_periodic",
        "sites": [
            {"name": "A", "lon": 0.0, "lat": 0.0, "alt_agl_m": 100.0},
            {"name": "B", "lon": 1.0, "lat": 1.0, "alt_agl_m": 200.0},
        ],
        "start_time": "2024-01-01T00:00:00Z",
        "period_seconds": 3600,
        "n_releases": 3,
        "duration_seconds": 1800,
        "n_particles_per_release": 100,
        "seed": 42,
    }
    rel.update(overrides)
    return rel


def test_run_config_accepts_multi_point_periodic() -> None:
    cfg = RunConfig.model_validate(_base_dict(release=_multi_site_release()))
    rels = [r for b in cfg.expand_to_batches() for r in b.releases]
    assert len(rels) == 6  # 2 sites × 3 times
    assert [r.label for r in rels] == ["A", "B", "A", "B", "A", "B"]  # time-major


def test_multi_site_rejects_site_outside_met_domain() -> None:
    bad = _multi_site_release(
        sites=[
            {"name": "A", "lon": 0.0, "lat": 0.0, "alt_agl_m": 100.0},
            {"name": "B", "lon": 99.0, "lat": 1.0, "alt_agl_m": 200.0},
        ]
    )
    with pytest.raises(ValidationError, match=r"release \(B\) lon=99.0 outside met_domain"):
        RunConfig.model_validate(_base_dict(release=bad))


def test_multi_site_rejects_duplicate_site_names() -> None:
    dup = _multi_site_release(
        sites=[
            {"name": "A", "lon": 0.0, "lat": 0.0, "alt_agl_m": 100.0},
            {"name": "A", "lon": 1.0, "lat": 1.0, "alt_agl_m": 200.0},
        ]
    )
    with pytest.raises(ValidationError, match="site names must be unique"):
        RunConfig.model_validate(_base_dict(release=dup))


def test_run_config_rejects_unknown_field() -> None:
    cfg_dict = _base_dict()
    cfg_dict["unexpected"] = "x"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(cfg_dict)


def test_run_config_with_overrides_round_trips() -> None:
    cfg = RunConfig.model_validate(_base_dict())
    overridden = cfg.with_overrides(
        device="cuda", output_uri="outputs/other", start_time="2025-06-01T00:00:00Z"
    )
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


def test_run_config_accepts_zarr_store_list() -> None:
    cfg = RunConfig.model_validate(
        _base_dict(io={"zarr_store": ["gs://a/x.zarr", "gs://b/y.zarr"]})
    )
    assert cfg.io.zarr_store == ["gs://a/x.zarr", "gs://b/y.zarr"]


def test_run_config_rejects_empty_zarr_store_list() -> None:
    with pytest.raises(ValidationError, match="at least one entry"):
        RunConfig.model_validate(_base_dict(io={"zarr_store": []}))


def test_run_config_rejects_blank_zarr_store_list_entry() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        RunConfig.model_validate(_base_dict(io={"zarr_store": ["gs://a.zarr", ""]}))


def test_point_release_expands_to_single_batch_at_simulation_start_time() -> None:
    """`kind: point` must produce one batch with one release at sim.start_time."""

    cfg = RunConfig.model_validate(_base_dict())
    batches = cfg.expand_to_batches()

    assert len(batches) == 1
    assert len(batches[0].releases) == 1
    only = batches[0].releases[0]
    assert isinstance(only, ConcreteRelease)
    assert only.release_idx == 0
    assert only.release_time == cfg.simulation.start_time
    assert only.lon == cfg.release.lon
    assert only.lat == cfg.release.lat
    assert only.alt_agl_m == cfg.release.alt_agl_m
    assert only.duration_seconds == cfg.release.duration_seconds
    assert only.n_particles == cfg.release.n_particles
    assert only.seed == cfg.release.seed


def _periodic_release_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "kind": "periodic_point",
        "point": {"lon": 1.0, "lat": 2.0, "alt_agl_m": 400.0},
        "start_time": "2024-01-01T00:00:00Z",
        "period_seconds": 3600,
        "n_releases": 3,
        "duration_seconds": 1800,
        "n_particles_per_release": 100,
        "seed": 10,
    }
    base.update(overrides)
    return base


def test_periodic_point_release_parses_and_validates() -> None:
    cfg = RunConfig.model_validate(_base_dict(release=_periodic_release_dict()))
    assert isinstance(cfg.release, PeriodicPointReleaseConfig)
    assert cfg.release.n_releases == 3
    assert cfg.release.point.lon == 1.0


def test_periodic_point_expands_to_correct_times_and_seeds() -> None:
    cfg = RunConfig.model_validate(
        _base_dict(
            release=_periodic_release_dict(n_releases=5, period_seconds=3600),
            batch={"max_releases_per_batch": 24},
        )
    )
    batches = cfg.expand_to_batches()

    assert len(batches) == 1
    assert len(batches[0].releases) == 5

    expected_t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    for i, rel in enumerate(batches[0].releases):
        assert rel.release_idx == i
        assert rel.release_time == expected_t0 + timedelta(seconds=i * 3600)
        assert rel.seed == 10 + i  # seed + release_idx derivation
        assert rel.n_particles == 100


def test_periodic_point_chunks_into_multiple_batches() -> None:
    """24 releases @ max_per_batch=10 should produce 3 batches of 10/10/4."""

    cfg = RunConfig.model_validate(
        _base_dict(
            release=_periodic_release_dict(n_releases=24),
            batch={"max_releases_per_batch": 10},
        )
    )
    batches = cfg.expand_to_batches()

    assert [len(b.releases) for b in batches] == [10, 10, 4]
    assert [b.batch_idx for b in batches] == [0, 1, 2]
    # Release indices should be contiguous across batch boundaries.
    flat_idxs = [r.release_idx for b in batches for r in b.releases]
    assert flat_idxs == list(range(24))


def test_periodic_point_with_null_seed_propagates_none() -> None:
    cfg = RunConfig.model_validate(
        _base_dict(release=_periodic_release_dict(seed=None, n_releases=3))
    )
    batches = cfg.expand_to_batches()
    assert all(r.seed is None for r in batches[0].releases)


def test_point_schedule_release_parses_and_expands() -> None:
    times = ["2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z", "2024-01-02T12:00:00Z"]
    cfg = RunConfig.model_validate(
        _base_dict(
            release={
                "kind": "point_schedule",
                "point": {"lon": 1.0, "lat": 2.0, "alt_agl_m": 400.0},
                "times": times,
                "duration_seconds": 1800,
                "n_particles_per_release": 100,
                "seed": 5,
            }
        )
    )
    assert isinstance(cfg.release, PointScheduleReleaseConfig)

    batches = cfg.expand_to_batches()
    assert len(batches) == 1
    actual = [r.release_time.isoformat() for r in batches[0].releases]
    assert actual == [
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T06:00:00+00:00",
        "2024-01-02T12:00:00+00:00",
    ]


def test_release_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_base_dict(release={"kind": "totally_bogus"}))


def test_periodic_point_rejects_release_outside_met_domain() -> None:
    """Cross-section validator must work for the periodic variant too."""

    with pytest.raises(ValidationError, match="outside met_domain"):
        RunConfig.model_validate(
            _base_dict(
                release=_periodic_release_dict(point={"lon": 99.0, "lat": 2.0, "alt_agl_m": 400.0})
            )
        )


def test_periodic_point_rejects_release_longer_than_sim_length() -> None:
    with pytest.raises(ValidationError, match="length_seconds must be > release"):
        RunConfig.model_validate(
            _base_dict(
                simulation={"length_seconds": 1800},
                release=_periodic_release_dict(duration_seconds=1800),
            )
        )


def test_point_schedule_rejects_empty_times() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            _base_dict(
                release={
                    "kind": "point_schedule",
                    "point": {"lon": 1.0, "lat": 2.0, "alt_agl_m": 400.0},
                    "times": [],
                    "duration_seconds": 1800,
                    "n_particles_per_release": 100,
                    "seed": 5,
                }
            )
        )


def test_batch_config_defaults_and_override() -> None:
    cfg_default = RunConfig.model_validate(_base_dict())
    assert cfg_default.batch.max_releases_per_batch == 24

    cfg_override = RunConfig.model_validate(_base_dict(batch={"max_releases_per_batch": 7}))
    assert cfg_override.batch.max_releases_per_batch == 7


def test_turbulence_meander_defaults_off() -> None:
    cfg = RunConfig.model_validate(_base_dict())
    assert cfg.turbulence.meander.enabled is False
    assert cfg.turbulence.meander.coefficient == 0.16
    assert cfg.turbulence.meander.timescale_seconds is None


def test_scheme_kwargs_forwards_meander_only_for_hanna() -> None:
    from lpdm.main import _scheme_kwargs

    cfg_hanna = RunConfig.model_validate(
        _base_dict(
            turbulence={
                "scheme": "hanna_1982",
                "meander": {
                    "enabled": True,
                    "coefficient": 0.2,
                    "stencil_radius": 2,
                    "timescale_seconds": 900.0,
                },
            }
        )
    )
    kw = _scheme_kwargs(cfg_hanna)
    assert kw == {
        "meander_enabled": True,
        "meander_coefficient": 0.2,
        "meander_stencil_radius": 2,
        "meander_timescale_seconds": 900.0,
        # Substepping knobs default to the historical hardcoded values.
        "substep_c": 0.5,
        "max_substeps": 50,
        # Near-surface mixing knobs default to the FLEXPART-matching behaviour
        # (T_L floors on, legacy surface-layer override off).
        "flexpart_tl_floors": True,
        "surface_layer_override": False,
        # Boundary-layer depth limits default to the long-standing 50 m floor and
        # no ceiling (A03, decision 0024).
        "blh_min_m": 50.0,
        "blh_max_m": None,
    }

    # Non-Hanna schemes take no kwargs (would reject meander), so forward nothing.
    cfg_placeholder = RunConfig.model_validate(
        _base_dict(turbulence={"scheme": "placeholder_constant_ou", "meander": {"enabled": True}})
    )
    assert _scheme_kwargs(cfg_placeholder) == {}


def test_point_release_back_compat_isinstance_check() -> None:
    """Existing code paths that assume a PointReleaseConfig must keep working."""

    cfg = RunConfig.model_validate(_base_dict())
    assert isinstance(cfg.release, PointReleaseConfig)


def test_footprint_time_bin_index_advances_each_hour() -> None:
    cfg = RunConfig.model_validate(
        _base_dict(simulation={"length_seconds": 10800}, release={"duration_seconds": 1800})
    )
    release_end = cfg.simulation.start_time + timedelta(seconds=cfg.release.duration_seconds)

    assert _footprint_time_bin_index(release_end, release_end, 3) == 0
    assert _footprint_time_bin_index(release_end, release_end - timedelta(minutes=55), 3) == 0
    assert _footprint_time_bin_index(release_end, release_end - timedelta(minutes=60), 3) == 1
    assert _footprint_time_bin_index(release_end, release_end - timedelta(minutes=125), 3) == 2


class _CoverageReader(MetReader):
    """Coverage-only stub. Subclasses MetReader for its cadence/bracketing defaults,
    which the preflight check uses to round the required window to met timestamps."""

    def __init__(self, start: datetime, end: datetime) -> None:
        self._start = start
        self._end = end

    def get_time_coverage(self) -> tuple[datetime, datetime]:
        return self._start, self._end

    def fetch_hourly_window(self, request):  # pragma: no cover - never fetched
        raise NotImplementedError


def test_validate_meteorology_time_coverage_rejects_insufficient_history() -> None:
    cfg = RunConfig.model_validate(
        _base_dict(
            simulation={"start_time": "2024-01-01T06:00:00Z", "length_seconds": 36000},
            release={"duration_seconds": 3600},
            output_grid={"n_time_bins": 1},
        )
    )
    release_end = cfg.simulation.start_time + timedelta(seconds=cfg.release.duration_seconds)
    sim_start = release_end - timedelta(seconds=cfg.simulation.length_seconds)
    reader = _CoverageReader(
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="Meteorological dataset does not cover"):
        _validate_meteorology_time_coverage(reader, cfg, release_end, sim_start)


# --- met_domain.vertical_levels (configurable internal AGL grid) -------------


def test_met_domain_default_ladder_is_40_stretched_levels() -> None:
    """Task A01, decision 0017: unset => 40 stretched levels from first_layer_m,
    spanning alt_max_m. The pre-A01 23-level ladder can still be pinned explicitly."""

    from lpdm.vertical_grid import default_agl_levels

    cfg = RunConfig.model_validate(_base_dict())
    md = cfg.met_domain
    assert md.vertical_levels is None
    levels = md.resolve_agl_levels()
    assert len(levels) == 40
    assert levels[0] == 0.0 and levels[-1] >= md.alt_max_m
    assert levels[1] == pytest.approx(md.first_layer_m)

    legacy = list(default_agl_levels(md.alt_max_m))
    pinned = RunConfig.model_validate(_base_dict(met_domain={"vertical_levels": legacy}))
    np.testing.assert_array_equal(pinned.met_domain.resolve_agl_levels(), legacy)


def test_met_domain_vertical_levels_count_generates_stretched_grid() -> None:
    cfg = RunConfig.model_validate(
        _base_dict(
            met_domain={
                "lon_bounds": [-3.0, 5.0],
                "lat_bounds": [-2.0, 6.0],
                "alt_max_m": 10000.0,
                "vertical_levels": 40,
            }
        )
    )
    levels = cfg.met_domain.resolve_agl_levels()
    assert levels is not None
    assert levels.size == 40
    assert levels[0] == 0.0
    assert levels[-1] == pytest.approx(10000.0)
    assert (levels[1:] > levels[:-1]).all()


def test_met_domain_vertical_levels_explicit_list_used_verbatim() -> None:
    explicit = [0.0, 25.0, 100.0, 1000.0, 10000.0]
    cfg = RunConfig.model_validate(
        _base_dict(
            met_domain={
                "lon_bounds": [-3.0, 5.0],
                "lat_bounds": [-2.0, 6.0],
                "alt_max_m": 10000.0,
                "vertical_levels": explicit,
            }
        )
    )
    assert list(cfg.met_domain.resolve_agl_levels()) == explicit


def test_met_domain_first_layer_m_controls_lowest_level() -> None:
    cfg = RunConfig.model_validate(
        _base_dict(
            met_domain={
                "lon_bounds": [-3.0, 5.0],
                "lat_bounds": [-2.0, 6.0],
                "alt_max_m": 10000.0,
                "vertical_levels": 30,
                "first_layer_m": 5.0,
            }
        )
    )
    assert cfg.met_domain.resolve_agl_levels()[1] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("levels", "match"),
    [
        (1, "must be >= 2"),
        ([0.0], "at least 2 heights"),
        ([10.0, 100.0, 10000.0], "must start at 0.0"),
        ([0.0, 500.0, 100.0, 10000.0], "strictly ascending"),
        ([0.0, 100.0, 500.0], "below alt_max_m"),
    ],
)
def test_met_domain_vertical_levels_rejects_bad_grids(levels, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        RunConfig.model_validate(
            _base_dict(
                met_domain={
                    "lon_bounds": [-3.0, 5.0],
                    "lat_bounds": [-2.0, 6.0],
                    "alt_max_m": 10000.0,
                    "vertical_levels": levels,
                }
            )
        )


def test_footprint_time_bin_index_uses_the_bin_width() -> None:
    end = datetime(2024, 1, 1, 12, tzinfo=UTC)
    assert _footprint_time_bin_index(end, end - timedelta(hours=5), 3, 7200.0) == 2
    assert _footprint_time_bin_index(end, end - timedelta(hours=3, minutes=59), 3, 7200.0) == 1
    assert _footprint_time_bin_index(end, end - timedelta(days=1, hours=1), 2, 86400.0) == 1


def test_preflight_refuses_met_domain_outside_the_grid() -> None:
    """A met domain in -180..180 over a grid left in 0..360 ran silently with
    the grid edge's winds (task B12); the preflight now refuses it."""

    from lpdm.main import PreflightValidationError, _validate_meteorology_space_coverage

    class _Grid:
        def __init__(self, coverage):
            self._c = coverage

        def get_lon_lat_coverage(self):
            return self._c

    cfg = RunConfig.model_validate(_base_dict())
    md = cfg.met_domain
    inside = (
        md.lon_bounds[0] - 1,
        md.lon_bounds[1] + 1,
        md.lat_bounds[0] - 1,
        md.lat_bounds[1] + 1,
    )
    _validate_meteorology_space_coverage(_Grid(inside), cfg)
    _validate_meteorology_space_coverage(_Grid(None), cfg)  # reader cannot say: no check
    shifted = (inside[0] + 360.0, inside[1] + 360.0, inside[2], inside[3])
    with pytest.raises(PreflightValidationError, match="longitude convention"):
        _validate_meteorology_space_coverage(_Grid(shifted), cfg)


# ---- A03: boundary-layer depth limits and the meander timescale (decision 0024) ----


def test_boundary_layer_depth_limits_default_and_forwarded() -> None:
    """Defaults keep the long-standing 50 m floor and no ceiling; set values (the
    validation runs' 100 m and 4,500 m) reach the Hanna scheme; an unset meander
    timescale is not forwarded, so the scheme takes half the met interval."""

    from lpdm.main import _scheme_kwargs
    from lpdm.turbulence import get_scheme

    default = RunConfig.model_validate(_base_dict(turbulence={"scheme": "hanna_1982"}))
    assert default.turbulence.boundary_layer_depth_min_m == 50.0
    assert default.turbulence.boundary_layer_depth_max_m is None
    assert "meander_timescale_seconds" not in _scheme_kwargs(default)

    cfg = RunConfig.model_validate(
        _base_dict(
            turbulence={
                "scheme": "hanna_1982",
                "boundary_layer_depth_min_m": 100.0,
                "boundary_layer_depth_max_m": 4500.0,
            }
        )
    )
    kw = _scheme_kwargs(cfg)
    assert kw["blh_min_m"] == 100.0
    assert kw["blh_max_m"] == 4500.0
    scheme = get_scheme("hanna_1982", **kw)
    assert (scheme.blh_min_m, scheme.blh_max_m) == (100.0, 4500.0)


@pytest.mark.parametrize(
    "limits",
    [
        {"boundary_layer_depth_min_m": 0.0},
        {"boundary_layer_depth_max_m": 0.0},
        {"boundary_layer_depth_min_m": 100.0, "boundary_layer_depth_max_m": 100.0},
        {"boundary_layer_depth_min_m": 500.0, "boundary_layer_depth_max_m": 400.0},
    ],
)
def test_boundary_layer_depth_limits_rejected_when_invalid(limits) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_base_dict(turbulence={"scheme": "hanna_1982", **limits}))
