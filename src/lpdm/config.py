"""Pydantic models for the GLIDE run configuration.

The top-level :class:`RunConfig` mirrors the YAML schema documented in
``configs/example_mhd_january.yaml``. All physics, IO, and runtime knobs live
here; ``main.py`` consumes a validated ``RunConfig`` and never reads from
argparse or the environment directly.

Construction paths:
- ``RunConfig.from_yaml(path)`` for the canonical CLI usage
- ``RunConfig.model_validate(dict)`` for tests and programmatic callers
- ``cfg.with_overrides(**kwargs)`` to apply small CLI overrides (device,
  output_uri, start_time)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lpdm.vertical_grid import DEFAULT_VERTICAL_LEVEL_COUNT, stretched_agl_levels


def _parse_datetime_utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IOConfig(_Frozen):
    zarr_store: str | list[str] = Field(
        ...,
        description=(
            "ERA5 ARCO Zarr store. Accepts a single URI, a local glob pattern "
            "(e.g. data/era5/EUROPE_*.zarr), or a list of URIs that "
            "share lat/lon/level coordinates and will be stitched along time."
        ),
    )
    output_uri: str = Field(..., min_length=1, description="Output directory or URI")

    @field_validator("zarr_store")
    @classmethod
    def _zarr_store_non_empty(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            if not v:
                raise ValueError("zarr_store must be a non-empty string")
            return v
        if not v:
            raise ValueError("zarr_store list must contain at least one entry")
        if not all(isinstance(s, str) and s for s in v):
            raise ValueError("zarr_store list entries must all be non-empty strings")
        return v


class SimulationConfig(_Frozen):
    start_time: datetime = Field(..., description="UTC ISO timestamp at start of release window")
    length_seconds: int = Field(
        ..., gt=0, description="Backward integration length from end of release"
    )
    dt_seconds: int = Field(..., gt=0, description="Integration timestep in seconds")
    device: str = Field("auto", description="Torch device: auto, cpu, cuda, mps, or cuda:N")

    @field_validator("start_time", mode="before")
    @classmethod
    def _coerce_start_time(cls, v: Any) -> datetime:
        return _parse_datetime_utc(v)


class PointSpec(_Frozen):
    """A 3D release point shared by the multi-release variants."""

    lon: float
    lat: float
    alt_agl_m: float = Field(..., ge=0)


class PointReleaseConfig(_Frozen):
    """Single point release (today's behaviour). Released over a single window
    starting at ``simulation.start_time``.
    """

    kind: Literal["point"] = "point"
    lon: float
    lat: float
    alt_agl_m: float = Field(..., ge=0)
    duration_seconds: int = Field(..., gt=0)
    n_particles: int = Field(..., gt=0)
    seed: int | None = Field(None, ge=0)


class PeriodicPointReleaseConfig(_Frozen):
    """Equally-spaced point releases from a single location.

    Generates ``n_releases`` releases at ``start_time + i * period_seconds``
    for ``i in [0, n_releases)``. Used for the common "every hour for N days"
    site workflow.
    """

    kind: Literal["periodic_point"]
    point: PointSpec
    start_time: datetime
    period_seconds: int = Field(..., gt=0)
    n_releases: int = Field(..., gt=0)
    duration_seconds: int = Field(..., gt=0)
    n_particles_per_release: int = Field(..., gt=0)
    seed: int | None = Field(None, ge=0)

    @field_validator("start_time", mode="before")
    @classmethod
    def _coerce_start_time(cls, v: Any) -> datetime:
        return _parse_datetime_utc(v)


class PointScheduleReleaseConfig(_Frozen):
    """Irregular point-release schedule from a single location.

    Each entry in ``times`` is the start of one release window of length
    ``duration_seconds``. Foundation for the eventual multi-point satellite
    workflow, which will add a per-time ``points`` field.
    """

    kind: Literal["point_schedule"]
    point: PointSpec
    times: tuple[datetime, ...] = Field(..., min_length=1)
    duration_seconds: int = Field(..., gt=0)
    n_particles_per_release: int = Field(..., gt=0)
    seed: int | None = Field(None, ge=0)

    @field_validator("times", mode="before")
    @classmethod
    def _coerce_times(cls, v: Any) -> tuple[datetime, ...]:
        if not isinstance(v, (list, tuple)):
            raise ValueError("times must be a list of datetime strings")
        return tuple(_parse_datetime_utc(t) for t in v)


class SiteSpec(_Frozen):
    """A named release location for multi-site runs."""

    name: str = Field(..., min_length=1)
    lon: float
    lat: float
    alt_agl_m: float = Field(..., ge=0)


class MultiPointPeriodicReleaseConfig(_Frozen):
    """Equally-spaced point releases from MULTIPLE named locations on a shared schedule.

    Generates one release per (site, time) for ``n_releases`` times at
    ``start_time + i * period_seconds`` and every site in ``sites``. All sites share the
    time grid, so they share met windows — the efficient case (one met fetch feeds every
    site's particles for that hour). The runtime/gridder already index by release, so each
    (site, time) is just another release carrying its own ``(lon, lat, alt)`` and ``label``.
    """

    kind: Literal["multi_point_periodic"]
    sites: tuple[SiteSpec, ...] = Field(..., min_length=1)
    start_time: datetime
    period_seconds: int = Field(..., gt=0)
    n_releases: int = Field(..., gt=0)
    duration_seconds: int = Field(..., gt=0)
    n_particles_per_release: int = Field(..., gt=0)
    seed: int | None = Field(None, ge=0)

    @field_validator("start_time", mode="before")
    @classmethod
    def _coerce_start_time(cls, v: Any) -> datetime:
        return _parse_datetime_utc(v)

    @field_validator("sites")
    @classmethod
    def _unique_site_names(cls, v: tuple[SiteSpec, ...]) -> tuple[SiteSpec, ...]:
        names = [s.name for s in v]
        if len(set(names)) != len(names):
            raise ValueError("site names must be unique")
        return v


class SampleSpec(_Frozen):
    """One measured sample: where, when and over which heights its particles start.

    Particles are released uniformly in time over ``[start_time, start_time +
    duration_seconds)`` and uniformly in height between ``alt_min_agl_m`` and
    ``alt_max_agl_m`` (a point when the two are equal).
    """

    name: str = Field(..., min_length=1)
    lon: float
    lat: float
    start_time: datetime
    duration_seconds: int = Field(..., gt=0)
    alt_min_agl_m: float = Field(..., ge=0)
    alt_max_agl_m: float = Field(..., ge=0)

    @field_validator("start_time", mode="before")
    @classmethod
    def _coerce_start_time(cls, v: Any) -> datetime:
        return _parse_datetime_utc(v)

    @model_validator(mode="after")
    def _height_range_ascending(self) -> SampleSpec:
        if self.alt_max_agl_m < self.alt_min_agl_m:
            raise ValueError("alt_max_agl_m must be >= alt_min_agl_m")
        return self


class SampleReleaseConfig(_Frozen):
    """One backward release per measured sample (queue task B12).

    Each sample has its own location, start, duration and height range, so a
    tracer experiment's samples can be configured one release each. Releases
    are expanded in the order given; list them in time order so that batches
    span short periods.
    """

    kind: Literal["samples"]
    samples: tuple[SampleSpec, ...] = Field(..., min_length=1)
    n_particles_per_release: int = Field(..., gt=0)
    seed: int | None = Field(None, ge=0)

    @property
    def duration_seconds(self) -> int:
        """The longest sample window (what the run-length check needs)."""

        return max(s.duration_seconds for s in self.samples)

    @field_validator("samples")
    @classmethod
    def _unique_sample_names(cls, v: tuple[SampleSpec, ...]) -> tuple[SampleSpec, ...]:
        names = [s.name for s in v]
        if len(set(names)) != len(names):
            raise ValueError("sample names must be unique")
        return v


ReleaseConfig = Annotated[
    PointReleaseConfig
    | PeriodicPointReleaseConfig
    | PointScheduleReleaseConfig
    | MultiPointPeriodicReleaseConfig
    | SampleReleaseConfig,
    Field(discriminator="kind"),
]


class BatchConfig(_Frozen):
    """How a multi-release schedule is decomposed for execution.

    The runtime groups consecutive releases into batches so they share met-fetch
    work and process startup. Single-release (``kind: "point"``) runs ignore this
    knob; they always emit one batch with one release.
    """

    max_releases_per_batch: int = Field(
        24,
        gt=0,
        description=(
            "Maximum number of releases to integrate in a single engine pass. "
            "Memory scales roughly as max_releases_per_batch × n_particles_per_release × "
            "particle_state_bytes. Default 24 (one calendar day for hourly cadence)."
        ),
    )


@dataclass(frozen=True)
class ConcreteRelease:
    """One fully-specified release inside an expanded schedule.

    Produced by :meth:`RunConfig.expand_to_batches` regardless of which YAML
    release ``kind`` produced it. The runtime consumes these uniformly.
    """

    release_idx: int
    release_time: datetime  # UTC; start of the release window
    lon: float
    lat: float
    alt_agl_m: float
    duration_seconds: int
    n_particles: int
    seed: int | None
    label: str | None = None  # site name for multi-site runs; None for single-location variants
    # Top of the release layer for `kind: samples` with a height range: particles start
    # uniformly between alt_agl_m and this. None (every other kind) releases at alt_agl_m.
    alt_top_agl_m: float | None = None


@dataclass(frozen=True)
class ReleaseBatch:
    """A contiguous slice of the schedule run together in one engine pass."""

    batch_idx: int
    releases: tuple[ConcreteRelease, ...]


def _release_point(release: Any) -> tuple[float, float, float]:
    """Return ``(lon, lat, alt_agl_m)`` for any release variant."""

    if isinstance(release, PointReleaseConfig):
        return float(release.lon), float(release.lat), float(release.alt_agl_m)
    if isinstance(release, (PeriodicPointReleaseConfig, PointScheduleReleaseConfig)):
        p = release.point
        return float(p.lon), float(p.lat), float(p.alt_agl_m)
    raise TypeError(f"Unsupported release variant: {type(release).__name__}")


class MeanderConfig(_Frozen):
    """Unresolved-mesoscale ("meander") horizontal turbulence.

    Maryon (1998) "meandering" as adopted by FLEXPART (Stohl et al. 2005, §4.5):
    an independent horizontal Langevin process whose velocity standard deviation
    is the local grid-scale wind variability (std-dev over the surrounding grid
    points) times ``coefficient``, with a Lagrangian timescale of roughly half
    the met-field interval. Resolution-dependent by construction. Hanna-scheme
    only; ignored by other schemes. Default off so existing baselines are
    unchanged.
    """

    enabled: bool = False
    coefficient: float = Field(
        0.16,
        gt=0,
        description="FLEXPART `turbmesoscale`: σ_meander = coefficient × local grid-wind std-dev.",
    )
    stencil_radius: int = Field(
        1,
        gt=0,
        description="Half-width (grid cells) of the neighbourhood used for the local wind std-dev.",
    )
    timescale_seconds: float | None = Field(
        None,
        gt=0,
        description="Lagrangian timescale of the meander process. None → half the interval between the store's meteorological fields (1800 s on hourly ERA5), measured from the store at run start.",
    )


# Hanna turbulence sub-steps per outer step (decision 0027); the same value as
# `lpdm.turbulence.hanna.N_SUBSTEPS_DEFAULT` (tests/test_main_config.py checks).
N_SUBSTEPS_DEFAULT = 16


class TurbulenceConfig(_Frozen):
    scheme: str = Field(..., min_length=1)
    meander: MeanderConfig = MeanderConfig()
    # Hanna sub-stepping (decision 0027, docs/physics.md §6). Every particle takes
    # n_substeps equal sub-steps per outer step, with all coefficients
    # re-evaluated at the current height each sub-step. The count is a constant
    # of the run: never per particle, per step or per batch. A new field rather
    # than a reinterpreted `max_substeps`, so that no config silently changes
    # meaning (the removed keys are refused below). Ignored by non-Hanna schemes.
    # It is a COUNT, not a length: the sub-step is dt_seconds / n_substeps, so a
    # longer outer step gives longer sub-steps. The run logs the length, records it
    # in run_metadata.json (runtime.substep_exceedance.substep_seconds) and warns
    # when it is longer than the 60 s / default the well-mixed tests are run at
    # (decision 0028 says why it stays a count).
    n_substeps: int = Field(
        N_SUBSTEPS_DEFAULT,
        ge=1,
        description="Turbulence sub-steps per outer step, the same for every particle and step. A count: the sub-step is dt_seconds / n_substeps, so changing dt_seconds changes the sub-step length.",
    )
    # Near-surface mixing controls (2026-07-02 physics-review follow-up). Defaults
    # are the FLEXPART-v11-matching "improved" behaviour; the legacy combination
    # (floors off + override on) is retained for A/B comparisons. Hanna-only.
    flexpart_tl_floors: bool = Field(
        True,
        description="Apply FLEXPART v11 Lagrangian-timescale floors (T_Lu,T_Lv ≥ 10 s; T_Lw ≥ 30 s) to the BL profile — prevents near-surface K collapse.",
    )
    surface_layer_override: bool = Field(
        False,
        description="Enable the legacy GLIDE-only Monin-Obukhov surface-layer override below 0.1·BLH (FLEXPART has no such override).",
    )
    # Limits on the boundary-layer depth h from the meteorology (decision 0024).
    # Applied wherever the turbulence scheme uses h. The defaults keep GLIDE's
    # long-standing behaviour (50 m floor, no ceiling); the validation runs set
    # 100 and 4500, the values FLEXPART (`hmixmin`, `hmixmax`) and NAME use.
    # Hanna-only.
    boundary_layer_depth_min_m: float = Field(
        50.0,
        gt=0,
        description="Lower limit on the boundary-layer depth (m) used by the turbulence scheme.",
    )
    boundary_layer_depth_max_m: float | None = Field(
        None,
        gt=0,
        description="Upper limit on the boundary-layer depth (m); null for none.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_adaptive_substep_keys(cls, data: Any) -> Any:
        # Removed with the adaptive schedule (decision 0027). Refused with a
        # migration message rather than ignored: a config that sets them expects a
        # behaviour GLIDE no longer has.
        if isinstance(data, dict):
            old = [k for k in ("substep_c", "max_substeps") if k in data]
            if old:
                raise ValueError(
                    f"turbulence.{' and turbulence.'.join(old)} no longer "
                    f"exist{'s' if len(old) == 1 else ''}: the per-particle adaptive "
                    "sub-step schedule was replaced by one fixed count for every "
                    "particle (decision 0027, docs/physics.md §6). Remove "
                    f"{' and '.join(old)}; set turbulence.n_substeps to choose the "
                    f"count (default {N_SUBSTEPS_DEFAULT}, a sub-step of "
                    f"{60.0 / N_SUBSTEPS_DEFAULT:g} s at dt_seconds = 60)."
                )
        return data

    @model_validator(mode="after")
    def _check_boundary_layer_depth_limits(self) -> TurbulenceConfig:
        hi = self.boundary_layer_depth_max_m
        if hi is not None and hi <= self.boundary_layer_depth_min_m:
            raise ValueError(
                f"turbulence.boundary_layer_depth_max_m ({hi}) must be greater than "
                f"boundary_layer_depth_min_m ({self.boundary_layer_depth_min_m})"
            )
        return self


class EmanuelConvectionConfig(_Frozen):
    """Per-scheme tuning for the reduced Emanuel convection (Forster 2007)."""

    closure_c: float = Field(
        0.03,
        gt=0,
        description="Cloud-base mass-flux closure constant. M_b ∝ closure_c · ρ_LCL · √(2·CAPE).",
    )
    trigger_dtv_k: float = Field(
        0.9,
        ge=0,
        description="Buoyancy excess at LCL+1 required to trigger convection (Forster 2007 Eq 34, default 0.9 K).",
    )
    min_cape_j_kg: float = Field(
        50.0,
        ge=0,
        description="CAPE floor below which convection never fires regardless of trigger check.",
    )
    min_cloud_depth_m: float = Field(
        500.0,
        ge=0,
        description="Skip shallow convection (cloud depth < this). This scheme is for DEEP convection.",
    )


class ConvectionConfig(_Frozen):
    """Deep-convection configuration. Default ``scheme = "none"`` is bit-
    equivalent to no-convection runs (the scheme is a pass-through). Switch to
    ``"emanuel_reduced"`` and configure via the ``emanuel`` block to enable.
    """

    scheme: str = Field(
        "none",
        min_length=1,
        description='Convection scheme name. "none" (default, pass-through) or "emanuel_reduced".',
    )
    emanuel: EmanuelConvectionConfig = EmanuelConvectionConfig()


class DepositionConfig(_Frozen):
    """Where in a step each particle's residence is counted (task A2, [PR-F4])."""

    # `endpoint`: all of the step at the particle's position after the step's
    # advection, turbulence and convection (the long-standing behaviour).
    # `trapezoid`: half the step at the start position and half at the end, each
    # split between time-ago bins by its overlap with them and with its own density
    # (decision 0029).
    # `step_mean` (decision 0026) was removed: scored against fine-time residence it
    # undercounts the 0-40 m layer by 3-5% (up to 15% in strong turbulence). The RK2 `midpoint` was removed
    # before it. Kernel deposition (decision 0012) is deferred to after v1.
    position: Literal["endpoint", "trapezoid"] = "endpoint"

    @field_validator("position", mode="before")
    @classmethod
    def _removed_positions(cls, value: object) -> object:
        removed = {
            "step_mean": (
                "output_grid.deposition.position 'step_mean' was removed (decision 0029): "
                "scored against fine-time residence it undercounts the 0-40 m layer "
                "(3-5% on the smoke run, up to 15% in strong turbulence). Use 'endpoint' "
                "(default) or 'trapezoid'."
            ),
            "midpoint": (
                "output_grid.deposition.position 'midpoint' was removed (decision 0026): "
                "it counts each first step at the release height. Use 'endpoint' or "
                "'trapezoid'."
            ),
        }
        if value in removed:
            raise ValueError(removed[value])
        return value


class OutputGridConfig(_Frozen):
    """Grid onto which residence-time footprints are accumulated."""

    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    n_x: int = Field(..., gt=0)
    n_y: int = Field(..., gt=0)
    z_edges_m: tuple[float, ...] = Field(..., min_length=2)
    n_time_bins: int = Field(..., gt=0)
    # Width of each time-ago bin (decision 0016). Ages count back from each release
    # window's end; bin k holds ages in [k, k+1) x time_bin_seconds. With one bin
    # the footprint is time-integrated and the width is not used.
    time_bin_seconds: int = Field(3600, gt=0)
    # With several bins, n_time_bins x time_bin_seconds must cover the simulation
    # length, or residence older than the axis is dropped. Dropping it must be
    # asked for (decision 0016).
    drop_older_ages: bool = False
    # What each counted particle-second is divided by (decision 0022). `dry_air`:
    # the dry-air density at the particle, (p - e) / (R_d T), at the step's met
    # time; the footprint is in s m3 kg-1 of dry air, and M_air / h gives
    # m2 s mol-1 with no further density. `none`: plain residence seconds.
    density_weighting: Literal["dry_air", "none"] = "dry_air"
    # Where in each step residence is counted (task A2); default endpoint.
    deposition: DepositionConfig = DepositionConfig()
    # Precision of the in-memory footprint accumulator (A2 follow-up, prototype).
    # `float32` (default) is the long-standing behaviour. `float64` sums each cell's
    # contributions in double precision and rounds to float32 once, when the batch
    # is written; the store stays float32. It doubles the accumulator's memory (one
    # batch's grid).
    accumulation_dtype: Literal["float32", "float64"] = "float32"
    # Footprint storage (task B2). Both None (default) keep the store's default
    # encoding, so existing output is unchanged. `footprint_keep_mantissa_bits`
    # rounds each value to that many float32 mantissa bits before compression
    # (relative error <= 2**-(bits+1); lossy, so off by default and never for
    # validation output); `footprint_zstd_level` compresses with zstd.
    footprint_keep_mantissa_bits: int | None = Field(None, ge=1, le=23)
    footprint_zstd_level: int | None = Field(None, ge=1, le=22)
    # Also write `footprints_sparse.zarr`, the footprint as COO per release
    # (lpdm.sparse_footprint; loader `comparison.load_sparse_footprint`).
    footprint_sparse: bool = False

    @field_validator("z_edges_m")
    @classmethod
    def _z_edges_strictly_ascending(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if any(v[i] >= v[i + 1] for i in range(len(v) - 1)):
            raise ValueError("z_edges_m must be strictly ascending")
        return v

    @field_validator("lon_bounds", "lat_bounds")
    @classmethod
    def _bounds_ascending(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError("bounds must satisfy lower < upper")
        return v


class MetDomainConfig(_Frozen):
    """Spatial cap for meteorological field fetches.

    Every per-hour fetch uses this bbox (lon/lat) and a vertical cap of
    [0, alt_max_m] AGL. Particles leaving this region are excluded from
    advection and footprint accumulation.
    """

    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    alt_max_m: float = Field(..., gt=0)
    # Internal terrain-following AGL grid every met hour is resampled onto. Either:
    #   - an int  -> that many geometrically stretched levels (see
    #     `vertical_grid.stretched_agl_levels`); the user-friendly form, and
    #   - a list  -> explicit ascending AGL heights in metres, for full control.
    # None (default) -> DEFAULT_VERTICAL_LEVEL_COUNT (40) stretched levels from
    # `first_layer_m` (decision 0017). The pre-A01 23-level ladder is
    # `list(vertical_grid.default_agl_levels(alt_max_m))`.
    #
    # This grid, NOT the met source, sets the vertical resolution the physics sees:
    # the default resolves ~13 levels below 1.5 km, which under-uses a model-level
    # (hybrid) source carrying ~20 there. Raise it to exploit such a source — but
    # note the met cache scales LINEARLY with the level count, so `met_cache_max_hours`
    # x levels drives host RAM.
    vertical_levels: int | list[float] | None = None
    # Lowest-layer thickness (m) when `vertical_levels` is a count. The default
    # matches ERA5's lowest model level (~10 m AGL).
    first_layer_m: float = Field(10.0, gt=0)

    @field_validator("lon_bounds", "lat_bounds")
    @classmethod
    def _bounds_ascending(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError("bounds must satisfy lower < upper")
        return v

    @field_validator("vertical_levels")
    @classmethod
    def _validate_vertical_levels(cls, v: int | list[float] | None) -> int | list[float] | None:
        if v is None:
            return v
        if isinstance(v, int):
            if v < 2:
                raise ValueError("vertical_levels count must be >= 2")
            return v
        if len(v) < 2:
            raise ValueError("vertical_levels list must have at least 2 heights")
        if v[0] != 0.0:
            raise ValueError("vertical_levels list must start at 0.0 m (the surface)")
        if any(b <= a for a, b in zip(v[:-1], v[1:], strict=True)):
            raise ValueError("vertical_levels list must be strictly ascending")
        return v

    @model_validator(mode="after")
    def _check_grid_reaches_top(self) -> MetDomainConfig:
        levels = self.vertical_levels
        if isinstance(levels, list) and levels[-1] < self.alt_max_m:
            raise ValueError(
                f"vertical_levels top ({levels[-1]} m) is below alt_max_m "
                f"({self.alt_max_m} m); the grid must span the met domain."
            )
        return self

    def resolve_agl_levels(self) -> np.ndarray | None:
        """The AGL height grid for this config.

        Unset -> ``DEFAULT_VERTICAL_LEVEL_COUNT`` stretched levels (decision 0017).
        """

        if self.vertical_levels is None:
            return stretched_agl_levels(
                DEFAULT_VERTICAL_LEVEL_COUNT, self.alt_max_m, first_layer_m=self.first_layer_m
            )
        if isinstance(self.vertical_levels, int):
            return stretched_agl_levels(
                self.vertical_levels, self.alt_max_m, first_layer_m=self.first_layer_m
            )
        return np.asarray(self.vertical_levels, dtype="float64")


class MemoryConfig(_Frozen):
    met_cache_max_hours: int = Field(2, ge=0)
    # Cache met windows in HOST RAM instead of on the compute device. Each cached window is
    # a [C,Z,Y,X] stack (~hundreds of MiB), so a large `met_cache_max_hours` on a GPU eats
    # GiBs of scarce device memory (snapshot 2026-06-29: 192 h ≈ 50 GiB of HBM). Host RAM
    # (e.g. Grace's LPDDR5X) is far larger, so we cache there and the per-step physics moves
    # the active window to the device via its existing `.to(device)` calls (cheap over
    # NVLink-C2C). No effect on a CPU run (host == device). Set False for the old behaviour.
    met_cache_on_host: bool = Field(True)
    # Prefetch the next (backward) met hour on a background thread so its I/O overlaps the
    # current window's compute (the run is met-I/O-bound: ~1.6 s/fetch, GPU otherwise idle).
    # Requires `met_cache_on_host` (the prefetch thread must produce HOST tensors — no CUDA
    # off the main thread); auto-disabled with a warning otherwise.
    met_prefetch: bool = Field(True)
    log_every_steps: int = Field(10, ge=0)
    gc_every_steps: int = Field(50, ge=0)
    guard_max_rss_gib: float | None = Field(None, gt=0)
    guard_max_device_allocated_gib: float | None = Field(None, gt=0)
    guard_max_device_reserved_gib: float | None = Field(None, gt=0)
    guard_check_every_steps: int = Field(1, gt=0)


class RunConfig(_Frozen):
    """Top-level GLIDE run configuration.

    Loaded from YAML via :meth:`from_yaml` or constructed in tests via
    :meth:`model_validate`. Sub-configs group related fields so the schema
    stays browseable as it grows.
    """

    io: IOConfig
    simulation: SimulationConfig
    release: ReleaseConfig
    turbulence: TurbulenceConfig
    output_grid: OutputGridConfig
    met_domain: MetDomainConfig
    memory: MemoryConfig = MemoryConfig()
    batch: BatchConfig = BatchConfig()
    # Deep-convection block. Defaulting to "none" makes existing YAMLs without a
    # `convection:` block bit-equivalent to the no-convection runtime path.
    convection: ConvectionConfig = ConvectionConfig()

    @model_validator(mode="after")
    def _check_simulation_length_vs_release(self) -> RunConfig:
        if self.simulation.length_seconds <= self.release.duration_seconds:
            raise ValueError("simulation.length_seconds must be > release.duration_seconds")
        return self

    @model_validator(mode="after")
    def _check_time_bins_cover_simulation(self) -> RunConfig:
        og = self.output_grid
        covered = og.n_time_bins * og.time_bin_seconds
        if (
            og.n_time_bins > 1
            and covered < self.simulation.length_seconds
            and not og.drop_older_ages
        ):
            raise ValueError(
                f"output_grid: {og.n_time_bins} time bins of {og.time_bin_seconds} s cover "
                f"{covered} s, less than simulation.length_seconds="
                f"{self.simulation.length_seconds}, so older residence would be dropped. "
                "Add bins, widen them, or set output_grid.drop_older_ages: true (decision 0016)."
            )
        return self

    @model_validator(mode="after")
    def _check_release_inside_met_domain(self) -> RunConfig:
        md = self.met_domain
        if isinstance(self.release, MultiPointPeriodicReleaseConfig):
            points = [(s.lon, s.lat, s.alt_agl_m, s.name) for s in self.release.sites]
        elif isinstance(self.release, SampleReleaseConfig):
            points = [(s.lon, s.lat, s.alt_max_agl_m, s.name) for s in self.release.samples]
        else:
            lon, lat, alt_agl_m = _release_point(self.release)
            points = [(lon, lat, alt_agl_m, None)]
        for lon, lat, alt_agl_m, name in points:
            tag = f" ({name})" if name else ""
            if not (md.lon_bounds[0] <= lon <= md.lon_bounds[1]):
                raise ValueError(
                    f"release{tag} lon={lon} outside met_domain.lon_bounds={md.lon_bounds}"
                )
            if not (md.lat_bounds[0] <= lat <= md.lat_bounds[1]):
                raise ValueError(
                    f"release{tag} lat={lat} outside met_domain.lat_bounds={md.lat_bounds}"
                )
            if alt_agl_m > md.alt_max_m:
                raise ValueError(
                    f"release{tag} alt_agl_m={alt_agl_m} above met_domain.alt_max_m={md.alt_max_m}"
                )
        return self

    def expand_to_batches(self) -> list[ReleaseBatch]:
        """Expand the configured release into a list of execution batches.

        For ``kind: "point"`` this returns a single batch with a single
        :class:`ConcreteRelease` at ``simulation.start_time``.

        For ``kind: "periodic_point"`` and ``kind: "point_schedule"`` it
        generates one :class:`ConcreteRelease` per release time, chunked into
        batches of at most ``batch.max_releases_per_batch`` consecutive
        releases each.

        Per-release seeds are derived as ``seed + release_idx`` when the
        top-level ``seed`` is non-null; otherwise each release's seed is None
        (non-deterministic). Single-release runs therefore see ``seed_0 == seed``.
        """

        rel = self.release
        max_per_batch = self.batch.max_releases_per_batch

        if isinstance(rel, PointReleaseConfig):
            concrete = ConcreteRelease(
                release_idx=0,
                release_time=self.simulation.start_time,
                lon=float(rel.lon),
                lat=float(rel.lat),
                alt_agl_m=float(rel.alt_agl_m),
                duration_seconds=int(rel.duration_seconds),
                n_particles=int(rel.n_particles),
                seed=rel.seed,
            )
            return [ReleaseBatch(batch_idx=0, releases=(concrete,))]

        if isinstance(rel, MultiPointPeriodicReleaseConfig):
            # One release per (time, site), in TIME-MAJOR order so chunking by
            # max_releases_per_batch yields batches that span contiguous times across ALL
            # sites — a dense active set sharing one met fetch per hour (the efficiency win).
            # Best results when max_releases_per_batch is a multiple of len(sites).
            times = tuple(
                rel.start_time + timedelta(seconds=i * rel.period_seconds)
                for i in range(rel.n_releases)
            )
            seed = rel.seed
            concretes = []
            idx = 0
            for t in times:
                for site in rel.sites:
                    concretes.append(
                        ConcreteRelease(
                            release_idx=idx,
                            release_time=t,
                            lon=float(site.lon),
                            lat=float(site.lat),
                            alt_agl_m=float(site.alt_agl_m),
                            duration_seconds=int(rel.duration_seconds),
                            n_particles=int(rel.n_particles_per_release),
                            seed=(seed + idx) if seed is not None else None,
                            label=site.name,
                        )
                    )
                    idx += 1
        elif isinstance(rel, SampleReleaseConfig):
            # One release per sample, in the order given, each with its own window and
            # (optionally) height range. Seeds derive as seed + release_idx, as for the
            # other multi-release kinds.
            seed = rel.seed
            concretes = [
                ConcreteRelease(
                    release_idx=i,
                    release_time=s.start_time,
                    lon=float(s.lon),
                    lat=float(s.lat),
                    alt_agl_m=float(s.alt_min_agl_m),
                    duration_seconds=int(s.duration_seconds),
                    n_particles=int(rel.n_particles_per_release),
                    seed=(seed + i) if seed is not None else None,
                    label=s.name,
                    alt_top_agl_m=(
                        float(s.alt_max_agl_m) if s.alt_max_agl_m > s.alt_min_agl_m else None
                    ),
                )
                for i, s in enumerate(rel.samples)
            ]
        else:
            if isinstance(rel, PeriodicPointReleaseConfig):
                times = tuple(
                    rel.start_time + timedelta(seconds=i * rel.period_seconds)
                    for i in range(rel.n_releases)
                )
            elif isinstance(rel, PointScheduleReleaseConfig):
                times = rel.times
            else:
                raise TypeError(f"Unsupported release variant: {type(rel).__name__}")

            lon, lat, alt_agl_m = _release_point(rel)
            seed = rel.seed
            concretes = [
                ConcreteRelease(
                    release_idx=i,
                    release_time=t,
                    lon=lon,
                    lat=lat,
                    alt_agl_m=alt_agl_m,
                    duration_seconds=int(rel.duration_seconds),
                    n_particles=int(rel.n_particles_per_release),
                    seed=(seed + i) if seed is not None else None,
                )
                for i, t in enumerate(times)
            ]

        batches: list[ReleaseBatch] = []
        for batch_idx, start in enumerate(range(0, len(concretes), max_per_batch)):
            chunk = tuple(concretes[start : start + max_per_batch])
            batches.append(ReleaseBatch(batch_idx=batch_idx, releases=chunk))
        return batches

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Top-level YAML in {path} must be a mapping")
        return cls.model_validate(data)

    def with_overrides(
        self,
        *,
        device: str | None = None,
        output_uri: str | None = None,
        start_time: datetime | str | None = None,
    ) -> RunConfig:
        """Return a copy with the given CLI-level overrides applied."""

        updates: dict[str, Any] = {}
        if device is not None:
            updates["simulation"] = self.simulation.model_copy(update={"device": device})
        if start_time is not None:
            sim_update = updates.get("simulation", self.simulation)
            updates["simulation"] = sim_update.model_copy(
                update={"start_time": _parse_datetime_utc(start_time)}
            )
        if output_uri is not None:
            updates["io"] = self.io.model_copy(update={"output_uri": output_uri})
        if not updates:
            return self
        return self.model_copy(update=updates)
