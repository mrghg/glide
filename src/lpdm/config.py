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

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _parse_datetime_utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IOConfig(_Frozen):
    zarr_store: str = Field(..., min_length=1, description="ERA5 ARCO Zarr store URI")
    output_uri: str = Field(..., min_length=1, description="Output directory or URI")


class SimulationConfig(_Frozen):
    start_time: datetime = Field(..., description="UTC ISO timestamp at start of release window")
    length_seconds: int = Field(..., gt=0, description="Backward integration length from end of release")
    dt_seconds: int = Field(..., gt=0, description="Integration timestep in seconds")
    device: str = Field("auto", description="Torch device: auto, cpu, cuda, mps, or cuda:N")

    @field_validator("start_time", mode="before")
    @classmethod
    def _coerce_start_time(cls, v: Any) -> datetime:
        return _parse_datetime_utc(v)


ReleaseKind = Literal["point"]


class ReleaseConfig(_Frozen):
    kind: ReleaseKind = "point"
    lon: float
    lat: float
    alt_agl_m: float = Field(..., ge=0)
    duration_seconds: int = Field(..., gt=0)
    n_particles: int = Field(..., gt=0)
    seed: int | None = Field(None, ge=0)


class TurbulenceConfig(_Frozen):
    scheme: str = Field(..., min_length=1)


class OutputGridConfig(_Frozen):
    """Grid onto which residence-time footprints are accumulated."""

    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    n_x: int = Field(..., gt=0)
    n_y: int = Field(..., gt=0)
    z_edges_m: tuple[float, ...] = Field(..., min_length=2)
    n_time_bins: int = Field(..., gt=0)

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

    @field_validator("lon_bounds", "lat_bounds")
    @classmethod
    def _bounds_ascending(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError("bounds must satisfy lower < upper")
        return v


class MemoryConfig(_Frozen):
    met_cache_max_hours: int = Field(2, ge=0)
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

    @model_validator(mode="after")
    def _check_simulation_length_vs_release(self) -> "RunConfig":
        if self.simulation.length_seconds <= self.release.duration_seconds:
            raise ValueError(
                "simulation.length_seconds must be > release.duration_seconds"
            )
        return self

    @model_validator(mode="after")
    def _check_release_inside_met_domain(self) -> "RunConfig":
        md = self.met_domain
        rel = self.release
        if not (md.lon_bounds[0] <= rel.lon <= md.lon_bounds[1]):
            raise ValueError(
                f"release.lon={rel.lon} outside met_domain.lon_bounds={md.lon_bounds}"
            )
        if not (md.lat_bounds[0] <= rel.lat <= md.lat_bounds[1]):
            raise ValueError(
                f"release.lat={rel.lat} outside met_domain.lat_bounds={md.lat_bounds}"
            )
        if rel.alt_agl_m > md.alt_max_m:
            raise ValueError(
                f"release.alt_agl_m={rel.alt_agl_m} above met_domain.alt_max_m={md.alt_max_m}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
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
    ) -> "RunConfig":
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
