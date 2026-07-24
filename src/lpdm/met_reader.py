"""Meteorology readers for LPDM.

This module defines a stable interface for meteorological data access so additional
reanalysis or forecast products can be added later without changing downstream
GPU physics code.

Design goals:
- Read only the particle-cloud bounding box for each model hour.
- Materialize that subset in CPU memory (xarray + dask compute boundary).
- Convert CPU arrays into torch tensors with consistent shape semantics.
- Return hour-start and hour-end fields separately for temporal interpolation.
"""

from __future__ import annotations

import glob
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import numpy as np
import torch
import xarray as xr

from lpdm.runtime import DEVICE
from lpdm.vertical_grid import (
    apply_agl_regrid,
    compute_agl_regrid_weights,
    default_agl_levels,
    slope_correct_w,
    terrain_gradient,
)


GRAVITY_M_S2 = 9.80665
R_DRY_AIR_J_KG_K = 287.05


@dataclass(frozen=True)
class SpatialBounds:
    """3D spatial bounds used to subset meteorology.

    Attributes:
        lon_min: Minimum longitude in degrees east.
        lon_max: Maximum longitude in degrees east.
        lat_min: Minimum latitude in degrees north.
        lat_max: Maximum latitude in degrees north.
        z_min: Minimum altitude (meters above ground level or sea level, depending
            on the configured vertical coordinate convention).
        z_max: Maximum altitude in meters.
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class TimeBounds:
    """Time interval used for one met fetch window.

    For hourly fields used with sub-hour integration, this will typically cover
    exactly one hour, e.g. [12:00, 13:00].
    """

    start: datetime
    end: datetime


@dataclass(frozen=True)
class BoundingBoxRequest:
    """Unified request object passed in from the main orchestrator."""

    spatial: SpatialBounds
    time: TimeBounds


@dataclass(frozen=True)
class MetFieldMetadata:
    """Metadata needed by GPU interpolation and coordinate normalization.

    Arrays correspond to the dimensions used in the returned tensors.
    """

    lon: np.ndarray
    lat: np.ndarray
    level: np.ndarray
    pressure_level_hpa: np.ndarray
    time_start: datetime
    time_end: datetime
    variable_units: Mapping[str, str]


@dataclass(frozen=True)
class HourlyMetTensors:
    """Hour bracketing tensors for fine-step temporal interpolation.

    Shape convention for each tensor:
        [channels, z, y, x]

    `channel_names[i]` identifies the logical variable at axis 0 index `i` (for
    example `("u", "v", "w", "blh", "sp")`). Use `channel(name)` or
    `channels(*names)` to slice by name rather than position.

    `height_agl_m` is the full 3D geopotential-derived height above ground level
    in metres, shape `[z, y, x]` (per-window average of the hour-start/end
    geopotential, which changes negligibly within an hour). It gives true
    per-column layer heights for vertical-gradient schemes (e.g. the Hanna
    free-troposphere N²/Richardson branch), as opposed to the bbox-averaged
    `metadata.level`. May be `None` for synthetic readers that don't model
    geopotential; schemes that need it must check and raise a clear error.
    """

    hour_start: torch.Tensor
    hour_end: torch.Tensor
    metadata: MetFieldMetadata
    channel_names: tuple[str, ...]
    height_agl_m: torch.Tensor | None = None

    def _channel_index(self, name: str) -> int:
        try:
            return self.channel_names.index(name)
        except ValueError as exc:
            available = ", ".join(self.channel_names) or "<none>"
            raise KeyError(
                f"Channel {name!r} not present in met window. Available: {available}"
            ) from exc

    def channel(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(hour_start_slice, hour_end_slice)` for one named channel.

        Each slice has shape `[z, y, x]`.
        """

        idx = self._channel_index(name)
        return self.hour_start[idx], self.hour_end[idx]

    def channels(self, *names: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stacked hour-start and hour-end slices for the named channels.

        Each returned tensor has shape `[len(names), z, y, x]`. Channel order in
        the result matches the order of `names` rather than the underlying tensor.
        """

        indices = [self._channel_index(name) for name in names]
        return self.hour_start[indices], self.hour_end[indices]


class MetReader(ABC):
    """Abstract meteorology reader interface.

    Any concrete reader must provide a method that returns two hourly snapshots
    already loaded in memory and converted to torch tensors.
    """

    @abstractmethod
    def fetch_hourly_window(self, request: BoundingBoxRequest) -> HourlyMetTensors:
        """Fetch and convert meteorology for a single hourly bracket.

        Args:
            request: Bounding-box and time-window request generated from current
                particle cloud extent.

        Returns:
            HourlyMetTensors containing start/end snapshots and metadata.
        """

    @abstractmethod
    def get_time_coverage(self) -> tuple[datetime, datetime]:
        """Return the dataset time coverage as UTC datetimes."""


class ArcoEra5ZarrReader(MetReader):
    """Reader for ARCO ERA5 Zarr stores hosted on Google Cloud Storage.

    Notes:
    - This class is written as a production-ready skeleton; exact variable names
      and coordinate names may differ across ARCO catalog variants.
    - The variable mapping in `DEFAULT_VARIABLE_MAP` should be adapted to the
      specific dataset schema you target.
    """

    # Logical names used by the LPDM engine mapped to dataset variable names.
    DEFAULT_VARIABLE_MAP: Mapping[str, str] = {
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "w": "vertical_velocity",
        "blh": "boundary_layer_height",
        "sp": "surface_pressure",
        "t": "temperature",
        "z": "geopotential",
        "z_sfc": "geopotential_at_surface",
        # Single-level fields used by turbulence schemes (e.g. Hanna). Override
        # to "instantaneous_surface_sensible_heat_flux" if the dataset provides it.
        "ustar": "friction_velocity",
        "shf": "surface_sensible_heat_flux",
        # 3D moisture field used by the deep-convection scheme
        # (EmanuelReducedConvection). Added when the convection scheme requests
        # "q" via required_met_keys().
        "q": "specific_humidity",
    }

    # Default channel order packed into the [C, Z, Y, X] tensors returned by
    # fetch_hourly_window. Override per-instance via the `channel_names` argument
    # to add scheme-specific extras (for example ustar, shf for Hanna).
    DEFAULT_CHANNEL_NAMES: tuple[str, ...] = ("u", "v", "w", "blh", "sp")

    # Logical keys always required for derivations, regardless of channel_names:
    #   t       -> omega -> w conversion
    #   z, z_sfc -> geopotential -> AGL conversion
    _DERIVATION_KEYS: tuple[str, ...] = ("t", "z", "z_sfc")

    # Per-hour processed cache size. Windows walk the hours monotonically, so a
    # handful of entries bridges adjacent windows; the runtime's window-level
    # LRU (memory.met_cache_max_hours) remains the big cache.
    _HOUR_CACHE_MAX: int = 6

    def __init__(
        self,
        zarr_store: str | Sequence[str],
        *,
        variable_map: Mapping[str, str] | None = None,
        channel_names: Sequence[str] | None = None,
        lon_name: str = "longitude",
        lat_name: str = "latitude",
        level_name: str = "level",
        time_name: str = "time",
        chunk_overrides: Mapping[str, int] | None = None,
        accumulation_seconds: int = 3600,
        terrain_following: bool = True,
        agl_levels_m: Sequence[float] | None = None,
        device: torch.device | str = DEVICE,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize reader configuration.

        Args:
            zarr_store: One of:
                * A single store URI (`gs://bucket/path.zarr` or a local path).
                * A local glob pattern (e.g. `~/data/arco-era5/EUROPE_*.zarr`) that
                  expands to one or more local Zarr directories. Glob expansion is
                  local-filesystem only; remote URIs are passed through verbatim.
                * A sequence of store URIs to stitch along the time dimension. All
                  stores must share identical lat/lon/level coordinates.
            variable_map: Optional override for logical-to-dataset variable names.
            channel_names: Optional override for the channels packed into the [C, Z,
                Y, X] hourly tensors. Defaults to `DEFAULT_CHANNEL_NAMES`. Add
                scheme-specific keys (e.g. "ustar", "shf") to make the new fields
                available via `HourlyMetTensors.channel(name)`.
            lon_name: Longitude coordinate name in source dataset.
            lat_name: Latitude coordinate name in source dataset.
            level_name: Vertical coordinate name in source dataset.
            time_name: Time coordinate name in source dataset.
            chunk_overrides: Optional xarray/dask chunk override dict.
            accumulation_seconds: Period over which accumulated surface fluxes (e.g.
                surface_sensible_heat_flux as J/m^2) are aggregated. Used to convert
                accumulated flux fields back to instantaneous W/m^2. Default 3600 s
                matches ERA5 hourly accumulation.
            terrain_following: If True (default), resample the pressure-level met onto
                a fixed terrain-following AGL grid once per window and slope-correct
                the vertical velocity (Finding 7 fix). If False, ship fields on the
                pressure grid with the legacy bbox-mean level array.
            agl_levels_m: Optional explicit ascending AGL height grid (m) for the
                terrain-following resample. Defaults to `default_agl_levels(z_max)`.
            device: Torch target device for returned tensors.
            dtype: Torch dtype for returned tensors.
        """

        self.zarr_stores: tuple[str, ...] = self._resolve_stores(zarr_store)
        # Back-compat single-store attribute for callers that inspect this directly.
        self.zarr_store = self.zarr_stores[0] if len(self.zarr_stores) == 1 else list(self.zarr_stores)
        self.variable_map = dict(variable_map or self.DEFAULT_VARIABLE_MAP)
        self.channel_names: tuple[str, ...] = (
            tuple(channel_names) if channel_names is not None else self.DEFAULT_CHANNEL_NAMES
        )
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.level_name = level_name
        self.time_name = time_name
        self.chunk_overrides = dict(chunk_overrides or {})
        self.accumulation_seconds = int(accumulation_seconds)
        # Terrain-following (hybrid) vertical coordinate: resample pressure levels
        # onto a fixed AGL grid once per window (Finding 7). Off => legacy path that
        # ships fields on the pressure grid with a bbox-mean level array (biased over
        # terrain; kept for A/B and for synthetic stores without geopotential).
        self.terrain_following = bool(terrain_following)
        self.agl_levels_m = (
            np.asarray(agl_levels_m, dtype="float64") if agl_levels_m is not None else None
        )
        # Per-hour processed-tensor cache (perf review 2026-07-16 #4), terrain
        # path only. Consecutive met windows share a physical hour (window k's
        # hour_end == window k+1's hour_start); caching the PROCESSED hour lets
        # adjacent windows share one tensor — the duplicate zarr read, omega->w,
        # and AGL regrid disappear, and downstream window caches hold shared
        # storage instead of two copies. Legacy pressure-grid windows are not
        # hour-cacheable (their level subset is window-dependent, so the same
        # hour can have different shapes in adjacent windows).
        self._hour_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._hour_cache_lock = threading.Lock()
        self.hour_cache_hits = 0
        self.device = torch.device(device)
        self.dtype = dtype

        unknown = [k for k in self.channel_names if k not in self.variable_map]
        if unknown:
            raise KeyError(
                f"channel_names contains keys with no entry in variable_map: {unknown}. "
                f"Add them to variable_map or remove from channel_names."
            )
        if self.accumulation_seconds <= 0:
            raise ValueError("accumulation_seconds must be > 0")

    @property
    def required_variable_keys(self) -> tuple[str, ...]:
        """Logical variable keys required for LPDM stepping.

        Union of `channel_names` (what gets packed into the channel tensor) with
        `_DERIVATION_KEYS` (always-needed for omega->w and AGL conversion).
        """

        return tuple(dict.fromkeys((*self.channel_names, *self._DERIVATION_KEYS)))

    def fetch_hourly_window(self, request: BoundingBoxRequest) -> HourlyMetTensors:
        """Read, slice, materialize, and convert one hourly met window.

        High-level flow:
        1. Open remote Zarr lazily.
        2. Select required variables and bounding box.
        3. Materialize the tiny subset into CPU memory with `.compute()`.
        4. Split into hour-start and hour-end snapshots.
        5. Convert both snapshots to consistent torch tensors.
        """

        t0, t1 = self._canonicalize_hour_bounds(request.time)
        request_hour = BoundingBoxRequest(
            spatial=request.spatial,
            time=TimeBounds(start=t0, end=t1),
        )

        ds = self._open_dataset()
        ds = self._select_variables(ds)

        try:
            if self.terrain_following:
                # Per-hour processing via the hour cache (perf #4): the shared
                # hour of adjacent windows is fetched/regridded once and its
                # tensor SHARED between the two windows.
                bundle_start = self._processed_hour(ds, t0, request.spatial)
                bundle_end = self._processed_hour(ds, t1, request.spatial)
            else:
                start_request = BoundingBoxRequest(
                    spatial=request_hour.spatial,
                    time=TimeBounds(start=t0, end=t0),
                )
                end_request = BoundingBoxRequest(
                    spatial=request_hour.spatial,
                    time=TimeBounds(start=t1, end=t1),
                )

                ds_start_lazy = self._slice_spatial_temporal(ds, start_request)
                ds_end_lazy = self._slice_spatial_temporal(ds, end_request)

                # Ensure time subsets only have 1 step before accessing values
                t0_sel, _ = self._coerce_time_bounds_for_dataset(ds_start_lazy, t0, t0)
                _, t1_sel = self._coerce_time_bounds_for_dataset(ds_end_lazy, t1, t1)
                ds_start_lazy = ds_start_lazy.sel({self.time_name: t0_sel})
                ds_end_lazy = ds_end_lazy.sel({self.time_name: t1_sel})

                # Subset vertical levels using lazy Dask arrays (only reads Z/Z_SFC
                # into memory to evaluate the bound)
                ds_start_sub, ds_end_sub, level_agl_m, height_agl_3d = self._subset_vertical_levels_by_agl(
                    ds_start=ds_start_lazy,
                    ds_end=ds_end_lazy,
                    spatial=request.spatial,
                )

                # Materialize only the fully sliced bounding boxes
                ds_start = ds_start_sub.compute()
                ds_end = ds_end_sub.compute()
        finally:
            close_fn = getattr(ds, "close", None)
            if callable(close_fn):
                close_fn()

        if self.terrain_following:
            agl = bundle_start["level"]
            metadata = MetFieldMetadata(
                lon=bundle_start["lon"],
                lat=bundle_start["lat"],
                level=agl,
                pressure_level_hpa=bundle_start["pressure_level_hpa"],
                time_start=t0,
                time_end=t1,
                variable_units=bundle_start["units"],
            )
            height_broadcast = np.broadcast_to(
                agl[:, None, None],
                (agl.size, bundle_start["lat"].size, bundle_start["lon"].size),
            ).copy()
            return HourlyMetTensors(
                hour_start=bundle_start["tensor"],
                hour_end=bundle_end["tensor"],
                metadata=metadata,
                channel_names=self.channel_names,
                height_agl_m=torch.as_tensor(height_broadcast, dtype=self.dtype, device=self.device),
            )

        hour_start = self._dataset_to_channel_tensor(ds_start)
        hour_end = self._dataset_to_channel_tensor(ds_end)

        metadata = MetFieldMetadata(
            lon=np.asarray(ds_start[self.lon_name].values),
            lat=np.asarray(ds_start[self.lat_name].values),
            level=level_agl_m,
            pressure_level_hpa=np.asarray(ds_start[self.level_name].values),
            time_start=t0,
            time_end=t1,
            variable_units=self._read_variable_units(ds_start),
        )

        height_agl_m = torch.as_tensor(height_agl_3d, dtype=self.dtype, device=self.device)

        return HourlyMetTensors(
            hour_start=hour_start,
            hour_end=hour_end,
            metadata=metadata,
            channel_names=self.channel_names,
            height_agl_m=height_agl_m,
        )

    def _processed_hour(self, ds: xr.Dataset, when: datetime, spatial: SpatialBounds) -> dict:
        """Fetch + fully process ONE met hour onto the terrain-following AGL grid,
        through the per-hour cache (perf review 2026-07-16 #4).

        Returns a bundle dict: ``tensor`` ([C, Za, Y, X], on the reader's
        device/dtype), ``lon``/``lat``/``level``/``pressure_level_hpa``/``units``.
        Cache hits return the SAME bundle object, so adjacent windows share the
        common hour's tensor — treat bundle tensors as read-only downstream.
        """
        key = (
            when,
            round(spatial.lon_min, 6), round(spatial.lon_max, 6),
            round(spatial.lat_min, 6), round(spatial.lat_max, 6),
            round(spatial.z_min, 3), round(spatial.z_max, 3),
        )
        with self._hour_cache_lock:
            bundle = self._hour_cache.get(key)
            if bundle is not None:
                self._hour_cache.move_to_end(key)
                self.hour_cache_hits += 1
                return bundle

        hour_request = BoundingBoxRequest(
            spatial=spatial, time=TimeBounds(start=when, end=when)
        )
        ds_hour_lazy = self._slice_spatial_temporal(ds, hour_request)
        t_sel, _ = self._coerce_time_bounds_for_dataset(ds_hour_lazy, when, when)
        ds_hour_lazy = ds_hour_lazy.sel({self.time_name: t_sel})

        ds_hour_sub, height_agl_p = self._subset_vertical_levels_single_hour(
            ds_hour_lazy, spatial
        )
        ds_hour = ds_hour_sub.compute()

        tensor_p = self._dataset_to_channel_tensor(ds_hour)
        lon_arr = np.asarray(ds_hour[self.lon_name].values)
        lat_arr = np.asarray(ds_hour[self.lat_name].values)
        tensor_agl, level_out, pressure_level_hpa = self._resample_hour_to_agl(
            tensor_p, height_agl_p, ds_hour, lat_arr, lon_arr, spatial.z_max
        )
        bundle = dict(
            tensor=tensor_agl,
            lon=lon_arr,
            lat=lat_arr,
            level=level_out,
            pressure_level_hpa=pressure_level_hpa,
            units=self._read_variable_units(ds_hour),
        )
        with self._hour_cache_lock:
            self._hour_cache[key] = bundle
            while len(self._hour_cache) > self._HOUR_CACHE_MAX:
                self._hour_cache.popitem(last=False)
        return bundle

    def _subset_vertical_levels_single_hour(
        self, ds_hour: xr.Dataset, spatial: SpatialBounds
    ) -> tuple[xr.Dataset, np.ndarray]:
        """Single-hour analogue of `_subset_vertical_levels_by_agl` (terrain path).

        Subsets pressure levels by THIS hour's own AGL bounds and returns the
        per-column AGL heights — not window-averaged. Geopotential drift within a
        window is negligible (the same approximation the window-average made), and
        per-hour heights make an hour's processed tensor identical in both
        adjacent windows, which is what makes it cacheable.
        """
        if (
            int(ds_hour.sizes.get(self.lon_name, 0)) == 0
            or int(ds_hour.sizes.get(self.lat_name, 0)) == 0
        ):
            raise ValueError(
                "Cannot subset vertical levels: empty spatial selection for met hour."
            )
        level_agl = self._compute_level_agl_m(ds_hour)
        if level_agl.size == 0:
            raise ValueError("Computed empty AGL arrays for met hour.")
        if not np.isfinite(level_agl).all():
            raise ValueError(
                "Geopotential-derived AGL heights contain non-finite values in the "
                "requested region/time window. The meteorology store is incomplete or "
                "corrupted, or the source dataset is missing required geopotential "
                "coverage for this slice."
            )
        level_values = np.asarray(ds_hour[self.level_name].values)
        level_mask = (np.nanmax(level_agl, axis=(1, 2)) >= spatial.z_min) & (
            np.nanmin(level_agl, axis=(1, 2)) <= spatial.z_max
        )
        if not np.any(level_mask):
            raise ValueError(
                "No vertical levels intersect requested AGL bounds "
                f"[{spatial.z_min}, {spatial.z_max}] m"
            )
        ds_sub = ds_hour.sel({self.level_name: level_values[level_mask]})
        return ds_sub, self._compute_level_agl_m(ds_sub)

    def _resample_hour_to_agl(
        self,
        tensor_p: torch.Tensor,
        height_agl_p: np.ndarray,
        ds_hour: xr.Dataset,
        lat_arr: np.ndarray,
        lon_arr: np.ndarray,
        z_max_m: float,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Resample ONE hour's pressure-level channels onto the fixed AGL grid.

        FLEXPART-style hybrid coordinate (Finding 7): per-column regrid excluding
        sub-surface levels, then slope-correct `w` into the AGL frame. Runs on the
        met prefetch thread, once per hour (perf #3/#4) — never in the per-step
        hot path.
        """
        agl_levels = (
            self.agl_levels_m if self.agl_levels_m is not None else default_agl_levels(z_max_m)
        )
        weights = compute_agl_regrid_weights(height_agl_p, agl_levels)
        arr_agl = apply_agl_regrid(tensor_p.detach().cpu().numpy(), weights)

        # Slope-correct w into the terrain-following frame (needs u, v, w present).
        if all(k in self.channel_names for k in ("u", "v", "w")):
            z_sfc = np.asarray(ds_hour[self.variable_map["z_sfc"]].values)
            if z_sfc.ndim == 3:
                z_sfc = z_sfc[0]
            terrain_m = z_sfc / GRAVITY_M_S2
            dhdx, dhdy = terrain_gradient(terrain_m, lat_arr, lon_arr)
            ui = self.channel_names.index("u")
            vi = self.channel_names.index("v")
            wi = self.channel_names.index("w")
            arr_agl[wi] = slope_correct_w(
                arr_agl[wi], arr_agl[ui], arr_agl[vi], dhdx, dhdy,
                agl_levels, float(agl_levels[-1]),
            )

        # Bbox-mean pressure per AGL level (hPa) — Hanna-FT / Emanuel read this as a
        # per-level (horizontally-uniform) pressure, so the AGL-grid mean is the same
        # class of approximation they already make on the pressure grid.
        level_hpa = np.asarray(ds_hour[self.level_name].values, dtype="float64")
        # Materialised (not broadcast_to): the torch-backed regrid can't wrap
        # zero-stride numpy views.
        pressure_p = np.ascontiguousarray(
            np.broadcast_to(level_hpa[:, None, None], height_agl_p.shape)
        )
        pressure_agl = apply_agl_regrid(pressure_p[None], weights)[0]
        pressure_level_hpa = pressure_agl.mean(axis=(1, 2))  # already hPa

        tensor_agl = torch.as_tensor(arr_agl, dtype=self.dtype, device=self.device)
        return tensor_agl, np.asarray(agl_levels, dtype="float64"), pressure_level_hpa

    def get_time_coverage(self) -> tuple[datetime, datetime]:
        """Return the first and last timestamps available in the dataset."""

        ds = self._open_dataset()
        try:
            time_coord = ds[self.time_name]
            if int(time_coord.size) == 0:
                raise ValueError(f"Dataset time coordinate {self.time_name!r} is empty")

            values = np.asarray(time_coord.values)
            start = self._time_value_to_utc_datetime(values[0])
            end = self._time_value_to_utc_datetime(values[-1])
            return start, end
        finally:
            close_fn = getattr(ds, "close", None)
            if callable(close_fn):
                close_fn()

    @staticmethod
    def _resolve_stores(zarr_store: str | Sequence[str]) -> tuple[str, ...]:
        """Normalize the user-supplied store argument into a tuple of URIs.

        Local glob patterns are expanded; remote URIs (e.g. `gs://...`) are passed
        through unchanged. The returned tuple is non-empty and deduplicated while
        preserving order.
        """

        if isinstance(zarr_store, str):
            candidates: list[str] = [zarr_store]
        else:
            candidates = list(zarr_store)
            if not candidates:
                raise ValueError("zarr_store list cannot be empty")
            if not all(isinstance(s, str) and s for s in candidates):
                raise ValueError("zarr_store list entries must all be non-empty strings")

        resolved: list[str] = []
        for candidate in candidates:
            expanded = os.path.expandvars(os.path.expanduser(candidate))
            is_remote = "://" in expanded
            has_glob = any(c in expanded for c in "*?[")
            if has_glob and not is_remote:
                matches = sorted(glob.glob(expanded))
                if not matches:
                    raise FileNotFoundError(
                        f"Glob pattern {candidate!r} matched no Zarr stores"
                    )
                resolved.extend(matches)
            else:
                resolved.append(expanded)

        # Dedupe while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for store in resolved:
            if store not in seen:
                seen.add(store)
                unique.append(store)

        return tuple(unique)

    def _ensure_monotonic_longitude(self, ds: xr.Dataset) -> xr.Dataset:
        """Guarantee a strictly ascending longitude coordinate.

        ARCO ERA5 uses 0..360 longitude. Regional subsets that cross Greenwich
        (e.g. our EUROPE domain) are stored as the concatenation of two halves
        — [262..359.75] then [0..39.5] — which leaves a non-monotonic 1D index
        on disk. Any pandas-style `.sel(slice(...))` against that coord raises
        KeyError, and downstream xarray/dask graph optimisations that prune
        chunks by coordinate range silently break, causing the runtime to
        materialise the full domain on every fetch.

        Mapping values >= 180 to (value - 360) restores monotonicity in the
        common case (data laid out as `[180..360, 0..180)` storage order with
        a single wrap) via a lazy `assign_coords` that does not touch chunk
        data. If a single convention swap is insufficient (data scrambled or
        descending), an explicit `isel` re-sort runs as a fallback. The
        downstream slice path assumes ascending order, so descending-monotonic
        layouts are normalised too.
        """

        lon = np.asarray(ds[self.lon_name].values, dtype=np.float64)
        if lon.size <= 1:
            return ds

        if np.all(np.diff(lon) > 0):
            return ds  # already strictly ascending

        new_lon = np.where(lon >= 180.0, lon - 360.0, lon)
        if np.all(np.diff(new_lon) > 0):
            return ds.assign_coords({self.lon_name: new_lon})

        order = np.argsort(new_lon)
        return ds.assign_coords({self.lon_name: new_lon}).isel({self.lon_name: order})

    def _open_dataset(self) -> xr.Dataset:
        """Open the configured Zarr store(s) lazily.

        For a single store this is a straight `xr.open_zarr` call. For multiple
        stores, each is opened independently, sorted by first timestamp, validated
        for matching spatial/level coordinates, and concatenated along the time
        dimension. Any duplicate timestamps (e.g. from month-boundary overlap)
        are collapsed by keeping the first occurrence. The returned dataset always
        has a strictly monotonic longitude coordinate.
        """

        if len(self.zarr_stores) == 1:
            ds = xr.open_zarr(
                self.zarr_stores[0],
                consolidated=True,
                chunks=self.chunk_overrides or "auto",
            )
            return self._ensure_monotonic_longitude(ds)

        datasets = [
            self._ensure_monotonic_longitude(
                xr.open_zarr(store, consolidated=True, chunks=self.chunk_overrides or "auto")
            )
            for store in self.zarr_stores
        ]

        order = sorted(
            range(len(datasets)),
            key=lambda i: np.asarray(datasets[i][self.time_name].values).min(),
        )
        datasets = [datasets[i] for i in order]
        ordered_stores = [self.zarr_stores[i] for i in order]

        ref = datasets[0]
        for i, ds in enumerate(datasets[1:], 1):
            for coord in (self.lat_name, self.lon_name, self.level_name):
                if coord not in ds.coords and coord not in ds.variables:
                    raise ValueError(
                        f"Zarr store {ordered_stores[i]!r} is missing coordinate {coord!r}"
                    )
                if not np.array_equal(
                    np.asarray(ref[coord].values), np.asarray(ds[coord].values)
                ):
                    raise ValueError(
                        f"Zarr store {ordered_stores[i]!r} has {coord!r} coordinate "
                        f"that does not match {ordered_stores[0]!r}; cannot stitch "
                        "stores along time."
                    )

        combined = xr.concat(
            datasets,
            dim=self.time_name,
            coords="minimal",
            compat="override",
            join="exact",
        )

        times = np.asarray(combined[self.time_name].values)
        _, first_idx = np.unique(times, return_index=True)
        if first_idx.size != times.size:
            combined = combined.isel({self.time_name: np.sort(first_idx)})

        return combined

    def _select_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Select only required meteorological fields."""

        dataset_vars: Sequence[str] = tuple(self.variable_map[k] for k in self.required_variable_keys)
        missing = [name for name in dataset_vars if name not in ds.variables]
        if missing:
            raise KeyError(f"Required variables missing from dataset: {missing}")
        return ds[list(dataset_vars)]

    def _read_variable_units(self, ds: xr.Dataset) -> dict[str, str]:
        """Read units attrs for all configured logical variables."""

        units: dict[str, str] = {}
        for key in self.required_variable_keys:
            var_name = self.variable_map[key]
            unit = str(ds[var_name].attrs.get("units", "")).strip()
            if not unit:
                raise ValueError(
                    f"Variable {var_name!r} is missing units metadata; cannot safely normalize fields"
                )
            units[key] = unit
        return units

    def _slice_spatial_temporal(self, ds: xr.Dataset, request: BoundingBoxRequest) -> xr.Dataset:
        """Apply bounding-box slicing in physical coordinates.

        Important details to finalize during implementation:
        - Longitude wrap handling around 180/-180.
        - Coordinate ordering (ERA5 latitude often descending).
        - Vertical coordinate transform if data are pressure levels while the
          particle state uses geometric altitude.
        """

        spatial = request.spatial
        time = request.time
        time_start, time_end = self._coerce_time_bounds_for_dataset(ds, time.start, time.end)

        lon_values = np.asarray(ds[self.lon_name].values, dtype=np.float64)
        lon_is_360 = float(np.nanmin(lon_values)) >= 0.0 and float(np.nanmax(lon_values)) > 180.0
        if lon_is_360:
            lon_min = spatial.lon_min % 360.0
            lon_max = spatial.lon_max % 360.0
        else:
            lon_min = ((spatial.lon_min + 180.0) % 360.0) - 180.0
            lon_max = ((spatial.lon_max + 180.0) % 360.0) - 180.0

        lon_coord = ds[self.lon_name]
        
        # Slice temporal and latitude bounds first to dramatically reduce 
        # the dask graph size and memory footprint before any longitude operations.
        ds = ds.sel({
            self.time_name: slice(time_start, time_end),
        })
        
        lat_values = ds[self.lat_name].values
        lat_descending = lat_values[0] > lat_values[-1]
        lat_slice = (
            slice(spatial.lat_max, spatial.lat_min)
            if lat_descending
            else slice(spatial.lat_min, spatial.lat_max)
        )
        ds = ds.sel({self.lat_name: lat_slice})

        if int(ds.sizes.get(self.lat_name, 0)) == 0:
            raise ValueError(
                "Latitude subset is empty after slicing. "
                f"Requested [{spatial.lat_min}, {spatial.lat_max}] deg."
            )

        # Slice longitude safely without .where(drop=True) over massive datasets
        if lon_min <= lon_max:
            ds = ds.sel({self.lon_name: slice(lon_min, lon_max)})
        else:
            ds1 = ds.sel({self.lon_name: slice(lon_min, None)})
            ds2 = ds.sel({self.lon_name: slice(None, lon_max)})
            ds = xr.concat([ds1, ds2], dim=self.lon_name)

        if int(ds.sizes.get(self.lon_name, 0)) == 0:
            lon_convention = "0..360" if lon_is_360 else "-180..180"
            raise ValueError(
                "Longitude subset is empty after slicing. "
                f"Requested [{spatial.lon_min}, {spatial.lon_max}] deg; "
                f"interpreted as [{lon_min}, {lon_max}] in {lon_convention} coordinates."
            )

        return ds

    def _subset_vertical_levels_by_agl(
        self,
        ds_start: xr.Dataset,
        ds_end: xr.Dataset,
        spatial: SpatialBounds,
    ) -> tuple[xr.Dataset, xr.Dataset, np.ndarray, np.ndarray]:
        """Subset pressure levels using geopotential-derived geometric AGL bounds.

        Returns ``(ds_start_sub, ds_end_sub, level_agl_mean, height_agl_3d)`` where
        ``level_agl_mean`` is the per-level spatial-mean AGL (1D, for metadata) and
        ``height_agl_3d`` is the full ``[Z, Y, X]`` AGL height (per-window average
        of start/end), for per-column vertical-gradient schemes.
        """

        if int(ds_start.sizes.get(self.lon_name, 0)) == 0 or int(ds_start.sizes.get(self.lat_name, 0)) == 0:
            raise ValueError(
                "Cannot subset vertical levels: empty spatial selection in ds_start "
                f"(lat={int(ds_start.sizes.get(self.lat_name, 0))}, lon={int(ds_start.sizes.get(self.lon_name, 0))})."
            )
        if int(ds_end.sizes.get(self.lon_name, 0)) == 0 or int(ds_end.sizes.get(self.lat_name, 0)) == 0:
            raise ValueError(
                "Cannot subset vertical levels: empty spatial selection in ds_end "
                f"(lat={int(ds_end.sizes.get(self.lat_name, 0))}, lon={int(ds_end.sizes.get(self.lon_name, 0))})."
            )

        level_agl_start = self._compute_level_agl_m(ds_start)
        level_agl_end = self._compute_level_agl_m(ds_end)

        if level_agl_start.size == 0 or level_agl_end.size == 0:
            raise ValueError(
                "Computed empty AGL arrays. This usually means the spatial subset is empty or the source "
                "geopotential fields are missing the expected [level, lat, lon] coverage."
            )

        if not np.isfinite(level_agl_start).all() or not np.isfinite(level_agl_end).all():
            raise ValueError(
                "Geopotential-derived AGL heights contain non-finite values in the requested region/time window. "
                "The meteorology store is incomplete or corrupted, or the source dataset is missing required "
                "geopotential coverage for this slice."
            )

        level_values = np.asarray(ds_start[self.level_name].values)

        level_agl_min = np.fmin(
            np.nanmin(level_agl_start, axis=(1, 2)),
            np.nanmin(level_agl_end, axis=(1, 2)),
        )
        level_agl_max = np.fmax(
            np.nanmax(level_agl_start, axis=(1, 2)),
            np.nanmax(level_agl_end, axis=(1, 2)),
        )
        level_mask = (level_agl_max >= spatial.z_min) & (level_agl_min <= spatial.z_max)

        if not np.any(level_mask):
            raise ValueError(
                "No vertical levels intersect requested AGL bounds "
                f"[{spatial.z_min}, {spatial.z_max}] m"
            )

        selected_levels = level_values[level_mask]
        ds_start_sub = ds_start.sel({self.level_name: selected_levels})
        ds_end_sub = ds_end.sel({self.level_name: selected_levels})

        level_agl_start_sub = self._compute_level_agl_m(ds_start_sub)
        level_agl_end_sub = self._compute_level_agl_m(ds_end_sub)
        # Per-window average 3D height [Z, Y, X]; geopotential barely shifts within
        # an hour so a single averaged field is sufficient for gradient schemes.
        height_agl_3d = 0.5 * (level_agl_start_sub + level_agl_end_sub)
        level_agl_m = 0.5 * (
            np.mean(level_agl_start_sub, axis=(1, 2)) + np.mean(level_agl_end_sub, axis=(1, 2))
        )

        return ds_start_sub, ds_end_sub, level_agl_m, height_agl_3d

    def _compute_level_agl_m(self, ds_time_slice: xr.Dataset) -> np.ndarray:
        """Compute geometric height above ground level in meters for each level."""

        z_name = self.variable_map["z"]
        z_sfc_name = self.variable_map["z_sfc"]
        z_level = np.asarray(ds_time_slice[z_name].values)
        z_sfc = np.asarray(ds_time_slice[z_sfc_name].values)

        self._validate_units("z", str(ds_time_slice[z_name].attrs.get("units", "")))
        self._validate_units("z_sfc", str(ds_time_slice[z_sfc_name].attrs.get("units", "")))

        if z_level.ndim != 3:
            raise ValueError("geopotential must be a 3D field [z, y, x]")
        if z_sfc.ndim == 2:
            z_sfc_3d = z_sfc[None, :, :]
        elif z_sfc.ndim == 3:
            z_sfc_3d = z_sfc
        else:
            raise ValueError("geopotential_at_surface must be a 2D or 3D field")

        # NOT clamped at 0: pressure levels below ground legitimately have negative
        # AGL, and clamping them collapsed every sub-surface level onto z=0. That
        # gave zero-thickness layers, which `_vertical_gradient` then divided by
        # (guarded only by its 1e-6 floor) -> |dtheta/dz| up to 5e6 K/m over the
        # Alps vs a true max of 0.1. Consumers that need a positive height clamp
        # locally (e.g. `free_trop_diffusivity(height.clamp(min=Z_MIN_M), ...)`).
        return (z_level - z_sfc_3d) / GRAVITY_M_S2

    def _pressure_levels_to_pa(self, ds_time_slice: xr.Dataset) -> np.ndarray:
        """Convert vertical coordinate values to pressure in Pascals."""

        level_vals = np.asarray(ds_time_slice[self.level_name].values, dtype=np.float64)
        level_units = self._normalize_units(str(ds_time_slice[self.level_name].attrs.get("units", "")))

        if level_units in {"pa", "pascal", "pascals"}:
            return level_vals
        if level_units in {"hpa", "hectopascal", "hectopascals", "mbar", "mb", "millibar"}:
            return level_vals * 100.0

        return level_vals * 100.0 if float(np.nanmax(level_vals)) <= 2000.0 else level_vals

    def _coerce_time_bounds_for_dataset(
        self,
        ds: xr.Dataset,
        start: datetime,
        end: datetime,
    ) -> tuple[datetime | np.datetime64, datetime | np.datetime64]:
        """Match request datetime objects to dataset time coordinate representation."""

        time_coord = ds[self.time_name]
        if np.issubdtype(time_coord.dtype, np.datetime64):
            start_utc = start.astimezone(timezone.utc).replace(tzinfo=None)
            end_utc = end.astimezone(timezone.utc).replace(tzinfo=None)
            return np.datetime64(start_utc), np.datetime64(end_utc)
        return start, end

    def _time_value_to_utc_datetime(self, value: object) -> datetime:
        """Normalize a dataset time coordinate value to UTC datetime."""

        value_array = np.asarray(value)
        if np.issubdtype(value_array.dtype, np.datetime64):
            iso_value = np.datetime_as_string(value_array.astype("datetime64[s]"), timezone="UTC")
            return datetime.fromisoformat(iso_value.replace("Z", "+00:00"))

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        raise TypeError(f"Unsupported time coordinate value type: {type(value)!r}")

    def _canonicalize_hour_bounds(self, bounds: TimeBounds) -> tuple[datetime, datetime]:
        """Return exact hour start/end timestamps used for met interpolation."""

        t0 = bounds.start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        t1 = t0 + timedelta(hours=1)
        return t0, t1

    def _dataset_to_channel_tensor(self, ds_time_slice: xr.Dataset) -> torch.Tensor:
        """Convert one xarray time-slice into [channels, z, y, x] torch tensor."""

        channels: list[np.ndarray] = []
        logical_keys = self.channel_names
        units_by_key = self._read_variable_units(ds_time_slice)

        temperature = np.asarray(ds_time_slice[self.variable_map["t"]].values)
        if temperature.ndim != 3:
            raise ValueError("temperature must be a 3D field [z, y, x] for omega->w conversion")
        self._validate_units("t", units_by_key["t"])

        for key in logical_keys:
            var_name = self.variable_map[key]
            arr = np.asarray(ds_time_slice[var_name].values)
            self._validate_units(key, units_by_key[key])

            # Ensure shape is [z, y, x]. For 2D fields (e.g. BLH), broadcast over z.
            if arr.ndim == 2:
                z_size = int(ds_time_slice[self.level_name].size)
                arr = np.broadcast_to(arr[None, :, :], (z_size, *arr.shape))
            elif arr.ndim != 3:
                raise ValueError(
                    f"Unsupported variable rank for {var_name}: expected 2D or 3D, got {arr.ndim}D"
                )

            if key == "w":
                arr = self._convert_vertical_velocity_to_m_s(
                    omega_or_w=arr,
                    units=units_by_key[key],
                    level_hpa=self._pressure_levels_to_pa(ds_time_slice),
                    temperature_k=temperature,
                )
            elif key == "shf":
                arr = self._convert_shf_to_w_per_m2(arr, units_by_key[key])

            channels.append(arr)

        stacked = np.stack(channels, axis=0)  # [channels, z, y, x]
        tensor = torch.as_tensor(stacked, dtype=self.dtype, device=self.device)
        return tensor

    def _validate_units(self, logical_key: str, units: str) -> None:
        """Validate units for each logical meteorological variable."""

        norm = self._normalize_units(units)

        if logical_key in ("u", "v", "w"):
            if self._is_velocity_units(norm):
                return
            if logical_key == "w" and self._is_pressure_tendency_units(norm):
                return
            raise ValueError(f"Unsupported units {units!r} for variable {logical_key!r}")

        if logical_key in ("blh",):
            if norm in {"m", "meter", "metre", "meters", "metres"}:
                return
            raise ValueError(f"Unsupported units {units!r} for variable {logical_key!r}")

        if logical_key in ("sp",):
            if norm in {"pa", "pascal", "pascals"}:
                return
            raise ValueError(f"Unsupported units {units!r} for variable {logical_key!r}")

        if logical_key in ("t",):
            if norm in {"k", "kelvin"}:
                return
            raise ValueError(f"Unsupported units {units!r} for variable {logical_key!r}")

        if logical_key in ("z", "z_sfc"):
            if norm in {
                "m**2s**-2",
                "m2/s2",
                "m^2s^-2",
                "m2s-2",
                "m2s**-2",
                "m^2/s^2",
                "m2/s^2",
            }:
                return
            raise ValueError(f"Unsupported units {units!r} for variable {logical_key!r}")

        if logical_key in ("ustar",):
            if self._is_velocity_units(norm):
                return
            raise ValueError(f"Unsupported units {units!r} for variable {logical_key!r}")

        if logical_key in ("shf",):
            if self._is_flux_density_units(norm) or self._is_flux_accumulated_units(norm):
                return
            raise ValueError(f"Unsupported units {units!r} for variable {logical_key!r}")

        if logical_key in ("q",):
            # Specific humidity is dimensionless (kg/kg). ARCO ERA5 publishes
            # it with units "kg kg**-1" or sometimes empty. Accept both.
            if norm in {"kg/kg", "kgkg-1", "kgkg**-1", "kg*kg-1", "kg*kg**-1", "1", ""}:
                return
            raise ValueError(f"Unsupported units {units!r} for variable {logical_key!r}")

    def _convert_vertical_velocity_to_m_s(
        self,
        omega_or_w: np.ndarray,
        units: str,
        level_hpa: np.ndarray,
        temperature_k: np.ndarray,
    ) -> np.ndarray:
        """Convert vertical velocity to geometric m/s when needed.

        ERA5 pressure-level vertical velocity is often provided as omega = dp/dt
        with units Pa/s. The LPDM particle state uses geometric altitude z (m),
        so advection requires dz/dt in m/s:

            w = dz/dt = -(R_d * T / (g * p)) * omega
        """

        norm = self._normalize_units(units)
        if self._is_velocity_units(norm):
            return omega_or_w
        if not self._is_pressure_tendency_units(norm):
            raise ValueError(f"Unsupported units {units!r} for vertical velocity")

        p_pa = np.asarray(level_hpa, dtype=np.float64)
        if p_pa.ndim != 1:
            raise ValueError("Vertical coordinate must be 1D pressure levels")

        p_3d = p_pa[:, None, None]
        omega = np.asarray(omega_or_w, dtype=np.float64)
        temp = np.asarray(temperature_k, dtype=np.float64)
        w_m_s = -(R_DRY_AIR_J_KG_K * temp / (GRAVITY_M_S2 * p_3d)) * omega
        return w_m_s

    @staticmethod
    def _normalize_units(units: str) -> str:
        """Normalize unit strings for robust comparisons."""

        return units.strip().lower().replace(" ", "")

    @staticmethod
    def _is_velocity_units(norm_units: str) -> bool:
        return norm_units in {"m/s", "m.s-1", "ms-1", "ms**-1", "m*s^-1", "m*s**-1"}

    @staticmethod
    def _is_pressure_tendency_units(norm_units: str) -> bool:
        return norm_units in {
            "pa/s",
            "pas-1",
            "pas**-1",
            "pa*s^-1",
            "pa*s**-1",
            "pa.s-1",
        }

    @staticmethod
    def _is_flux_density_units(norm_units: str) -> bool:
        """Recognize instantaneous heat-flux units (W/m^2)."""

        return norm_units in {
            "w/m**2",
            "wm**-2",
            "wm-2",
            "w/m^2",
            "w/m2",
            "w*m**-2",
            "w*m^-2",
            "w.m-2",
        }

    @staticmethod
    def _is_flux_accumulated_units(norm_units: str) -> bool:
        """Recognize accumulated heat-flux units (J/m^2)."""

        return norm_units in {
            "j/m**2",
            "jm**-2",
            "jm-2",
            "j/m^2",
            "j/m2",
            "j*m**-2",
            "j*m^-2",
            "j.m-2",
        }

    def _convert_shf_to_w_per_m2(self, arr: np.ndarray, units: str) -> np.ndarray:
        """Return sensible heat flux in W/m^2, POSITIVE UPWARD.

        Two conversions happen here:

        1. De-accumulation: ECMWF/ARCO ERA5 stores `surface_sensible_heat_flux`
           accumulated over the hour (J/m^2); dividing by `accumulation_seconds`
           gives the mean W/m^2. Instantaneous W/m^2 fields pass through.
        2. Sign flip: ECMWF uses the convention **positive = downward** for
           surface energy fluxes, so a daytime (upward) sensible heat flux is
           stored NEGATIVE. GLIDE's boundary-layer physics
           (`obukhov_length`, `convective_velocity` in `turbulence/hanna.py`)
           assumes **positive = upward**. We negate here so the whole downstream
           pipeline sees the physics convention.

        NOTE: the ARCO store's CF `standard_name` is
        `surface_upward_sensible_heat_flux`, which is MISLABELLED — the data
        follows the ECMWF downward-positive convention (verified against the
        EUROPE store 2026-07-02: midday-over-land sshf is negative). Trust the
        convention, not the attribute. The de-accumulated
        `instantaneous_surface_sensible_heat_flux` alternative uses the same
        downward-positive convention, so the flip applies to both unit branches.
        """

        norm = self._normalize_units(units)
        if self._is_flux_density_units(norm):
            flux_w_m2 = arr
        elif self._is_flux_accumulated_units(norm):
            flux_w_m2 = arr / float(self.accumulation_seconds)
        else:
            raise ValueError(f"Unsupported units {units!r} for surface_sensible_heat_flux")
        # ECMWF downward-positive -> GLIDE upward-positive.
        return -flux_w_m2
