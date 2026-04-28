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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import numpy as np
import torch
import xarray as xr

from lpdm.runtime import DEVICE


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

    Channel order is determined by the reader implementation and should be
    documented centrally (for example: [u, v, w, blh, sp]).
    """

    hour_start: torch.Tensor
    hour_end: torch.Tensor
    metadata: MetFieldMetadata


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
    }

    def __init__(
        self,
        zarr_store: str,
        *,
        variable_map: Mapping[str, str] | None = None,
        lon_name: str = "longitude",
        lat_name: str = "latitude",
        level_name: str = "level",
        time_name: str = "time",
        chunk_overrides: Mapping[str, int] | None = None,
        device: torch.device | str = DEVICE,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize reader configuration.

        Args:
            zarr_store: GCS path (for example `gs://bucket/path/to.zarr`).
            variable_map: Optional override for logical-to-dataset variable names.
            lon_name: Longitude coordinate name in source dataset.
            lat_name: Latitude coordinate name in source dataset.
            level_name: Vertical coordinate name in source dataset.
            time_name: Time coordinate name in source dataset.
            chunk_overrides: Optional xarray/dask chunk override dict.
            device: Torch target device for returned tensors.
            dtype: Torch dtype for returned tensors.
        """

        self.zarr_store = zarr_store
        self.variable_map = dict(variable_map or self.DEFAULT_VARIABLE_MAP)
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.level_name = level_name
        self.time_name = time_name
        self.chunk_overrides = dict(chunk_overrides or {})
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def required_variable_keys(self) -> tuple[str, ...]:
        """Logical variable keys required for LPDM stepping."""

        return ("u", "v", "w", "blh", "sp", "t", "z", "z_sfc")

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

            # Subset vertical levels using lazy Dask arrays (only reads Z/Z_SFC into memory to evaluate the bound)
            ds_start_sub, ds_end_sub, level_agl_m = self._subset_vertical_levels_by_agl(
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

        return HourlyMetTensors(
            hour_start=hour_start,
            hour_end=hour_end,
            metadata=metadata,
        )

    def _open_dataset(self) -> xr.Dataset:
        """Open the remote Zarr store lazily.

        Keep this method isolated so later versions can add dataset caching,
        retries, consolidated metadata toggles, or auth customization.
        """

        return xr.open_zarr(
            self.zarr_store,
            consolidated=True,
            chunks=self.chunk_overrides or None,
        )

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
    ) -> tuple[xr.Dataset, xr.Dataset, np.ndarray]:
        """Subset pressure levels using geopotential-derived geometric AGL bounds."""

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

        level_agl_mean = 0.5 * (
            np.mean(level_agl_start, axis=(1, 2)) + np.mean(level_agl_end, axis=(1, 2))
        )
        level_values = np.asarray(ds_start[self.level_name].values)
        level_mask = (level_agl_mean >= spatial.z_min) & (level_agl_mean <= spatial.z_max)

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
        level_agl_m = 0.5 * (
            np.mean(level_agl_start_sub, axis=(1, 2)) + np.mean(level_agl_end_sub, axis=(1, 2))
        )

        return ds_start_sub, ds_end_sub, level_agl_m

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

        agl_m = (z_level - z_sfc_3d) / GRAVITY_M_S2
        return np.maximum(agl_m, 0.0)

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

    def _canonicalize_hour_bounds(self, bounds: TimeBounds) -> tuple[datetime, datetime]:
        """Return exact hour start/end timestamps used for met interpolation."""

        t0 = bounds.start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        t1 = t0 + timedelta(hours=1)
        return t0, t1

    def _dataset_to_channel_tensor(self, ds_time_slice: xr.Dataset) -> torch.Tensor:
        """Convert one xarray time-slice into [channels, z, y, x] torch tensor."""

        channels: list[np.ndarray] = []
        logical_keys = ("u", "v", "w", "blh", "sp")
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
