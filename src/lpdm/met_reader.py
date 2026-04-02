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
    time_start: datetime
    time_end: datetime


@dataclass(frozen=True)
class HourlyMetTensors:
    """Hour bracketing tensors for fine-step temporal interpolation.

    Shape convention for each tensor:
        [channels, z, y, x]

    Channel order is determined by the reader implementation and should be
    documented centrally (for example: [u, v, w, blh]).
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
        "u": "u",
        "v": "v",
        "w": "w",
        "blh": "blh",
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

    def fetch_hourly_window(self, request: BoundingBoxRequest) -> HourlyMetTensors:
        """Read, slice, materialize, and convert one hourly met window.

        High-level flow:
        1. Open remote Zarr lazily.
        2. Select required variables and bounding box.
        3. Materialize the tiny subset into CPU memory with `.compute()`.
        4. Split into hour-start and hour-end snapshots.
        5. Convert both snapshots to consistent torch tensors.
        """

        ds = self._open_dataset()
        ds = self._select_variables(ds)
        ds = self._slice_spatial_temporal(ds, request)

        # Explicitly trigger dask graph execution here so the rest of the
        # pipeline works with in-memory numpy-backed data.
        ds_local = ds.compute()

        t0, t1 = self._canonicalize_hour_bounds(request.time)
        ds_start = ds_local.sel({self.time_name: t0})
        ds_end = ds_local.sel({self.time_name: t1})

        hour_start = self._dataset_to_channel_tensor(ds_start)
        hour_end = self._dataset_to_channel_tensor(ds_end)

        metadata = MetFieldMetadata(
            lon=np.asarray(ds_local[self.lon_name].values),
            lat=np.asarray(ds_local[self.lat_name].values),
            level=np.asarray(ds_local[self.level_name].values),
            time_start=t0,
            time_end=t1,
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

        dataset_vars: Sequence[str] = tuple(self.variable_map.values())
        missing = [name for name in dataset_vars if name not in ds.variables]
        if missing:
            raise KeyError(f"Required variables missing from dataset: {missing}")
        return ds[list(dataset_vars)]

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

        ds = ds.sel(
            {
                self.time_name: slice(time.start, time.end),
                self.lon_name: slice(spatial.lon_min, spatial.lon_max),
                self.level_name: slice(spatial.z_min, spatial.z_max),
            }
        )

        lat_values = ds[self.lat_name].values
        lat_descending = lat_values[0] > lat_values[-1]
        lat_slice = (
            slice(spatial.lat_max, spatial.lat_min)
            if lat_descending
            else slice(spatial.lat_min, spatial.lat_max)
        )
        ds = ds.sel({self.lat_name: lat_slice})

        return ds

    def _canonicalize_hour_bounds(self, bounds: TimeBounds) -> tuple[datetime, datetime]:
        """Return exact hour start/end timestamps used for met interpolation."""

        t0 = bounds.start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        t1 = t0 + timedelta(hours=1)
        return t0, t1

    def _dataset_to_channel_tensor(self, ds_time_slice: xr.Dataset) -> torch.Tensor:
        """Convert one xarray time-slice into [channels, z, y, x] torch tensor."""

        channels: list[np.ndarray] = []
        logical_keys = ("u", "v", "w", "blh")

        for key in logical_keys:
            var_name = self.variable_map[key]
            arr = np.asarray(ds_time_slice[var_name].values)

            # Ensure shape is [z, y, x]. For 2D fields (e.g. BLH), broadcast over z.
            if arr.ndim == 2:
                z_size = int(ds_time_slice[self.level_name].size)
                arr = np.broadcast_to(arr[None, :, :], (z_size, *arr.shape))
            elif arr.ndim != 3:
                raise ValueError(
                    f"Unsupported variable rank for {var_name}: expected 2D or 3D, got {arr.ndim}D"
                )

            channels.append(arr)

        stacked = np.stack(channels, axis=0)  # [channels, z, y, x]
        tensor = torch.as_tensor(stacked, dtype=self.dtype, device=self.device)
        return tensor
