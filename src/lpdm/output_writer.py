"""Output serialization utilities for LPDM runs.

Supports both local filesystem paths and cloud paths (for example gs://)
through fsspec.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import fsspec
import numpy as np
import pandas as pd
import torch
import xarray as xr


def _coord_dims(coord_value: Any) -> tuple[str, ...]:
	if isinstance(coord_value, tuple):
		coord_dims = coord_value[0]
		if isinstance(coord_dims, str):
			return (coord_dims,)
		return tuple(coord_dims)
	return ()


def _is_remote_path(path: str) -> bool:
	return "://" in path


def _ensure_parent_dir(path: str) -> None:
	"""Create local parent directories when writing local files."""

	if _is_remote_path(path):
		return
	Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_bytes(path: str, payload: bytes) -> None:
	"""Write bytes to local or remote path."""

	_ensure_parent_dir(path)
	with fsspec.open(path, "wb") as f:
		f.write(payload)


class OutputWriter:
	"""Persist LPDM outputs to local or cloud storage."""

	def write_particles_parquet(self, path: str, particles: torch.Tensor) -> None:
		"""Write particle tensor shaped (N, 4) to Parquet.

		Expected column order is [lon, lat, alt, weight].
		"""

		arr = particles.detach().to(device="cpu").numpy()
		if arr.ndim != 2 or arr.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")

		df = pd.DataFrame(arr, columns=["lon", "lat", "alt", "weight"])
		buf = BytesIO()
		df.to_parquet(buf, index=False)
		_write_bytes(path, buf.getvalue())

	def write_trajectory_parquet(
		self,
		path: str,
		step_seconds: int,
		rows: list[Mapping[str, Any]],
	) -> None:
		"""Write trajectory diagnostics rows to Parquet."""

		df = pd.DataFrame(rows)
		if "step" in df.columns and "elapsed_seconds" not in df.columns:
			df["elapsed_seconds"] = df["step"].astype(np.int64) * int(step_seconds)

		buf = BytesIO()
		df.to_parquet(buf, index=False)
		_write_bytes(path, buf.getvalue())

	def write_metadata_json(self, path: str, payload: Mapping[str, Any]) -> None:
		"""Write run metadata JSON."""

		encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
		_write_bytes(path, encoded)

	def write_footprint_zarr(
		self,
		path: str,
		footprint: torch.Tensor,
		*,
		dims: tuple[str, str, str, str] = ("time_ago", "z_bin", "latitude", "longitude"),
		coords: Mapping[str, Any] | None = None,
		attrs: Mapping[str, Any] | None = None,
	) -> None:
		"""Write footprint tensor shaped (T, Z, Y, X) to Zarr store."""

		arr = footprint.detach().to(device="cpu").numpy()
		if arr.ndim != 4:
			raise ValueError("footprint must have shape (T, Z, Y, X)")

		array_coords: dict[str, Any] = {}
		dataset_coords: dict[str, Any] = {}
		for coord_name, coord_value in (coords or {}).items():
			coord_dims = _coord_dims(coord_value)
			if all(dim in dims for dim in coord_dims):
				array_coords[coord_name] = coord_value
			else:
				dataset_coords[coord_name] = coord_value

		da = xr.DataArray(arr, dims=dims, coords=array_coords, name="footprint")
		ds = da.to_dataset()
		if dataset_coords:
			ds = ds.assign_coords(dataset_coords)
		if attrs:
			ds.attrs.update(dict(attrs))

		if not _is_remote_path(path):
			Path(path).mkdir(parents=True, exist_ok=True)

		ds.to_zarr(store=path, mode="w")
