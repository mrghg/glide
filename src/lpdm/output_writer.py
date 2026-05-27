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

	def write_particles_parquet(
		self,
		path: str,
		particles: torch.Tensor,
		*,
		release_idx: torch.Tensor | None = None,
	) -> None:
		"""Write particle tensor shaped (N, 4) to Parquet.

		Expected column order is ``[lon, lat, alt, weight]``. When ``release_idx``
		is provided (M5 stage 5+ multi-release runs), it is added as a fifth
		``release_idx`` int64 column aligned with the leading dimension; consumers
		group endpoint particles by release via this column. Single-release runs
		omit the argument and the column.
		"""

		arr = particles.detach().to(device="cpu").numpy()
		if arr.ndim != 2 or arr.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")

		df = pd.DataFrame(arr, columns=["lon", "lat", "alt", "weight"])
		if release_idx is not None:
			ridx = release_idx.detach().to(device="cpu").numpy().astype(np.int64)
			if ridx.shape != (arr.shape[0],):
				raise ValueError(
					f"release_idx shape {ridx.shape} does not match particles ({arr.shape[0]},)"
				)
			df["release_idx"] = ridx

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

	FOOTPRINT_DIMS: tuple[str, str, str, str, str] = (
		"release_time",
		"time_ago",
		"z_bin",
		"latitude",
		"longitude",
	)

	@staticmethod
	def _split_footprint_coords(
		coords: Mapping[str, Any] | None,
		dims: tuple[str, ...],
	) -> tuple[dict[str, Any], dict[str, Any]]:
		"""Partition coords into those that live on the footprint dims vs auxiliary
		coords on their own dims (e.g. latitude_edge/longitude_edge)."""

		array_coords: dict[str, Any] = {}
		dataset_coords: dict[str, Any] = {}
		for coord_name, coord_value in (coords or {}).items():
			coord_dims = _coord_dims(coord_value)
			if all(dim in dims for dim in coord_dims):
				array_coords[coord_name] = coord_value
			else:
				dataset_coords[coord_name] = coord_value
		return array_coords, dataset_coords

	def write_footprint_zarr(
		self,
		path: str,
		footprint: torch.Tensor,
		*,
		dims: tuple[str, str, str, str, str] = FOOTPRINT_DIMS,
		coords: Mapping[str, Any] | None = None,
		attrs: Mapping[str, Any] | None = None,
	) -> None:
		"""Write a complete 5D footprint tensor (R, T, Z, Y, X) to a Zarr store.

		Single-shot write: the whole tensor must already be in memory. Multi-release
		runs that don't want to hold the full schedule's footprint in RAM use the
		streaming pair `create_footprint_store` + `write_footprint_region` instead.

		The leading ``release_time`` axis was added in M5 stage 4 so multi-release
		batches can share a single output file. Single-release runs have a size-1
		leading axis; downstream tools squeeze or select it.
		"""

		arr = footprint.detach().to(device="cpu").numpy()
		if arr.ndim != 5:
			raise ValueError("footprint must have shape (R, T, Z, Y, X)")

		array_coords, dataset_coords = self._split_footprint_coords(coords, dims)

		da = xr.DataArray(arr, dims=dims, coords=array_coords, name="footprint")
		ds = da.to_dataset()
		if dataset_coords:
			ds = ds.assign_coords(dataset_coords)
		if attrs:
			ds.attrs.update(dict(attrs))

		if not _is_remote_path(path):
			Path(path).mkdir(parents=True, exist_ok=True)

		ds.to_zarr(store=path, mode="w")

	def create_footprint_store(
		self,
		path: str,
		*,
		shape: tuple[int, int, int, int, int],
		dims: tuple[str, str, str, str, str] = FOOTPRINT_DIMS,
		coords: Mapping[str, Any] | None = None,
		attrs: Mapping[str, Any] | None = None,
	) -> None:
		"""Create an empty 5D footprint Zarr store sized for the whole schedule.

		Writes coordinate arrays, attrs, and array metadata, but **not** the
		footprint data: the data variable is a lazy dask array of zeros written
		with ``compute=False``, so the full ``(R, T, Z, Y, X)`` tensor is never
		materialised in memory. Fill it incrementally with
		:meth:`write_footprint_region`. Chunked one release per chunk so each
		region write touches a disjoint set of chunks.
		"""

		import dask.array as da_mod

		if len(shape) != 5:
			raise ValueError("footprint store shape must be (R, T, Z, Y, X)")

		array_coords, dataset_coords = self._split_footprint_coords(coords, dims)
		footprint = da_mod.zeros(shape, chunks=(1,) + tuple(shape[1:]), dtype="float32")
		ds = xr.DataArray(footprint, dims=dims, coords=array_coords, name="footprint").to_dataset()
		if dataset_coords:
			ds = ds.assign_coords(dataset_coords)
		if attrs:
			ds.attrs.update(dict(attrs))

		if not _is_remote_path(path):
			Path(path).mkdir(parents=True, exist_ok=True)

		ds.to_zarr(store=path, mode="w", compute=False)

	def write_footprint_region(
		self,
		path: str,
		footprint: torch.Tensor,
		*,
		release_start: int,
		release_stop: int,
		dims: tuple[str, str, str, str, str] = FOOTPRINT_DIMS,
	) -> None:
		"""Write one contiguous block of releases into an existing footprint store.

		``footprint`` is shaped ``(release_stop - release_start, T, Z, Y, X)`` and
		lands in ``release_time[release_start:release_stop]`` of the store created
		by :meth:`create_footprint_store`. Only the data variable is written (no
		coords), so the call is a pure region update.
		"""

		arr = footprint.detach().to(device="cpu").numpy()
		if arr.ndim != 5:
			raise ValueError("footprint must have shape (R, T, Z, Y, X)")
		expected = release_stop - release_start
		if arr.shape[0] != expected:
			raise ValueError(
				f"footprint leading dim {arr.shape[0]} != region size {expected} "
				f"(release_start={release_start}, release_stop={release_stop})"
			)

		ds = xr.Dataset({"footprint": (dims, arr)})
		ds.to_zarr(store=path, region={"release_time": slice(release_start, release_stop)})
