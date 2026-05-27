from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from lpdm.output_writer import OutputWriter


def test_output_writer_local_files(tmp_path: Path) -> None:
    writer = OutputWriter()

    particles = torch.tensor(
        [
            [10.0, 20.0, 500.0, 0.5],
            [11.0, 21.0, 550.0, 0.5],
        ],
        dtype=torch.float32,
    )

    endpoint_path = tmp_path / "endpoint_particles.parquet"
    writer.write_particles_parquet(str(endpoint_path), particles)
    assert endpoint_path.exists()

    df_particles = pd.read_parquet(endpoint_path)
    assert list(df_particles.columns) == ["lon", "lat", "alt", "weight"]
    assert len(df_particles) == 2

    trajectory_rows = [
        {"step": 1, "mean_lon": 10.1, "mean_lat": 20.1, "mean_alt_agl_m": 510.0},
        {"step": 2, "mean_lon": 10.2, "mean_lat": 20.2, "mean_alt_agl_m": 520.0},
    ]
    trajectory_path = tmp_path / "trajectory.parquet"
    writer.write_trajectory_parquet(str(trajectory_path), step_seconds=300, rows=trajectory_rows)
    assert trajectory_path.exists()

    df_traj = pd.read_parquet(trajectory_path)
    assert "elapsed_seconds" in df_traj.columns
    assert list(df_traj["elapsed_seconds"]) == [300, 600]

    metadata_path = tmp_path / "run_metadata.json"
    payload = {"hello": "world", "n": 2}
    writer.write_metadata_json(str(metadata_path), payload)
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text()) == payload


def test_output_writer_footprint_zarr(tmp_path: Path) -> None:
    """Writer accepts 5D (R, T, Z, Y, X) tensor and round-trips coords + attrs."""

    writer = OutputWriter()

    footprint = torch.ones((1, 2, 3, 4, 5), dtype=torch.float32)
    zarr_path = tmp_path / "footprint.zarr"
    release_time = np.array([np.datetime64("2024-01-01T00:00:00", "ns")])
    writer.write_footprint_zarr(
        str(zarr_path),
        footprint,
        coords={
            "release_time": release_time,
            "release_duration_seconds": ("release_time", np.array([3600.0], dtype=np.float64)),
            "time_ago": np.array([0, 1], dtype=np.int64),
            "time_ago_start_hours": ("time_ago", np.array([0.0, 1.0], dtype=np.float64)),
            "time_ago_end_hours": ("time_ago", np.array([1.0, 2.0], dtype=np.float64)),
            "z_bin": np.array([500.0, 1500.0, 2500.0], dtype=np.float64),
            "z_bottom_m": ("z_bin", np.array([0.0, 1000.0, 2000.0], dtype=np.float64)),
            "z_top_m": ("z_bin", np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)),
            "latitude": np.array([35.25, 35.75, 36.25, 36.75], dtype=np.float64),
            "longitude": np.array([-122.5, -122.0, -121.5, -121.0, -120.5], dtype=np.float64),
            "latitude_edge": ("latitude_edge", np.array([35.0, 35.5, 36.0, 36.5, 37.0], dtype=np.float64)),
            "longitude_edge": ("longitude_edge", np.array([-122.75, -122.25, -121.75, -121.25, -120.75, -120.25], dtype=np.float64)),
        },
        attrs={
            "release_lon": -122.3,
            "release_lat": 37.9,
            "release_alt_agl_m": 500.0,
        },
    )

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    assert "footprint" in ds
    assert ds["footprint"].shape == (1, 2, 3, 4, 5)
    assert ds["footprint"].dims == ("release_time", "time_ago", "z_bin", "latitude", "longitude")
    assert ds["release_time"].values[0] == release_time[0]
    assert np.allclose(ds["release_duration_seconds"].values, np.array([3600.0]))
    assert np.allclose(ds["time_ago_start_hours"].values, np.array([0.0, 1.0]))
    assert np.allclose(ds["time_ago_end_hours"].values, np.array([1.0, 2.0]))
    assert np.allclose(ds["z_bottom_m"].values, np.array([0.0, 1000.0, 2000.0]))
    assert np.allclose(ds["z_top_m"].values, np.array([1000.0, 2000.0, 3000.0]))
    assert np.allclose(ds["latitude_edge"].values, np.array([35.0, 35.5, 36.0, 36.5, 37.0]))
    assert np.allclose(ds["longitude_edge"].values, np.array([-122.75, -122.25, -121.75, -121.25, -120.75, -120.25]))
    assert ds.attrs["release_lon"] == -122.3
    assert ds.attrs["release_lat"] == 37.9
    assert ds.attrs["release_alt_agl_m"] == 500.0


def test_output_writer_footprint_rejects_4d_tensor(tmp_path: Path) -> None:
    """Stage 4 makes the writer strictly 5D; passing a legacy 4D tensor must error."""

    import pytest

    writer = OutputWriter()
    footprint_4d = torch.ones((2, 3, 4, 5), dtype=torch.float32)
    with pytest.raises(ValueError, match="must have shape"):
        writer.write_footprint_zarr(str(tmp_path / "fp.zarr"), footprint_4d)


def test_output_writer_particles_with_release_idx_column(tmp_path: Path) -> None:
    """M5 stage 5: multi-release runs pass per-particle release_idx; writer adds the column."""

    writer = OutputWriter()
    particles = torch.tensor(
        [
            [10.0, 20.0, 500.0, 0.5],
            [11.0, 21.0, 550.0, 0.5],
            [12.0, 22.0, 600.0, 0.5],
        ],
        dtype=torch.float32,
    )
    release_idx = torch.tensor([0, 1, 1], dtype=torch.int64)

    path = tmp_path / "particles.parquet"
    writer.write_particles_parquet(str(path), particles, release_idx=release_idx)

    df = pd.read_parquet(path)
    assert list(df.columns) == ["lon", "lat", "alt", "weight", "release_idx"]
    assert df["release_idx"].tolist() == [0, 1, 1]


def test_output_writer_particles_release_idx_shape_mismatch_rejected(tmp_path: Path) -> None:
    import pytest

    writer = OutputWriter()
    particles = torch.tensor([[10.0, 20.0, 500.0, 0.5]], dtype=torch.float32)
    bad_release_idx = torch.tensor([0, 1, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="release_idx shape"):
        writer.write_particles_parquet(
            str(tmp_path / "x.parquet"), particles, release_idx=bad_release_idx
        )


def test_output_writer_particles_back_compat_without_release_idx(tmp_path: Path) -> None:
    """Single-release callers pass no release_idx and get the legacy 4-column output."""

    writer = OutputWriter()
    particles = torch.tensor([[10.0, 20.0, 500.0, 0.5]], dtype=torch.float32)
    path = tmp_path / "single.parquet"
    writer.write_particles_parquet(str(path), particles)
    df = pd.read_parquet(path)
    assert list(df.columns) == ["lon", "lat", "alt", "weight"]


def _footprint_coords(n_releases: int) -> dict:
    """Minimal coord set for a (n_releases, 2, 3, 4, 5) footprint store."""

    import numpy as np

    return {
        "release_time": (
            np.datetime64("2024-01-01T00:00:00", "ns")
            + np.arange(n_releases) * np.timedelta64(1, "h")
        ),
        "time_ago": np.array([0, 1], dtype=np.int64),
        "z_bin": np.array([500.0, 1500.0, 2500.0], dtype=np.float64),
        "latitude": np.array([35.25, 35.75, 36.25, 36.75], dtype=np.float64),
        "longitude": np.array([-122.5, -122.0, -121.5, -121.0, -120.5], dtype=np.float64),
        "latitude_edge": ("latitude_edge", np.array([35.0, 35.5, 36.0, 36.5, 37.0], dtype=np.float64)),
    }


def test_streaming_footprint_store_create_then_region_writes(tmp_path: Path) -> None:
    """M5 streaming: create an empty store, then fill it one batch at a time.

    Verifies that create_footprint_store never materialises the full tensor (it
    uses a lazy dask template) and that region writes land in disjoint
    release_time slices without overwriting each other.
    """

    writer = OutputWriter()
    path = str(tmp_path / "fp.zarr")
    n_releases = 5
    shape = (n_releases, 2, 3, 4, 5)

    writer.create_footprint_store(path, shape=shape, coords=_footprint_coords(n_releases))

    # Three batches: releases [0:2], [2:4], [4:5], each filled with a distinct value.
    for start, stop, value in [(0, 2, 1.0), (2, 4, 2.0), (4, 5, 3.0)]:
        block = torch.full((stop - start, 2, 3, 4, 5), value, dtype=torch.float32)
        writer.write_footprint_region(path, block, release_start=start, release_stop=stop)

    ds = xr.open_zarr(path)
    fp = ds["footprint"]
    assert fp.shape == shape
    assert fp.dims == ("release_time", "time_ago", "z_bin", "latitude", "longitude")
    per_release = [float(fp.isel(release_time=i).mean()) for i in range(n_releases)]
    assert per_release == [1.0, 1.0, 2.0, 2.0, 3.0]
    # Coords survived the template write.
    assert np.allclose(ds["latitude_edge"].values, np.array([35.0, 35.5, 36.0, 36.5, 37.0]))


def test_streaming_footprint_region_shape_mismatch_rejected(tmp_path: Path) -> None:
    import pytest

    writer = OutputWriter()
    path = str(tmp_path / "fp.zarr")
    writer.create_footprint_store(path, shape=(3, 2, 3, 4, 5), coords=_footprint_coords(3))

    # Region [0:2] expects leading dim 2; pass 3.
    bad = torch.ones((3, 2, 3, 4, 5), dtype=torch.float32)
    with pytest.raises(ValueError, match="region size"):
        writer.write_footprint_region(path, bad, release_start=0, release_stop=2)


def test_streaming_footprint_store_rejects_non_5d_shape(tmp_path: Path) -> None:
    import pytest

    writer = OutputWriter()
    with pytest.raises(ValueError, match="must be"):
        writer.create_footprint_store(str(tmp_path / "fp.zarr"), shape=(3, 2, 3, 4))  # type: ignore[arg-type]
