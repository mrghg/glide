from __future__ import annotations

import json
from pathlib import Path

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
    writer = OutputWriter()

    footprint = torch.ones((2, 3, 4, 5), dtype=torch.float32)
    zarr_path = tmp_path / "footprint.zarr"
    writer.write_footprint_zarr(str(zarr_path), footprint)

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    assert "footprint" in ds
    assert ds["footprint"].shape == (2, 3, 4, 5)
