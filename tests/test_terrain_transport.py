"""T4 — terrain-following transport over a hill, end-to-end (Finding 7).

The CPU-side acceptance test for the terrain-following vertical coordinate: build a
synthetic *pressure-level* met store with a Gaussian hill and the vertical velocity
air genuinely has riding over it, run it through the REAL `ArcoEra5ZarrReader`
resample, and verify that after the slope correction a near-surface particle holds
its height above ground while crossing the hill — the mechanism that fills the
Finding-7 surface-footprint holes. The `terrain_following=False` path (no slope
correction) must fail the same check, so the test has teeth.

See dev/TEST_REVIEW_2026-07-16.md (T4) and dev/CHECKPOINT.md (Finding 7).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import xarray as xr

from lpdm.gpu_engine import GPUEngine
from lpdm.main import _advect_active_particles
from lpdm.met_reader import (
    ArcoEra5ZarrReader,
    BoundingBoxRequest,
    SpatialBounds,
    TimeBounds,
)
from lpdm.vertical_grid import terrain_gradient

G = 9.80665


class _InMemoryReader(ArcoEra5ZarrReader):
    def __init__(self, dataset: xr.Dataset, *, terrain_following: bool) -> None:
        super().__init__(
            zarr_store="in-memory", device="cpu", dtype=torch.float64,
            channel_names=("u", "v", "w", "blh", "sp"),
            terrain_following=terrain_following,
        )
        self._dataset = dataset

    def _open_dataset(self) -> xr.Dataset:
        return self._ensure_monotonic_longitude(self._dataset)


def _hill_store(*, u_ms: float, hill_height_m: float, halfwidth_deg: float) -> xr.Dataset:
    """Uniform eastward wind U over a Gaussian hill h(lon); the stored vertical
    velocity is the terrain-following flow w = U·∂h/∂x (so a parcel rides up the
    windward slope and down the lee). Pressure levels sit at FIXED AGL offsets
    (geopotential = (terrain + offset)·g), so the implied AGL is terrain-invariant
    and only the slope correction can keep a particle at constant AGL."""
    lon = np.round(np.arange(-3.5, 3.51, 0.1), 3)
    lat = np.array([44.8, 45.0, 45.2])  # ~uniform → negligible lat terrain gradient
    agl_offsets = np.array([20.0, 100.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0, 12000.0])
    times = np.array([
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-01T01:00:00"),
    ])
    nt, nz, ny, nx = times.size, agl_offsets.size, lat.size, lon.size

    h_1d = hill_height_m * np.exp(-((lon / halfwidth_deg) ** 2))  # [X]
    terrain = np.broadcast_to(h_1d[None, :], (ny, nx)).copy()  # [Y, X]
    z_sfc = terrain * G  # geopotential at surface

    # Vertical velocity = U · ∂h/∂x, computed with the SAME terrain gradient the
    # reader uses, so the slope correction cancels it exactly (to the taper).
    dhdx, _ = terrain_gradient(terrain, lat, lon)  # [Y, X], m per m
    w_2d = u_ms * dhdx  # [Y, X], m/s

    def lev_field(vals_2d):  # broadcast a [Y,X] field over levels and time
        return np.broadcast_to(vals_2d[None, None], (nt, nz, ny, nx)).astype(np.float64)

    z = (terrain[None, :, :] + agl_offsets[:, None, None]) * G  # [Z, Y, X]
    ds = xr.Dataset(
        data_vars=dict(
            u_component_of_wind=(("time", "level", "latitude", "longitude"), np.full((nt, nz, ny, nx), u_ms)),
            v_component_of_wind=(("time", "level", "latitude", "longitude"), np.zeros((nt, nz, ny, nx))),
            vertical_velocity=(("time", "level", "latitude", "longitude"), lev_field(w_2d)),
            temperature=(("time", "level", "latitude", "longitude"), np.full((nt, nz, ny, nx), 280.0)),
            boundary_layer_height=(("time", "latitude", "longitude"), np.full((nt, ny, nx), 1000.0)),
            surface_pressure=(("time", "latitude", "longitude"), np.full((nt, ny, nx), 101325.0)),
            geopotential=(("time", "level", "latitude", "longitude"), np.broadcast_to(z[None], (nt, nz, ny, nx)).copy()),
            geopotential_at_surface=(("time", "latitude", "longitude"), np.broadcast_to(z_sfc[None], (nt, ny, nx)).copy()),
        ),
        coords=dict(time=times, level=agl_offsets, latitude=lat, longitude=lon),
    )
    units = {
        "u_component_of_wind": "m s**-1", "v_component_of_wind": "m s**-1",
        "vertical_velocity": "m s**-1", "temperature": "K",
        "boundary_layer_height": "m", "surface_pressure": "Pa",
        "geopotential": "m**2 s**-2", "geopotential_at_surface": "m**2 s**-2",
    }
    for k, u in units.items():
        ds[k].attrs["units"] = u
    ds["level"].attrs["units"] = "hPa"  # nominal; the reader uses geopotential AGL
    return ds


def _crossing_max_agl_excursion(*, terrain_following: bool) -> tuple[float, float]:
    """Backward-advect a particle released at ~50 m AGL downwind of the hill,
    across the hill, through the real reader path. Returns (max |z-50| over the
    hill region, terrain height at the peak) — the AGL excursion a correct
    coordinate must keep small."""
    u_ms = 15.0
    hill_height = 800.0
    ds = _hill_store(u_ms=u_ms, hill_height_m=hill_height, halfwidth_deg=0.5)
    reader = _InMemoryReader(ds, terrain_following=terrain_following)

    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    window = reader.fetch_hourly_window(
        BoundingBoxRequest(
            spatial=SpatialBounds(
                lon_min=-3.5, lon_max=3.5, lat_min=44.7, lat_max=45.3, z_min=0.0, z_max=15000.0,
            ),
            time=TimeBounds(start=t0, end=t0 + timedelta(hours=1)),
        )
    )

    engine = GPUEngine(device="cpu", dtype=torch.float64)
    z0 = 50.0
    # [lon, lat, z(AGL), weight]; start downwind (+lon) so backward advection
    # carries it upwind across the hill at lon=0.
    p = torch.tensor([[2.5, 45.0, z0, 1.0]], dtype=torch.float64)
    t_mid = t0 + timedelta(minutes=30)  # steady fields → t_cursor value is immaterial

    max_exc = 0.0
    for _ in range(400):  # ~6.7 h at dt=60 s → traverses the hill
        p, _, _ = _advect_active_particles(
            engine, torch.device("cpu"), p, window, t_mid, 60.0, torch.float64,
        )
        lon = float(p[0, 0])
        if abs(lon) < 1.5:  # within the hill's influence
            max_exc = max(max_exc, abs(float(p[0, 2]) - z0))
        if lon < -2.0:
            break
    peak_terrain = hill_height
    return max_exc, peak_terrain


def test_terrain_following_preserves_agl_crossing_hill() -> None:
    """With the slope correction, a near-surface particle holds its AGL crossing
    an 800 m hill (the mechanism that keeps the surface footprint continuous)."""
    max_exc, hill = _crossing_max_agl_excursion(terrain_following=True)
    assert max_exc < 25.0, (
        f"AGL excursion {max_exc:.1f} m crossing a {hill:.0f} m hill — the slope "
        f"correction should hold the particle at ~constant height above ground"
    )


def test_no_slope_correction_lets_particle_ride_the_terrain() -> None:
    """Teeth: without the terrain-following resample the raw vertical velocity is
    uncorrected, so the same particle is driven up over the hill by a large
    fraction of its height — proving the passing test above measures the fix, not
    a trivially flat field."""
    max_exc, hill = _crossing_max_agl_excursion(terrain_following=False)
    assert max_exc > 200.0, (
        f"expected a large AGL excursion without slope correction; got {max_exc:.1f} m "
        f"over a {hill:.0f} m hill (if this is small, the test has no teeth)"
    )
