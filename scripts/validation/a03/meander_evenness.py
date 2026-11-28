"""Tier 0 meander test, part 2 (task A03): how uneven does an even cloud become?

GLIDE's meander term (docs/turbulence.md §7) is an Ornstein–Uhlenbeck velocity
with a velocity scale σ_m that varies in space, and no drift correction for that
variation. Without one, particles slowly collect where σ_m is small. This script
measures how much, in a realistic field.

Method:
- particles are spread evenly in area over the EUR box (stratified per 0.25° cell,
  count proportional to the cell's area), at fixed heights above ground;
- everything but the meander term is off: no advection, no small-scale turbulence,
  no convection, so the heights never change. The meander update is GLIDE's own:
  `HannaScheme._meander_sigma_fields`, `_interp_3d_field`, and the engine's
  `update_langevin_velocity` / `apply_horizontal_turbulence`, exactly as the step
  uses them, with the timescale bound from the store (half its interval);
- the run goes backward `--days` from `--end`, as GLIDE does, with 300 s steps and
  each hourly window's σ_m field;
- two replicas, A and B, have independent random draws and independent initial
  positions (the same count in every cell; sharing positions would correlate
  which particles start near a cell edge). For each cell the unevenness is r = final count / initial count − 1.
  What A and B share is systematic (the missing drift); what differs is particle
  noise:  systematic variance = cov(r_A, r_B),  noise variance = var(r_A − r_B)/2.
  Statistics are over interior cells (≥ 2° from the box edge), at 0.25° and 1°;
- the systematic pattern is compared with the diffusion-limit prediction
  δρ/ρ ≈ t ∇²K̄ with K = σ_m² τ (Itô diffusion without the drift: ∂ρ/∂t = ∇²(Kρ)).

The noise depends on the particle count; the summary also gives the count per
0.25° cell at which the systematic part would equal the noise.

Usage (GH200; one GPU job, a few GB of host memory):
    python scripts/validation/a03/meander_evenness.py \\
        --store data/era5-arco/EUR_20181222-20190115_ERA5-ARCO.zarr \\
        --end 2019-01-02T00:00:00Z --days 10 --per-cell 256 --device cuda \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from lpdm.config import MetDomainConfig
from lpdm.gpu_engine import GPUEngine
from lpdm.local_frame import DEGREES
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, SpatialBounds, TimeBounds
from lpdm.turbulence import HannaScheme

LOGGER = logging.getLogger("meander_evenness")

EUR_LON = (-30.0, 40.0)
EUR_LAT = (32.0, 72.0)
CELL_DEG = 0.25
INTERIOR_MARGIN_DEG = 2.0
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0


def _even_cloud(per_cell: int, generator: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Positions evenly in area: per 0.25° cell, round(per_cell·cos φ / cos 32°)."""

    lon_edges = np.arange(EUR_LON[0], EUR_LON[1] + 1e-9, CELL_DEG)
    lat_edges = np.arange(EUR_LAT[0], EUR_LAT[1] + 1e-9, CELL_DEG)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    weight = np.cos(np.deg2rad(lat_c)) / np.cos(np.deg2rad(EUR_LAT[0]))
    per_row = np.maximum(1, np.round(per_cell * weight)).astype(np.int64)
    n_lon = lon_edges.size - 1
    lons, lats = [], []
    for j, n_row in enumerate(per_row):
        k = int(n_row * n_lon)
        cell_i = np.repeat(np.arange(n_lon), n_row)
        u = torch.rand(k, generator=generator, dtype=torch.float64).numpy()
        w = torch.rand(k, generator=generator, dtype=torch.float64).numpy()
        lons.append(lon_edges[cell_i] + u * CELL_DEG)
        # Uniform in sin(lat) within the cell: uniform in area.
        s0, s1 = np.sin(np.deg2rad(lat_edges[j])), np.sin(np.deg2rad(lat_edges[j + 1]))
        lats.append(np.rad2deg(np.arcsin(s0 + w * (s1 - s0))))
    return np.concatenate(lons), np.concatenate(lats)


def _counts(lon: np.ndarray, lat: np.ndarray, cell: float) -> np.ndarray:
    lon_edges = np.arange(EUR_LON[0], EUR_LON[1] + 1e-9, cell)
    lat_edges = np.arange(EUR_LAT[0], EUR_LAT[1] + 1e-9, cell)
    h, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges))
    return h


def _interior(cell: float) -> tuple[slice, slice]:
    m = int(round(INTERIOR_MARGIN_DEG / cell))
    ny = int(round((EUR_LAT[1] - EUR_LAT[0]) / cell))
    nx = int(round((EUR_LON[1] - EUR_LON[0]) / cell))
    return slice(m, ny - m), slice(m, nx - m)


def _laplacian(field: np.ndarray, lat: np.ndarray, dlon: float, dlat: float) -> np.ndarray:
    """∇² on a regular lat-lon grid [Y, X] (metres), interior points; edges NaN."""

    dx = (dlon * M_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat)))[:, None]
    dy = dlat * M_PER_DEG_LAT
    out = np.full_like(field, np.nan)
    out[1:-1, 1:-1] = (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / dx[
        1:-1
    ] ** 2 + (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / dy**2
    return out


def _height_slice(fields: torch.Tensor, levels: torch.Tensor, h: float) -> torch.Tensor:
    """Linear interpolation of a [C, Z, Y, X] stack to height h -> [C, Y, X]."""

    k = int(torch.searchsorted(levels, torch.tensor(h, device=levels.device, dtype=levels.dtype)))
    k = min(max(k, 1), levels.numel() - 1)
    f = float((h - levels[k - 1]) / (levels[k] - levels[k - 1]))
    return (1 - f) * fields[:, k - 1] + f * fields[:, k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--end", required=True, help="ISO time the backward run starts from")
    ap.add_argument("--days", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=300.0)
    ap.add_argument("--per-cell", type=int, default=256)
    ap.add_argument("--heights", type=float, nargs="+", default=[50.0, 1000.0, 5000.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float32
    engine = GPUEngine(device=args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    md = MetDomainConfig(lon_bounds=EUR_LON, lat_bounds=EUR_LAT, alt_max_m=15000.0)
    reader = ArcoEra5ZarrReader(
        zarr_store=args.store,
        channel_names=("u", "v", "w", "blh", "sp"),
        device="cpu",
        agl_levels_m=md.resolve_agl_levels(),
    )
    scheme = HannaScheme(meander_enabled=True)
    scheme.bind_met_interval(reader.met_interval_seconds)
    tau = scheme.meander_timescale_seconds
    interval = timedelta(seconds=reader.met_interval_seconds)
    LOGGER.info("met interval %s, meander timescale %.0f s", interval, tau)

    init = [_even_cloud(args.per_cell, gen) for _ in range(2)]  # replicas A, B
    init_lon, init_lat = init[0]
    n_h = init_lon.size
    assert init[1][0].size == n_h
    heights = args.heights
    n_groups = len(heights)
    # Replicas A and B, each with every height: [A(h0), A(h1), ..., B(h0), B(h1), ...].
    n_tot = 2 * n_groups * n_h
    LOGGER.info("%d positions x %d heights x 2 replicas = %d particles", n_h, n_groups, n_tot)
    xyz = torch.empty(n_tot, 4, device=device, dtype=dtype)
    for r in range(2):
        lon_t = torch.as_tensor(init[r][0], device=device, dtype=dtype)
        lat_t = torch.as_tensor(init[r][1], device=device, dtype=dtype)
        for g, h in enumerate(heights):
            sl = slice((r * n_groups + g) * n_h, (r * n_groups + g + 1) * n_h)
            xyz[sl, 0] = lon_t
            xyz[sl, 1] = lat_t
            xyz[sl, 2] = h
            xyz[sl, 3] = 1.0
    del lon_t, lat_t
    um = torch.zeros(n_tot, device=device, dtype=dtype)
    vm = torch.zeros_like(um)

    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    n_windows = int(round(args.days * 86400 / interval.total_seconds()))
    steps_per_window = int(round(interval.total_seconds() / args.dt))
    spatial = SpatialBounds(
        lon_min=EUR_LON[0],
        lon_max=EUR_LON[1],
        lat_min=EUR_LAT[0],
        lat_max=EUR_LAT[1],
        z_min=0.0,
        z_max=md.alt_max_m,
    )
    k_sum = None  # Σ over windows of K = σ_m² τ at each height [G, Y, X]
    sigma_mean_sum = None  # Σ over windows of mean(σ_u, σ_v) at each height [G, Y, X]
    t0 = time.time()
    met_lat = None
    for w in range(n_windows):
        w_end = end - w * interval
        met = reader.fetch_hourly_window(
            BoundingBoxRequest(spatial=spatial, time=TimeBounds(start=w_end - interval, end=w_end))
        )
        fields, bounds = scheme._meander_sigma_fields(met, device, dtype, frame=DEGREES)
        levels = torch.as_tensor(np.asarray(met.metadata.level), device=device, dtype=dtype)
        per_h = torch.stack([_height_slice(fields, levels, h) for h in heights])  # [G, 2, Y, X]
        kk = 0.5 * per_h.pow(2).sum(1) * tau
        sm_grid = 0.5 * per_h.sum(1)
        k_sum = kk if k_sum is None else k_sum + kk
        sigma_mean_sum = sm_grid if sigma_mean_sum is None else sigma_mean_sum + sm_grid
        met_lat = np.asarray(met.metadata.lat, dtype=np.float64)
        for _ in range(steps_per_window):
            sm = scheme._interp_3d_field(fields, bounds, xyz[:, 0], xyz[:, 1], xyz[:, 2], engine)
            um = engine.update_langevin_velocity(
                um, t_lagrangian=tau, sigma_w2=sm[0].pow(2), dt_seconds=args.dt
            )
            vm = engine.update_langevin_velocity(
                vm, t_lagrangian=tau, sigma_w2=sm[1].pow(2), dt_seconds=args.dt
            )
            xyz = engine.apply_horizontal_turbulence(xyz, um, vm, dt_seconds=args.dt, backward=True)
            del sm
        del met, fields
        if w % 12 == 0:
            LOGGER.info(
                "window %d/%d (ends %s), %.0f s elapsed; grid-mean σ_m per height %s",
                w + 1,
                n_windows,
                w_end.isoformat(),
                time.time() - t0,
                np.round(sm_grid.mean(dim=(1, 2)).cpu().numpy(), 3).tolist(),
            )

    final = xyz[:, :2].double().cpu().numpy()
    k_mean = (k_sum / n_windows).cpu().numpy()
    sigma_mean = (sigma_mean_sum / n_windows).cpu().numpy()
    if met_lat[0] > met_lat[-1]:  # ERA5 stores run north to south
        met_lat = met_lat[::-1].copy()
        k_mean = k_mean[:, ::-1, :].copy()
        sigma_mean = sigma_mean[:, ::-1, :].copy()
    total_s = args.days * 86400.0
    results: dict[str, object] = {
        "store": args.store,
        "end": args.end,
        "days": args.days,
        "dt_s": args.dt,
        "meander_timescale_s": tau,
        "per_cell_at_32N": args.per_cell,
        "heights_m": heights,
        "positions_per_height": int(n_h),
        "particles_total": int(n_tot),
        "seed": args.seed,
        "by_height": {},
    }
    arrays: dict[str, np.ndarray] = {"met_lat": met_lat, "k_mean": k_mean, "sigma_mean": sigma_mean}
    for g, h in enumerate(heights):
        sa = slice(g * n_h, (g + 1) * n_h)
        sb = slice((n_groups + g) * n_h, (n_groups + g + 1) * n_h)
        res_h: dict[str, object] = {
            "sigma_m_mean_m_s": float(np.mean(sigma_mean[g])),
            "sigma_m_p99_m_s": float(np.percentile(sigma_mean[g], 99)),
        }
        for cell in (CELL_DEG, 1.0):
            c0 = _counts(init_lon, init_lat, cell)
            assert np.array_equal(c0, _counts(init[1][0], init[1][1], cell))
            ra_full = _counts(final[sa, 0], final[sa, 1], cell) / c0 - 1.0
            rb_full = _counts(final[sb, 0], final[sb, 1], cell) / c0 - 1.0
            iy, ix = _interior(cell)
            ra, rb = ra_full[iy, ix].ravel(), rb_full[iy, ix].ravel()
            cov = float(np.mean((ra - ra.mean()) * (rb - rb.mean())))
            noise_var = float(np.var(ra - rb) / 2.0)
            sys_std = math.sqrt(max(cov, 0.0))
            noise_std = math.sqrt(noise_var)
            n0 = float(c0[iy, ix].mean())
            entry: dict[str, object] = {
                "cells": int(ra.size),
                "mean_initial_count": n0,
                "std_total": float(np.std(ra)),
                "systematic_std": sys_std,
                "noise_std": noise_std,
                "systematic_over_noise": sys_std / noise_std,
                "replica_correlation": float(np.corrcoef(ra, rb)[0, 1]),
                "mean_r": float(ra.mean()),
                "p01_p99_mean_of_replicas": [
                    float(np.percentile(0.5 * (ra + rb), 1)),
                    float(np.percentile(0.5 * (ra + rb), 99)),
                ],
                # Particles per cell at which the systematic part would equal the noise
                # (noise variance ∝ 1/count).
                "count_where_systematic_equals_noise": (
                    n0 * (noise_std / sys_std) ** 2 if sys_std > 0 else None
                ),
            }
            if cell == CELL_DEG:
                lap = _laplacian(k_mean[g], met_lat, CELL_DEG, CELL_DEG) * total_s
                # Met grid points are the cell corners: average the four.
                pred = 0.25 * (lap[:-1, :-1] + lap[1:, :-1] + lap[:-1, 1:] + lap[1:, 1:])
                p = pred[iy, ix].ravel()
                ok = np.isfinite(p)
                mean_ab = 0.5 * (ra + rb)
                entry["corr_with_prediction"] = float(np.corrcoef(mean_ab[ok], p[ok])[0, 1])
                entry["std_prediction"] = float(np.std(p[ok]))
                slope = float(np.polyfit(p[ok], mean_ab[ok], 1)[0])
                entry["regression_slope_on_prediction"] = slope
                arrays[f"pred_{h:g}m"] = pred
                arrays[f"r_a_{h:g}m"] = ra_full
                arrays[f"r_b_{h:g}m"] = rb_full
            res_h[f"{cell:g}deg"] = entry
        results["by_height"][f"{h:g}m"] = res_h
        LOGGER.info("height %g m: %s", h, json.dumps(res_h))
    results["wall_seconds"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(out / "fields.npz", **arrays)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
