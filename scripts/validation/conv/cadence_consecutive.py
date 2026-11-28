"""Task CONV: hourly against 3-hourly convection on real, changing meteorology.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree>/src .venv/bin/python \
        scripts/validation/conv/cadence_consecutive.py configs/ab_multisite_perf.yaml \
        --starts 2023-12-29T06 2023-12-31T03

For each start t0 it reads the hourly windows t0, t0+1 h, t0+2 h (default
40-level grid) and, per variant, composes the three hourly FORWARD transitions
of every column that fires in any of the hours (identity in an hour where it
does not fire): the hourly cadence. The 3-hourly cadence is ONE event with a
3-h interval and the meteorology of t0 (what a 3-hourly store would give). It
prints the sub-cloud air moved into the cloud over the 3 h, summed over those
columns with cos(lat) weights, for both cadences and their ratio. The sub-cloud
layer is the one of the hour t0 for both, so only columns firing at t0 count.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import torch

import lpdm
from lpdm.config import RunConfig
from lpdm.convection import emanuel as em
from lpdm.main import _met_domain_bounds, _met_interval
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, TimeBounds

VARIANTS = {
    "legacy": dict(closure="cape_velocity", application="cfl_scaled"),
    "legacy_sub4": dict(closure="cape_velocity", application="substeps", n_substeps=4),
    "c003_exp": dict(closure="cape_velocity", application="exponential", closure_c=0.003),
    "buoy_exp": dict(closure="buoyancy", application="exponential"),
    "buoy_sub4": dict(closure="buoyancy", application="substeps", n_substeps=4),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--starts", nargs="+", required=True)
    ap.add_argument("--zarr-store", default=None)
    ap.add_argument("--met-domain", nargs=4, type=float, default=None)
    ap.add_argument("--trigger-dz", type=float, default=None)
    args = ap.parse_args()
    print(f"lpdm: {lpdm.__file__}", flush=True)
    torch.set_num_threads(8)
    cfg = RunConfig.from_yaml(args.config)
    reader = ArcoEra5ZarrReader(
        zarr_store=args.zarr_store or cfg.io.zarr_store,
        channel_names=("u", "v", "w", "blh", "sp", "t", "q"),
        device="cpu",
        agl_levels_m=cfg.met_domain.resolve_agl_levels(),
    )
    bounds = _met_domain_bounds(cfg)
    if args.met_domain is not None:
        lon0, lon1, lat0, lat1 = args.met_domain
        bounds = replace(bounds, lon_min=lon0, lon_max=lon1, lat_min=lat0, lat_max=lat1)
    schemes = {
        v: em.EmanuelReducedConvection(
            **kw, trigger="glide", trigger_height_above_lcl_m=args.trigger_dz
        )
        for v, kw in VARIANTS.items()
    }
    totals = {v: [0.0, 0.0] for v in VARIANTS}
    for stamp in args.starts:
        t0 = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
        mets = [
            reader.fetch_hourly_window(
                BoundingBoxRequest(
                    spatial=bounds,
                    time=TimeBounds(start=t, end=t + _met_interval(reader)),
                )
            )
            for t in (t0, t0 + timedelta(hours=1), t0 + timedelta(hours=2))
        ]
        lat = np.asarray(mets[0].metadata.lat, dtype=np.float64)
        n_lon = np.asarray(mets[0].metadata.lon).size
        cos = torch.as_tensor(np.cos(np.deg2rad(np.repeat(lat, n_lon))))
        for v, scheme in schemes.items():
            cols = [
                scheme.column_state(m, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32)
                for m in mets
            ]
            idx = cols[0].fires.nonzero().squeeze(1)
            if idx.numel() == 0:
                continue
            m0 = em._layer_masses_per_area(cols[0].pressure_pa[idx].double()).clamp(min=1e-6)
            k = torch.arange(m0.shape[1])
            src = (k.unsqueeze(0) < cols[0].i_lcl[idx].unsqueeze(1)).double()
            n_lev = m0.shape[1]
            comp = torch.eye(n_lev, dtype=torch.float64).expand(idx.numel(), n_lev, n_lev).clone()
            for c in cols:
                p = torch.eye(n_lev, dtype=torch.float64).expand_as(comp).clone()
                f = c.fires[idx]
                if bool(f.any()):
                    p[f] = scheme.transition_matrices(c, idx[f], backward=False).double()
                comp = comp @ p
            p3 = scheme.transition_matrices(
                cols[0], idx, interval_s=3 * cols[0].interval_s, backward=False
            )
            e_h = (torch.einsum("ci,cij->cj", src * m0, comp) * (1 - src)).sum(1)
            e_3 = (torch.einsum("ci,cij->cj", src * m0, p3.double()) * (1 - src)).sum(1)
            w = cos[idx]
            totals[v][0] += float((e_h * w).sum())
            totals[v][1] += float((e_3 * w).sum())
            fires_all = sum(int(c.fires[idx].sum()) for c in cols)
            print(
                f"{stamp} {v:12s}: {idx.numel()} columns fire at t0 ({fires_all} column-hours over 3 h); "
                f"sum E hourly {float((e_h * w).sum()):.4g}, 3-hourly {float((e_3 * w).sum()):.4g}, "
                f"ratio {float((e_3 * w).sum() / (e_h * w).sum()):.3f}",
                flush=True,
            )
    print("\nall starts: 3-hourly / hourly cadence, sub-cloud air into the cloud over 3 h")
    for v, (h, t) in totals.items():
        print(f"   {v:12s} {t / max(h, 1e-30):.3f}")


if __name__ == "__main__":
    main()
