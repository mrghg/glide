"""Decision 6 study: GLIDE's trigger against FLEXPART's cloud-base trigger, per column.

    PYTHONPATH=<worktree>/src .venv/bin/python scripts/validation/conv/trigger/trigger_columns.py \
        configs/ab_multisite_perf.yaml <out.npz> --times 2023-12-28T00 ... --nlev 23 40 80

Same met, same closure (buoyancy) and application (exponential) as the defaults;
only the trigger differs. For each grid and met window, every column is diagnosed
with three schemes:

- ``glide``: the default trigger (ΔT_v ≥ 0.9 K at LCL+1, CAPE ≥ 50 J/kg, depth ≥ 500 m);
- ``fpcb0``: ``trigger=flexpart_cloud_base`` (no depth check, as FLEXPART);
- ``fpcb``: the same with GLIDE's 500 m depth check added (``fpcb0`` and
  cloud depth >= 500 m; the option as first written, before the depth check
  was dropped from it).

M_b and the matrix of a column do not depend on the trigger (only whether it is
used), so the FORWARD transition over the interval is built once per column that
fires under any of them, and each trigger's totals are sums over its own mask.
Saved per column (all columns): masks, cos φ, M_b (if it fired), DTMIN, cloud
depth, CAPE, ΔT_v at ICB and at LCL+1, the sub-cloud air mass ``src`` (kg m-2),
and for columns firing under any trigger ``E`` (sub-cloud air moved into the
cloud layers over Δt, kg m-2), ``f3``/``f6`` (fraction of the sub-cloud air above
3 / 6 km AGL after Δt). The glide mask is checked against the earlier CONV run
(``--check``) for bit-identity on real meteorology.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

import lpdm
from lpdm.config import RunConfig
from lpdm.convection import emanuel as em
from lpdm.main import _met_domain_bounds, _met_interval
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, TimeBounds
from lpdm.turbulence import get_scheme
from lpdm.vertical_grid import default_agl_levels, stretched_agl_levels

SCHEMES = {
    "glide": dict(),
    "fpcb0": dict(trigger="flexpart_cloud_base"),
}


def levels_for(n: int, cfg: RunConfig) -> np.ndarray:
    if n == 23:
        return default_agl_levels(cfg.met_domain.alt_max_m)
    return stretched_agl_levels(
        n, cfg.met_domain.alt_max_m, first_layer_m=cfg.met_domain.first_layer_m
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--times", nargs="+", required=True)
    ap.add_argument("--nlev", nargs="+", type=int, default=[40])
    ap.add_argument("--zarr-store", default=None)
    ap.add_argument("--met-domain", nargs=4, type=float, default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--check", default=None, help="earlier grid npz with L<n>/<hour>/fires")
    args = ap.parse_args()

    print(f"lpdm: {lpdm.__file__}", flush=True)
    torch.set_num_threads(args.threads)
    cfg = RunConfig.from_yaml(args.config)
    store = args.zarr_store or cfg.io.zarr_store
    schemes = {k: em.EmanuelReducedConvection(**v) for k, v in SCHEMES.items()}
    turb = get_scheme(cfg.turbulence.scheme)
    channels = tuple(
        dict.fromkeys(("u", "v", "w", "blh", "sp", *turb.required_met_keys(), "t", "q"))
    )
    bounds = _met_domain_bounds(cfg)
    if args.met_domain is not None:
        lon0, lon1, lat0, lat1 = args.met_domain
        bounds = replace(bounds, lon_min=lon0, lon_max=lon1, lat_min=lat0, lat_max=lat1)
    check = np.load(args.check) if args.check else None
    out: dict[str, np.ndarray] = {}
    cpu = torch.device("cpu")
    for nlev in args.nlev:
        reader = ArcoEra5ZarrReader(
            zarr_store=store,
            channel_names=channels,
            device="cpu",
            agl_levels_m=levels_for(nlev, cfg),
        )
        for stamp in args.times:
            tic = time.perf_counter()
            t0 = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
            met = reader.fetch_hourly_window(
                BoundingBoxRequest(
                    spatial=bounds, time=TimeBounds(start=t0, end=t0 + _met_interval(reader))
                )
            )
            key = f"L{nlev}/{t0.strftime('%Y%m%dT%H')}"
            cols = {
                k: s.column_state(met, t_alpha=0.0, device=cpu, dtype=torch.float32)
                for k, s in schemes.items()
            }
            g = cols["glide"]
            n_col, n_lev = g.t.shape
            lat = np.asarray(met.metadata.lat, dtype=np.float64)
            lon = np.asarray(met.metadata.lon, dtype=np.float64)
            out[f"{key}/cosphi"] = np.cos(np.deg2rad(np.repeat(lat, lon.size)))
            cols["fpcb"] = cols["fpcb0"]._replace(
                fires=cols["fpcb0"].fires & (cols["fpcb0"].cloud_depth_m.double() >= 500.0)
            )
            for k, c in cols.items():
                out[f"{key}/fires_{k}"] = c.fires.numpy()
                assert torch.equal(c.i_lcl, g.i_lcl) and torch.equal(c.i_lnb, g.i_lnb)
                assert torch.equal(c.dtmin_k, g.dtmin_k)
            if check is not None and f"{key}/fires" in check.files:
                same = np.array_equal(check[f"{key}/fires"], g.fires.numpy())
                print(f"{key}: glide fires identical to the earlier run: {same}", flush=True)
                out[f"{key}/glide_identical"] = np.asarray(same)
            mb_all = schemes["glide"].buoyancy_mass_flux_per_k * g.dtmin_k.double().clamp(min=0.0)
            out[f"{key}/mb_all"] = mb_all.numpy()
            out[f"{key}/dtmin"] = g.dtmin_k.numpy()
            out[f"{key}/depth"] = g.cloud_depth_m.numpy()
            out[f"{key}/cape"] = g.cape.numpy()
            out[f"{key}/dtv_lcl1"] = g.dtv_lcl_plus_1.numpy()
            out[f"{key}/dtv_icb"] = cols["fpcb"].dtv_lcl_plus_1.numpy()
            out[f"{key}/i_lcl"] = g.i_lcl.numpy().astype(np.int16)
            out[f"{key}/i_lnb"] = g.i_lnb.numpy().astype(np.int16)
            h = g.height_m.double()
            out[f"{key}/h_lcl"] = h.gather(1, g.i_lcl.clamp(min=0).unsqueeze(1)).squeeze(1).numpy()
            m_all = em._layer_masses_per_area(g.pressure_pa.double()).clamp(min=1e-6)
            k = torch.arange(n_lev)
            src_mask = (k.unsqueeze(0) < g.i_lcl.clamp(min=0).unsqueeze(1)).double()
            out[f"{key}/src"] = (src_mask * m_all).sum(1).numpy()
            any_fire = cols["glide"].fires | cols["fpcb"].fires | cols["fpcb0"].fires
            idx = any_fire.nonzero().squeeze(1)
            out[f"{key}/idx"] = idx.numpy()
            # Matrices with M_b of the closure (same for every trigger).
            cu = cols["fpcb0"]._replace(m_b=mb_all)
            acc = {n: [] for n in ("E", "f3", "f6", "f3_0", "f6_0")}
            for ch in torch.split(idx, 2048):
                p = schemes["glide"].transition_matrices(cu, ch, backward=False).double()
                m = m_all[ch]
                src = src_mask[ch]
                tracer = torch.einsum("ci,cij->cj", src * m, p)
                tot = (src * m).sum(1).clamp(min=1e-12)
                acc["E"].append(((tracer * (1.0 - src)).sum(1)).numpy())
                acc["f3"].append(((tracer * (h[ch] >= 3000.0)).sum(1) / tot).numpy())
                acc["f6"].append(((tracer * (h[ch] >= 6000.0)).sum(1) / tot).numpy())
                # Sub-cloud air already above 3 / 6 km (LCL higher than that).
                acc["f3_0"].append(((src * m * (h[ch] >= 3000.0)).sum(1) / tot).numpy())
                acc["f6_0"].append(((src * m * (h[ch] >= 6000.0)).sum(1) / tot).numpy())
            for n, v in acc.items():
                out[f"{key}/{n}"] = np.concatenate(v) if v else np.zeros(0)
            out[f"{key}/interval_s"] = np.asarray(g.interval_s)
            nf = {k: int(c.fires.sum()) for k, c in cols.items()}
            print(
                f"{key}: {n_col} columns x {n_lev} levels, fires {nf}, "
                f"{time.perf_counter() - tic:.1f} s",
                flush=True,
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)


if __name__ == "__main__":
    main()
