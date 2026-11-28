"""Task CONV: convective transport per unit time against the vertical grid and the
met cadence, for the closure / application variants of the reduced Emanuel scheme.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree>/src .venv/bin/python \
        scripts/validation/conv/closure_grid_cadence.py configs/ab_multisite_perf.yaml \
        <out.npz> --times 2023-12-29T06 ... --nlev 23 40 80

For each grid (number of stretched AGL levels; 23 = the pre-A01 ladder) and each
met window it reads the window as `lpdm.main` does, diagnoses every column
(`column_state`, t_alpha 0) and, for the columns that fire, builds the FORWARD
transition matrix over the interval for each variant (`transition_matrices`).
Per firing column it saves:

- the trigger checks (also for non-firing columns: pass counts per check);
- ``mb``: the cloud-base mass flux of the variant's closure, kg m-2 s-1;
- ``courant``: Δt · max_i(outflux_i / m_i) of the UNSCALED matrix (M_b Δt);
- ``E``: sub-cloud air [0, LCL) moved to the cloud layers [LCL, top] over Δt,
  kg m-2 (forward transition, each sub-cloud layer weighted by its mass);
- ``f3``, ``f6``: fraction of a sub-cloud tracer above 3 km / 6 km AGL after Δt;
- ``E3x1``: E over 3 h from three 1-h transitions of this window (P^3);
  ``E3``: E from ONE event with a 3-h interval (the synthetic 3-hourly cadence).

Nothing is tuned here; the scheme is the code on PYTHONPATH.
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

VARIANTS = {
    # name: constructor kwargs (all else at the defaults)
    "legacy": dict(closure="cape_velocity", application="cfl_scaled"),
    "legacy_sub4": dict(closure="cape_velocity", application="substeps", n_substeps=4),
    "legacy_exp": dict(closure="cape_velocity", application="exponential"),
    "c003_exp": dict(closure="cape_velocity", application="exponential", closure_c=0.003),
    "buoy_exp": dict(closure="buoyancy", application="exponential"),
    "buoy_sub4": dict(closure="buoyancy", application="substeps", n_substeps=4),
    "buoy_cfl": dict(closure="buoyancy", application="cfl_scaled"),
}


def levels_for(n: int, cfg: RunConfig) -> np.ndarray:
    if n == 23:
        return default_agl_levels(cfg.met_domain.alt_max_m)
    return stretched_agl_levels(
        n, cfg.met_domain.alt_max_m, first_layer_m=cfg.met_domain.first_layer_m
    )


def transport_metrics(scheme, cols, idx, interval_s, height):
    """E (kg/m2), f3, f6 for the forward transition over `interval_s`."""
    p = scheme.transition_matrices(cols, idx, interval_s=interval_s, backward=False).double()
    m = em._layer_masses_per_area(cols.pressure_pa[idx].double()).clamp(min=1e-6)
    k = torch.arange(m.shape[1])
    src = (k.unsqueeze(0) < cols.i_lcl[idx].unsqueeze(1)).double()
    tracer = torch.einsum("ci,cij->cj", src * m, p)  # kg/m2 of sub-cloud air per layer
    e = (tracer * (1.0 - src)).sum(1)
    tot = (src * m).sum(1)
    f3 = (tracer * (height[idx] >= 3000.0)).sum(1) / tot
    f6 = (tracer * (height[idx] >= 6000.0)).sum(1) / tot
    return e, f3, f6, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--times", nargs="+", required=True)
    ap.add_argument("--nlev", nargs="+", type=int, default=[40])
    ap.add_argument("--zarr-store", default=None)
    ap.add_argument("--met-domain", nargs=4, type=float, default=None)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--trigger-dz",
        type=float,
        default=None,
        help="trigger_height_above_lcl_m for every variant (default: the level LCL+1)",
    )
    args = ap.parse_args()

    print(f"lpdm: {lpdm.__file__}", flush=True)
    torch.set_num_threads(args.threads)
    cfg = RunConfig.from_yaml(args.config)
    store = args.zarr_store or cfg.io.zarr_store
    schemes = {
        v: em.EmanuelReducedConvection(**VARIANTS[v], trigger_height_above_lcl_m=args.trigger_dz)
        for v in args.variants
    }
    base = schemes[args.variants[0]]
    turb = get_scheme(cfg.turbulence.scheme)
    channels = tuple(
        dict.fromkeys(("u", "v", "w", "blh", "sp", *turb.required_met_keys(), "t", "q"))
    )
    bounds = _met_domain_bounds(cfg)
    if args.met_domain is not None:
        lon0, lon1, lat0, lat1 = args.met_domain
        bounds = replace(bounds, lon_min=lon0, lon_max=lon1, lat_min=lat0, lat_max=lat1)
    out: dict[str, np.ndarray] = {}
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
            cols0 = base.column_state(
                met, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32
            )
            lat = np.asarray(met.metadata.lat, dtype=np.float64)
            lon = np.asarray(met.metadata.lon, dtype=np.float64)
            lat_c = np.repeat(lat, lon.size)
            key = f"L{nlev}/{t0.strftime('%Y%m%dT%H')}"
            if args.trigger_dz is not None:
                key = f"L{nlev}T{args.trigger_dz:g}/{t0.strftime('%Y%m%dT%H')}"
            n_lev = cols0.t.shape[1]
            # Trigger checks on every column.
            has_lcl = cols0.i_lcl >= 0
            chk = {
                "has_lcl": has_lcl,
                "depth_pos": cols0.i_lnb > cols0.i_lcl,
                "cape": cols0.cape.double() >= base.min_cape_j_kg,
                "dtv": cols0.dtv_lcl_plus_1.double() >= base.trigger_dtv_k,
                "cdepth": cols0.cloud_depth_m.double() >= base.min_cloud_depth_m,
            }
            for name, c in chk.items():
                out[f"{key}/chk_{name}"] = c.numpy()
            out[f"{key}/cosphi"] = np.cos(np.deg2rad(lat_c))
            fire = cols0.fires
            idx = fire.nonzero().squeeze(1)
            out[f"{key}/fires"] = fire.numpy()
            out[f"{key}/idx"] = idx.numpy()
            # Height of the LCL+1 level above the LCL level (the trigger's test depth).
            h = cols0.height_m.double()
            il = cols0.i_lcl.clamp(min=0, max=n_lev - 2)
            out[f"{key}/dz_test"] = (
                (h.gather(1, (il + 1).unsqueeze(1)) - h.gather(1, il.unsqueeze(1)))
                .squeeze(1)
                .numpy()
            )
            out[f"{key}/h_lcl"] = h.gather(1, il.unsqueeze(1)).squeeze(1).numpy()
            # ... and above the LCL itself (height interpolated in ln p).
            t_s, q_s, p_s = (
                cols0.t[:, 0].double(),
                cols0.q[:, 0].double(),
                cols0.pressure_pa[:, 0].double(),
            )
            p_lcl = em.lcl_pressure_poisson(t_s, em.lcl_temperature_bolton(t_s, q_s, p_s), p_s)
            pp = cols0.pressure_pa.double()
            hi = cols0.i_lcl.clamp(min=1, max=n_lev - 1).unsqueeze(1)
            lp = torch.log(pp)
            a, b = lp.gather(1, hi - 1).squeeze(1), lp.gather(1, hi).squeeze(1)
            w = ((a - torch.log(p_lcl)) / (a - b)).clamp(0, 1)
            z_lcl = h.gather(1, hi - 1).squeeze(1) + w * (
                h.gather(1, hi).squeeze(1) - h.gather(1, hi - 1).squeeze(1)
            )
            out[f"{key}/z_lcl"] = z_lcl.numpy()
            out[f"{key}/dz_test_lcl"] = (
                h.gather(1, (il + 1).unsqueeze(1)).squeeze(1) - z_lcl
            ).numpy()
            out[f"{key}/dtv"] = cols0.dtv_lcl_plus_1.numpy()
            out[f"{key}/cape"] = cols0.cape.numpy()
            out[f"{key}/dtmin"] = cols0.dtmin_k.numpy()
            out[f"{key}/depth"] = cols0.cloud_depth_m.numpy()
            if idx.numel() == 0:
                print(f"{key}: no firing column", flush=True)
                continue
            m_all = em._layer_masses_per_area(cols0.pressure_pa.double()).clamp(min=1e-6)
            for v, scheme in schemes.items():
                cols = scheme.column_state(
                    met, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32
                )
                assert torch.equal(cols.fires, fire)
                acc = {k: [] for k in ("mb", "courant", "E", "f3", "f6", "E3x1", "E3")}
                for ch in torch.split(idx, 1024):
                    rate = em.compute_mass_flux_matrices(
                        cols.pressure_pa[ch].double(),
                        cols.i_lcl[ch],
                        cols.i_lnb[ch],
                        cols.m_b[ch].double(),
                        cfl_max_outflux_frac=None,
                    )
                    courant = (rate.sum(2) / m_all[ch]).amax(1) * cols.interval_s
                    e, f3, f6, p1 = transport_metrics(scheme, cols, ch, cols.interval_s, h)
                    m = m_all[ch]
                    k = torch.arange(m.shape[1])
                    src = (k.unsqueeze(0) < cols.i_lcl[ch].unsqueeze(1)).double()
                    p3 = p1 @ p1 @ p1
                    e3x1 = (torch.einsum("ci,cij->cj", src * m, p3) * (1 - src)).sum(1)
                    e3, _, _, _ = transport_metrics(scheme, cols, ch, 3.0 * cols.interval_s, h)
                    for name, val in (
                        ("mb", cols.m_b[ch]),
                        ("courant", courant),
                        ("E", e),
                        ("f3", f3),
                        ("f6", f6),
                        ("E3x1", e3x1),
                        ("E3", e3),
                    ):
                        acc[name].append(val.double().numpy())
                for name, val in acc.items():
                    out[f"{key}/{v}/{name}"] = np.concatenate(val)
            out[f"{key}/interval_s"] = np.asarray(cols0.interval_s)
            print(
                f"{key}: {fire.numel()} columns x {n_lev} levels, {idx.numel()} fire, "
                f"{time.perf_counter() - tic:.1f} s",
                flush=True,
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)


if __name__ == "__main__":
    main()
