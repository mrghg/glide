"""The cloud-base mass flux in firing columns on real meteorology (review question, not a change).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<checkout>/src .venv/bin/python \
        scripts/validation/cape/closure_diagnostics.py configs/ab_multisite_perf.yaml \
        --times 2023-12-30T12 ... [--zarr-store ... --met-domain LON0 LON1 LAT0 LAT1]

For the columns that fire in each window, with the scheme on ``PYTHONPATH``:
CAPE, √(2·CAPE) before the 5 m/s cap, ρ at the LCL level, M_b, the CFL scale
applied to the matrix (1 = not binding), and the fraction of the sub-cloud
layer's air mass the updraft entrains in one event after that scale.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import torch

import lpdm
from lpdm.config import RunConfig
from lpdm.convection import emanuel as em
from lpdm.convection import get_scheme as get_convection_scheme
from lpdm.main import _convection_scheme_kwargs, _met_domain_bounds, _met_interval
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, TimeBounds
from lpdm.turbulence import get_scheme


def pct(x: np.ndarray) -> str:
    q = np.percentile(x, [0, 10, 50, 90, 100])
    return " / ".join(f"{v:.3g}" for v in q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--times", nargs="+", required=True)
    ap.add_argument("--zarr-store", default=None)
    ap.add_argument("--met-domain", nargs=4, type=float, default=None)
    args = ap.parse_args()
    print(f"lpdm: {lpdm.__file__}")
    torch.set_num_threads(8)
    cfg = RunConfig.from_yaml(args.config)
    conv = get_convection_scheme(cfg.convection.scheme, **_convection_scheme_kwargs(cfg))
    turb = get_scheme(cfg.turbulence.scheme)
    channels = tuple(
        dict.fromkeys(
            ("u", "v", "w", "blh", "sp", *turb.required_met_keys(), *conv.required_met_keys())
        )
    )
    reader = ArcoEra5ZarrReader(
        zarr_store=args.zarr_store or cfg.io.zarr_store,
        channel_names=channels,
        device="cpu",
        agl_levels_m=cfg.met_domain.resolve_agl_levels(),
    )
    bounds = _met_domain_bounds(cfg)
    if args.met_domain is not None:
        lon0, lon1, lat0, lat1 = args.met_domain
        bounds = replace(bounds, lon_min=lon0, lon_max=lon1, lat_min=lat0, lat_max=lat1)
    acc = {k: [] for k in ("cape", "w", "rho", "m_b", "scale", "bl_frac")}
    for stamp in args.times:
        t0 = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
        met = reader.fetch_hourly_window(
            BoundingBoxRequest(
                spatial=bounds, time=TimeBounds(start=t0, end=t0 + _met_interval(reader))
            )
        )
        cols = conv.column_state(met, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32)
        idx = cols.fires.nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        p = cols.pressure_pa[idx].double()
        t = cols.t[idx].double()
        i_lcl, i_lnb = cols.i_lcl[idx], cols.i_lnb[idx]
        m_b = cols.m_b[idx]
        rho = p.gather(1, i_lcl.unsqueeze(1)).squeeze(1) / (
            em.R_DRY_AIR_J_KG_K * t.gather(1, i_lcl.unsqueeze(1)).squeeze(1)
        )
        for chunk in torch.split(torch.arange(idx.numel()), 4096):
            fmass = em.compute_mass_flux_matrices(
                p[chunk], i_lcl[chunk], i_lnb[chunk], m_b[chunk] * cols.interval_s
            )
            unscaled = m_b[chunk] * cols.interval_s
            m = em._layer_masses_per_area(p[chunk])
            k = torch.arange(p.shape[1])
            bl_mass = torch.where(k < i_lcl[chunk].unsqueeze(1), m, 0.0).sum(dim=1)
            src = (k < i_lcl[chunk].unsqueeze(1)).double()  # sub-cloud rows
            dst = 1.0 - src  # cloud and above
            # Updraft mass: sub-cloud rows into cloud columns (Σ e = M_b·Δt·scale).
            entrained = torch.einsum("ci,cij,cj->c", src, fmass, dst)
            # The CFL scale is the entrained mass over the unscaled M_b·Δt.
            acc["scale"].append((entrained / unscaled).numpy())
            acc["bl_frac"].append((entrained / bl_mass).numpy())
            del fmass
        acc["cape"].append(cols.cape[idx].double().numpy())
        acc["w"].append(np.sqrt(2.0 * cols.cape[idx].double().numpy()))
        acc["rho"].append(rho.numpy())
        acc["m_b"].append(m_b.numpy())
        print(f"{stamp}: {idx.numel()} firing columns")
    a = {k: np.concatenate(v) for k, v in acc.items()}
    n = a["m_b"].size
    print(f"\nfiring columns, all windows: {n}  (percentiles 0/10/50/90/100)")
    print(f"CAPE, J/kg: {pct(a['cape'])}")
    print(f"sqrt(2 CAPE) before the cap, m/s: {pct(a['w'])}")
    print(f"fraction with sqrt(2 CAPE) >= {em.UPDRAFT_W_MAX_M_S} m/s: {np.mean(a['w'] >= 5.0):.4f}")
    print(f"rho at the LCL level, kg/m3: {pct(a['rho'])}")
    print(f"M_b, kg m-2 s-1: {pct(a['m_b'])}")
    print(f"CFL scale (1 = not binding): {pct(a['scale'])}")
    print(
        f"fraction of columns where the CFL scale binds (< 0.9999): {np.mean(a['scale'] < 0.9999):.4f}"
    )
    print(f"sub-cloud air mass entrained per event (fraction): {pct(a['bl_frac'])}")


if __name__ == "__main__":
    main()
