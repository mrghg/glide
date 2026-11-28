"""Decision 0031: what separates the cloud from the buoyant layer above it, in the
columns that stop firing.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<checkout>/src .venv/bin/python \
        scripts/validation/cape/gap_diagnostics.py configs/ab_multisite_perf.yaml \
        before.npz after.npz [--zarr-store ... --met-domain LON0 LON1 LAT0 LAT1]

For every window in the two ``trigger_counts.py`` outputs, the columns that fire
before and not after are re-lifted (the same arithmetic as
`compute_lcl_lnb_cape_columns`) and, per column, the stable gap between the
cloud top (first LNB) and the next buoyant level is described: its number of
levels and thickness, its most negative buoyancy as a T_v deficit (K), its
signed energy g·Σ B dz (J/kg, negative), and the positive energy above it. Also
the cloud's depth and the parcel's highest buoyant level, the cloud top
FLEXPART would take.
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


def buoyancy(t: torch.Tensor, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Parcel buoyancy [C, Z] exactly as `compute_lcl_lnb_cape_columns` forms it."""

    t_s, q_s, p_s = t[:, 0], q[:, 0], p[:, 0]
    t_lcl = em.lcl_temperature_bolton(t_s, q_s, p_s)
    p_lcl = em.lcl_pressure_poisson(t_s, t_lcl, p_s)
    theta_e = em.equivalent_potential_temperature(t_s, q_s, p_s)
    dry_t = em.potential_temperature(t_s, p_s).unsqueeze(1) * (p / em.P0_PA).pow(em.KAPPA_POISSON)
    moist_t = em.lift_parcel_moist_pseudo_adiabatic(theta_e.unsqueeze(1), p)
    below = p >= p_lcl.unsqueeze(1)
    t_par = torch.where(below, dry_t, moist_t)
    q_par = torch.where(
        below, q_s.unsqueeze(1).expand_as(t_par), em.saturation_specific_humidity(t_par, p)
    )
    tv_env = em.virtual_temperature(t, q)
    tv_par = em.virtual_temperature(t_par, q_par)
    return (tv_par - tv_env) / tv_env.clamp(min=em.T_MIN_K), tv_par - tv_env


def pct(x) -> str:
    x = np.asarray(x, dtype=float)
    return " / ".join(f"{v:.3g}" for v in np.percentile(x, [0, 10, 50, 90, 100]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("before")
    ap.add_argument("after")
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
    before, after = np.load(args.before), np.load(args.after)
    keys = sorted({k.split("/")[0] for k in before.files})
    rows = {
        k: []
        for k in (
            "gap_levels",
            "gap_m",
            "gap_min_dtv",
            "gap_cin",
            "cape_above",
            "cloud_depth",
            "top_m",
            "highest_top_m",
        )
    }
    for key in keys:
        stop = np.nonzero(before[f"{key}/fires"] & ~after[f"{key}/fires"])[0]
        if stop.size == 0:
            continue
        t0 = datetime.strptime(key, "%Y%m%dT%H").replace(tzinfo=UTC)
        met = reader.fetch_hourly_window(
            BoundingBoxRequest(
                spatial=bounds, time=TimeBounds(start=t0, end=t0 + _met_interval(reader))
            )
        )
        cols = conv.column_state(met, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32)
        idx = torch.as_tensor(stop)
        t, q, p, h = cols.t[idx], cols.q[idx], cols.pressure_pa[idx], cols.height_m[idx]
        b, dtv = buoyancy(t, q, p)
        dz = torch.zeros_like(t)
        dz[:, 1:] = h[:, 1:] - h[:, :-1]
        for j, c in enumerate(stop):
            lnb = int(cols.i_lnb[c])
            bj = b[j].numpy()
            nxt = lnb + int(np.argmax(bj[lnb:] > 0.0))
            pos_above = np.nonzero(bj[lnb:] > 0.0)[0]
            highest = lnb + int(pos_above[-1]) if pos_above.size else lnb
            gap = slice(lnb, nxt)
            rows["gap_levels"].append(nxt - lnb)
            rows["gap_m"].append(float(h[j, nxt - 1] - h[j, lnb - 1]) if nxt > lnb else 0.0)
            rows["gap_min_dtv"].append(float(dtv[j, gap].min()))
            rows["gap_cin"].append(float(em.GRAVITY_M_S2 * (b[j, gap] * dz[j, gap]).sum()))
            rows["cape_above"].append(
                float(em.GRAVITY_M_S2 * (b[j, nxt:].clamp(min=0) * dz[j, nxt:]).sum())
            )
            rows["cloud_depth"].append(float(cols.cloud_depth_m[c]))
            rows["top_m"].append(float(h[j, lnb]))
            rows["highest_top_m"].append(float(h[j, highest]))
        print(f"{key}: {stop.size} columns stop firing")
    n = len(rows["gap_levels"])
    print(f"\ncolumns that stop firing: {n}  (percentiles 0/10/50/90/100)")
    gl = np.asarray(rows["gap_levels"])
    print(f"stable gap, levels: {pct(gl)}; one level: {np.mean(gl == 1):.3f}")
    print(f"stable gap thickness, m: {pct(rows['gap_m'])}")
    gm = np.asarray(rows["gap_min_dtv"])
    print(f"most negative T_v excess in the gap, K: {pct(gm)}; > -0.1 K: {np.mean(gm > -0.1):.3f}")
    print(f"signed energy of the gap (CIN), J/kg: {pct(rows['gap_cin'])}")
    print(f"positive energy above the gap, J/kg: {pct(rows['cape_above'])}")
    print(f"cloud depth (first LNB), m: {pct(rows['cloud_depth'])}")
    print(f"cloud top (first LNB) height, m: {pct(rows['top_m'])}")
    print(f"highest buoyant level height, m: {pct(rows['highest_top_m'])}")


if __name__ == "__main__":
    main()
