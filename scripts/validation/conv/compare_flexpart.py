"""Task CONV: GLIDE closure variants against FLEXPART 11's CONVECT, same hours.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree>/src .venv/bin/python \
        scripts/validation/conv/compare_flexpart.py <config> <glide.npz> <grid key, e.g. L40> \
        <flexpart.npz> [<flexpart.npz> ...] [--zarr-store S --met-domain ...]

Joins columns on (lat, lon): GLIDE's column c = iy * n_lon + ix of its met window
(read once to get the coordinates); FLEXPART's columns carry their own. Prints:

1. Domain: GLIDE's firing fraction and FLEXPART's (CBMF > 0 after spin-up), and
   the area-weighted domain sums (cos lat) of GLIDE's M_b and transport E/Δt per
   variant, and of FLEXPART's CBMF and E/Δt.
2. In GLIDE's firing columns: FLEXPART's CBMF and transport there.
3. FLEXPART's own cadence: CBMF after one 3600 s call vs four 900 s calls, and
   how often its matrix would move more than a level holds (negative stay).

FLEXPART runs are labelled ``delt:ncalls:nspin`` (flexpart_convect.py); the
steady hour is ``900:12:8`` (CBMF spun up for 2 h, tracer moved in the last 4
calls), the cold-start hour ``900:4:0``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from lpdm.config import RunConfig
from lpdm.main import _met_domain_bounds, _met_interval
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, TimeBounds


def pct(x):
    x = np.asarray(x, float)
    return (
        "n/a"
        if x.size == 0
        else " / ".join(f"{v:.3g}" for v in np.percentile(x, [0, 10, 50, 90, 100]))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("glide")
    ap.add_argument("grid")
    ap.add_argument("flexpart", nargs="+")
    ap.add_argument("--zarr-store", default=None)
    ap.add_argument("--met-domain", nargs=4, type=float, default=None)
    args = ap.parse_args()

    g = np.load(args.glide)
    f = {}
    for p in args.flexpart:
        z = np.load(p)
        f.update({k: z[k] for k in z.files})
    hours = sorted({k.split("/")[1] for k in g.files if k.startswith(args.grid + "/")})
    variants = sorted(
        {k.split("/")[2] for k in g.files if k.startswith(args.grid + "/") and k.count("/") == 3}
    )
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
    t0 = datetime.strptime(hours[0], "%Y%m%dT%H").replace(tzinfo=UTC)
    met = reader.fetch_hourly_window(
        BoundingBoxRequest(
            spatial=bounds, time=TimeBounds(start=t0, end=t0 + _met_interval(reader))
        )
    )
    glat = np.repeat(np.asarray(met.metadata.lat, float), np.asarray(met.metadata.lon).size)
    glon = np.tile(np.asarray(met.metadata.lon, float), np.asarray(met.metadata.lat).size)
    gkey = np.round(glat * 4).astype(np.int64) * 100000 + np.round(glon * 4).astype(np.int64)

    agg = {"g_fire": 0, "g_n": 0, "f_fire": 0, "f_n": 0}
    sums: dict[str, float] = {}
    in_fire: dict[str, list] = {}
    cad = {"cb1x3600": [], "cb4x900": [], "neg900": 0, "neg3600": 0, "n": 0}
    for h in hours:
        fk = f"{h[:8]}T{h[9:11]}"
        if f"{fk}/lat" not in f:
            print(f"{h}: no FLEXPART data")
            continue
        flat, flon = f[f"{fk}/lat"], f[f"{fk}/lon"]
        fkey = np.round(flat * 4).astype(np.int64) * 100000 + np.round(flon * 4).astype(np.int64)
        order = np.argsort(fkey)
        pos = np.searchsorted(fkey[order], gkey)
        ok = (pos < fkey.size) & (fkey[order][np.clip(pos, 0, fkey.size - 1)] == gkey)
        g2f = np.where(ok, order[np.clip(pos, 0, fkey.size - 1)], -1)
        cos_g = g[f"{args.grid}/{h}/cosphi"]
        fires = g[f"{args.grid}/{h}/fires"]
        idx = g[f"{args.grid}/{h}/idx"]
        dt = float(g[f"{args.grid}/{h}/interval_s"])
        agg["g_fire"] += int(fires.sum())
        agg["g_n"] += fires.size
        # FLEXPART over the same columns as GLIDE's domain
        sel = g2f[ok]
        cbs = f[f"{fk}/900:12:8/cbmf"][:, -1].astype(float)
        cos_f = np.cos(np.deg2rad(flat))
        agg["f_fire"] += int((cbs[sel] > 0).sum())
        agg["f_n"] += sel.size
        for run, lab in (
            ("900:12:8", "FLEXPART steady (900 s)"),
            ("900:4:0", "FLEXPART cold start (900 s)"),
            ("3600:1:0", "FLEXPART hourly, cold start"),
            ("3600:4:3", "FLEXPART hourly, 4th call"),
        ):
            if f"{fk}/{run}/cbmf" not in f:
                continue
            cb = f[f"{fk}/{run}/cbmf"][:, -1].astype(float)
            ncall_tracer = {"900:12:8": 4, "900:4:0": 4, "3600:1:0": 1, "3600:4:3": 1}[run]
            dtr = float(run.split(":")[0]) * ncall_tracer
            e = f[f"{fk}/{run}/E"].astype(float)
            sums[f"{lab}: sum CBMF cos"] = sums.get(f"{lab}: sum CBMF cos", 0.0) + float(
                (cb[sel] * cos_f[sel]).sum()
            )
            sums[f"{lab}: sum E/dt cos (matrix only)"] = sums.get(
                f"{lab}: sum E/dt cos (matrix only)", 0.0
            ) + float((e[sel] / dtr * cos_f[sel]).sum())
            fi = g2f[idx]
            good = fi >= 0
            in_fire.setdefault(f"{lab}: CBMF", []).append(cb[fi[good]])
            in_fire.setdefault(f"{lab}: E/dt (matrix only)", []).append(e[fi[good]] / dtr)
            in_fire.setdefault(f"{lab}: f3", []).append(f[f"{fk}/{run}/f3"][fi[good]])
            in_fire.setdefault(f"{lab}: f6", []).append(f[f"{fk}/{run}/f6"][fi[good]])
            in_fire.setdefault(f"{lab}: fraction with CBMF > 0", []).append(
                (cb[fi[good]] > 0).astype(float)
            )
        for v in variants:
            mb = g[f"{args.grid}/{h}/{v}/mb"]
            e = g[f"{args.grid}/{h}/{v}/E"]
            c = cos_g[idx]
            sums[f"GLIDE {v}: sum M_b cos"] = sums.get(f"GLIDE {v}: sum M_b cos", 0.0) + float(
                (mb * c).sum()
            )
            sums[f"GLIDE {v}: sum E/dt cos"] = sums.get(f"GLIDE {v}: sum E/dt cos", 0.0) + float(
                (e / dt * c).sum()
            )
            in_fire.setdefault(f"GLIDE {v}: M_b", []).append(mb)
            in_fire.setdefault(f"GLIDE {v}: E/dt", []).append(e / dt)
            in_fire.setdefault(f"GLIDE {v}: f3", []).append(g[f"{args.grid}/{h}/{v}/f3"])
            in_fire.setdefault(f"GLIDE {v}: f6", []).append(g[f"{args.grid}/{h}/{v}/f6"])
        if f"{fk}/3600:1:0/cbmf" in f and f"{fk}/900:4:0/cbmf" in f:
            a = f[f"{fk}/3600:1:0/cbmf"][sel, -1].astype(float)
            b = f[f"{fk}/900:4:0/cbmf"][sel, -1].astype(float)
            m = b > 0
            cad["cb1x3600"].append(a[m])
            cad["cb4x900"].append(b[m])
            cad["neg900"] += int((f[f"{fk}/900:12:8/mindiag"][sel][cbs[sel] > 0] < 0).sum())
            cad["neg3600"] += int((f[f"{fk}/3600:4:3/mindiag"][sel][cbs[sel] > 0] < 0).sum())
            cad["n"] += int((cbs[sel] > 0).sum())
    print(f"hours: {' '.join(hours)}; GLIDE grid {args.grid}")
    print(
        f"\n1. firing: GLIDE {agg['g_fire']} / {agg['g_n']} column-hours ({agg['g_fire'] / agg['g_n']:.3%}); "
        f"FLEXPART (CBMF > 0, steady) {agg['f_fire']} / {agg['f_n']} ({agg['f_fire'] / max(agg['f_n'], 1):.3%})"
    )
    print("   domain sums over all hours (kg m-2 s-1 x columns, cos-lat weighted):")
    for k, v in sums.items():
        print(f"     {k:55s} {v:10.4g}")
    print("\n2. in GLIDE's firing columns (percentiles 0/10/50/90/100):")
    for k, v in in_fire.items():
        print(f"     {k:55s} {pct(np.concatenate(v))}")
    if cad["n"]:
        r = np.concatenate(cad["cb1x3600"]) / np.concatenate(cad["cb4x900"])
        print(
            f"\n3. FLEXPART cadence: CBMF after one 3600 s call / after four 900 s calls (cold start): {pct(r)}\n"
            f"   columns with CBMF > 0 whose matrix moves more than a level holds (negative stay): "
            f"900 s steady {cad['neg900']} / {cad['n']} ({cad['neg900'] / cad['n']:.2%}); 3600 s {cad['neg3600']} ({cad['neg3600'] / cad['n']:.2%})"
        )


if __name__ == "__main__":
    main()
