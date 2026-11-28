"""Task CONV: FLEXPART 11's own CONVECT on ERA5 model-level columns.

    .venv/bin/python scripts/validation/conv/flexpart_convect.py <driver exe> <out.npz> \
        --zarr-store data/era5-arco/EUROPE_20231201-20231231_ERA5-ARCO.zarr \
        --times 2023-12-29T06 --box -98 39.5 10.75 79 --runs 900:4:0 900:12:8 3600:1:0 3600:4:3

Reads temperature, humidity, surface pressure and geopotential on the store's
native L137 levels (FLEXPART convects on the model levels), builds the full- and
half-level pressures from the ECMWF L137 a/b (tests/data/ecmwf_l137_ab.csv) and
runs the driver (`flexpart_driver/`, FLEXPART's CONVECT extracted from the local
checkout) on every column. Each run ``delt:ncalls:nspin`` calls CONVECT ``ncalls``
times with time step ``delt``, carrying the cloud-base mass flux CBMF between
calls as FLEXPART's convmix does, starting from CBMF = 0; the first ``nspin``
calls only spin CBMF up. A sub-cloud tracer (the air below FLEXPART's LCL) is
redistributed forward by FLEXPART's calcmatrix/redist over the remaining calls.

Saved per column and run: CBMF after every call, the last IFLAG, the smallest
stay fraction of the matrix (negative = FLEXPART's matrix would move more than a
level holds: its CFL flag), the sub-cloud air above the LCL at the end (``E``,
kg m-2) and the fractions of it above 3 and 6 km. FLEXPART's subsidence
displacement of particles the matrix does not move is not applied.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

G = 9.80665


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("out")
    ap.add_argument("--zarr-store", required=True)
    ap.add_argument("--times", nargs="+", required=True)
    ap.add_argument(
        "--box", nargs=4, type=float, required=True, metavar=("LON0", "LON1", "LAT0", "LAT1")
    )
    ap.add_argument("--runs", nargs="+", default=["900:4:0", "900:12:8", "3600:1:0", "3600:4:3"])
    ap.add_argument(
        "--ab", default=str(Path(__file__).resolve().parents[3] / "tests/data/ecmwf_l137_ab.csv")
    )
    args = ap.parse_args()

    ab = pd.read_csv(args.ab, comment="#")
    cols = {c.lower(): c for c in ab.columns}
    a_half = ab[cols["a"]].to_numpy(np.float64)  # index n = 0 (top) .. 137 (surface)
    b_half = ab[cols["b"]].to_numpy(np.float64)
    ds = xr.open_zarr(args.zarr_store)
    lon0, lon1, lat0, lat1 = args.box
    out: dict[str, np.ndarray] = {}
    for stamp in args.times:
        sub = ds.sel(time=np.datetime64(stamp), longitude=slice(lon0, lon1))
        lat = sub.latitude.values
        sub = sub.isel(latitude=np.nonzero((lat >= lat0) & (lat <= lat1))[0])
        t = sub.temperature.values.astype(np.float64)  # [137, Y, X], top first
        q = sub.specific_humidity.values.astype(np.float64)
        z = sub.geopotential.values.astype(np.float64)
        zs = sub.geopotential_at_surface.values.astype(np.float64)
        sp = sub.surface_pressure.values.astype(np.float64)
        nlev, ny, nx = t.shape
        assert nlev == 137 and a_half.size == 138, (nlev, a_half.size)
        lat_c = np.repeat(sub.latitude.values, nx)
        lon_c = np.tile(sub.longitude.values, ny)

        # Lowest level first, [C, Z].
        def col(x, nlev=nlev):
            return x.reshape(nlev, -1).T[:, ::-1]

        tc, qc, hc = col(t), np.clip(col(q), 0.0, None), (col(z) - zs.reshape(-1, 1)) / G
        ph = (a_half[None, :] + b_half[None, :] * sp.reshape(-1, 1))[
            :, ::-1
        ]  # [C, 138], surface first
        pf = 0.5 * (ph[:, :-1] + ph[:, 1:])
        p_std = (
            0.5
            * ((a_half[:-1] + b_half[:-1] * 101325.0) + (a_half[1:] + b_half[1:] * 101325.0))[::-1]
        )
        nl = int((p_std > 5000.0).sum()) + 1
        ncol = tc.shape[0]
        rec_in = np.concatenate([pf, ph, tc, qc], axis=1).astype(np.float32)
        key = stamp.replace("-", "").replace(":", "")
        out[f"{key}/lat"] = lat_c
        out[f"{key}/lon"] = lon_c
        for run in args.runs:
            delt, ncalls, nspin = (float(run.split(":")[0]), *map(int, run.split(":")[1:]))
            with tempfile.TemporaryDirectory() as tmp:
                with open(Path(tmp) / "in.bin", "wb") as f:
                    np.array([ncol, nlev, nl, ncalls, nspin], dtype=np.int32).tofile(f)
                    np.array([delt], dtype=np.float32).tofile(f)
                    rec_in.tofile(f)
                subprocess.run([args.exe], cwd=tmp, check=True)
                dt = np.dtype(
                    [
                        ("iflag", "<i4"),
                        ("nt", "<i4"),
                        ("plcl", "<f4"),
                        ("mindiag", "<f4"),
                        ("cbmf", "<f4", (ncalls,)),
                        ("tracer", "<f4", (nlev,)),
                    ]
                )
                res = np.fromfile(Path(tmp) / "out.bin", dtype=dt)
            assert res.size == ncol
            above = pf / 100.0 < res["plcl"][:, None]
            tr = res["tracer"].astype(np.float64)
            src = (~above * (ph[:, :-1] - ph[:, 1:]) / 9.81).sum(1)
            e = (tr * above).sum(1)
            rk = f"{key}/{run}"
            out[f"{rk}/cbmf"] = res["cbmf"]
            out[f"{rk}/iflag"] = res["iflag"]
            out[f"{rk}/mindiag"] = res["mindiag"]
            out[f"{rk}/E"] = e
            out[f"{rk}/src"] = src
            out[f"{rk}/f3"] = (tr * (hc >= 3000.0)).sum(1) / np.maximum(src, 1e-9)
            out[f"{rk}/f6"] = (tr * (hc >= 6000.0)).sum(1) / np.maximum(src, 1e-9)
            cb = res["cbmf"][:, -1]
            print(
                f"{key} run {run}: {ncol} columns, CBMF>0 in {(cb > 0).sum()} ({(cb > 0).mean():.3%}), "
                f"CBMF median (where >0) {np.median(cb[cb > 0]) if (cb > 0).any() else 0:.4f}, "
                f"negative stay fraction in {(res['mindiag'] < 0).sum()}",
                flush=True,
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)


if __name__ == "__main__":
    main()
