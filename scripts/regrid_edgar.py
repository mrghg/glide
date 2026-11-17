#!/usr/bin/env python
"""Regrid a global EDGAR CH4 flux map onto the validation output grids (EUR, NAM).

The target grids are the 0.25 deg footprint output grids of
``docs/reference-runs.md`` §2 (cells centred on ERA5 grid points, one ring in
from the particle domain), so the regridded flux multiplies GLIDE, FLEXPART and
NAME footprints cell by cell with no further regridding.

EDGAR fluxes are per unit area (kg m-2 s-1). ``regrid_conservative`` works on
per-cell quantities, so the flux is multiplied by the source-cell area,
regridded, and divided by the target-cell area. The output is converted to
mol m-2 s-1 (the unit ``scripts/make_validation_timeseries.py`` expects).
Source values are not altered: EDGAR's few negative cells are kept.

Usage:

    python scripts/regrid_edgar.py EDGAR_2025_GHG_CH4_2019_TOTALS_flx.nc --year 2019 \
        --out-dir "$GLIDE_EDGAR"

Writes ``EDGAR_CH4_<year>_<DOMAIN>_0p25.nc`` per domain and prints the domain
totals before and after regridding as a conservation check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from lpdm.comparison import regrid_conservative

EARTH_RADIUS_M = 6_371_000.0
CH4_KG_PER_MOL = 0.01604
SECONDS_PER_YEAR = 365.25 * 86400

# Output cell centres, docs/reference-runs.md §2.
GRIDS = {
    "EUR": {"lat": (32.25, 71.75), "lon": (-29.75, 39.75)},
    "NAM": {"lat": (22.25, 57.75), "lon": (-119.75, -55.25)},
}
RES = 0.25


def _cell_area(lat_c: np.ndarray, d_lat: float, d_lon: float) -> np.ndarray:
    """Spherical cell area (m2) per latitude row for a regular grid."""
    half = np.deg2rad(d_lat / 2)
    la = np.deg2rad(lat_c)
    return EARTH_RADIUS_M**2 * (np.sin(la + half) - np.sin(la - half)) * np.deg2rad(d_lon)


def _spacing(c: np.ndarray) -> float:
    d = np.diff(c)
    if not np.allclose(d, d[0], rtol=1e-6):
        raise ValueError("source grid is not regular")
    return float(d[0])


def _box_total(mass: xr.DataArray, s: float, n: float, w: float, e: float) -> float:
    """Sum per-cell mass over a lat/lon box, partial cells weighted by area overlap."""
    lat, lon = mass.latitude.values, mass.longitude.values
    dla, dlo = _spacing(lat), _spacing(lon)
    lo_e, hi_e = lat - dla / 2, lat + dla / 2
    ov_lat = np.clip(
        np.sin(np.deg2rad(np.clip(hi_e, s, n))) - np.sin(np.deg2rad(np.clip(lo_e, s, n))), 0, None
    )
    f_lat = ov_lat / (np.sin(np.deg2rad(hi_e)) - np.sin(np.deg2rad(lo_e)))
    f_lon = np.clip(np.minimum(lon + dlo / 2, e) - np.maximum(lon - dlo / 2, w), 0, None) / dlo
    return float((mass.values * f_lat[:, None] * f_lon[None, :]).sum())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=Path, help="global EDGAR flux file (kg m-2 s-1)")
    ap.add_argument("--year", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--var", default="fluxes")
    args = ap.parse_args()

    ds = xr.open_dataset(args.src)
    flux = ds[args.var].rename({"lat": "latitude", "lon": "longitude"}).astype("float64").load()
    units = flux.attrs.get("units", ds.attrs.get("units"))
    if units != "kg m-2 s-1":
        raise ValueError(f"expected kg m-2 s-1, got {units!r}")

    lat_s, lon_s = flux.latitude.values, flux.longitude.values
    a_src = _cell_area(lat_s, _spacing(lat_s), _spacing(lon_s))[:, None]
    mass_src = flux * a_src  # kg s-1 per source cell

    for name, g in GRIDS.items():
        lat_t = np.arange(g["lat"][0], g["lat"][1] + RES / 2, RES)
        lon_t = np.arange(g["lon"][0], g["lon"][1] + RES / 2, RES)
        mass_t = regrid_conservative(mass_src, target_latitude=lat_t, target_longitude=lon_t)
        a_t = _cell_area(lat_t, RES, RES)[:, None]
        out = (mass_t / a_t / CH4_KG_PER_MOL).astype("float32")

        # Conservation check, independent of the regridder: source mass inside the
        # box edges, each source cell weighted by its fractional overlap.
        tot_out = float(mass_t.sum()) * SECONDS_PER_YEAR / 1e9
        tot_ref = (
            _box_total(
                mass_src,
                lat_t[0] - RES / 2,
                lat_t[-1] + RES / 2,
                lon_t[0] - RES / 2,
                lon_t[-1] + RES / 2,
            )
            * SECONDS_PER_YEAR
            / 1e9
        )
        rel = abs(tot_out - tot_ref) / tot_ref
        if rel > 1e-6:
            raise RuntimeError(f"{name}: regridded total {tot_out} != source box total {tot_ref}")

        out.name = "flux"
        out.attrs = {
            "units": "mol m-2 s-1",
            "long_name": f"EDGAR CH4 total emission flux, {args.year} annual mean",
            "species": "CH4",
            "molar_mass_kg_mol": CH4_KG_PER_MOL,
        }
        out_ds = out.to_dataset()
        out_ds.latitude.attrs = {"units": "degrees_north", "long_name": "cell centre latitude"}
        out_ds.longitude.attrs = {"units": "degrees_east", "long_name": "cell centre longitude"}
        out_ds.attrs = {
            "title": f"EDGAR CH4 {args.year} on the GLIDE validation {name} grid",
            "source_file": args.src.name,
            "source": ds.attrs.get("source", ""),
            "how_to_cite": ds.attrs.get("how_to_cite", ""),
            "copyright_notice": ds.attrs.get("copyright_notice", ""),
            "grid": f"{name}, 0.25 deg, docs/reference-runs.md section 2 ({lon_t.size} x {lat_t.size} cells)",
            "method": "area-weighted conservative regridding (lpdm.comparison.regrid_conservative), "
            "flux x source cell area -> regrid -> / target cell area; kg -> mol",
            "domain_total_Tg_per_yr": tot_out,
        }
        path = args.out_dir / f"EDGAR_CH4_{args.year}_{name}_0p25.nc"
        out_ds.to_netcdf(path)
        print(
            f"{name}: {lon_t.size}x{lat_t.size}  total {tot_out:.3f} Tg/yr "
            f"(source box {tot_ref:.3f}, rel diff {rel:.1e})  "
            f"min {float(out.min()):.3e}  -> {path}"
        )


if __name__ == "__main__":
    main()
