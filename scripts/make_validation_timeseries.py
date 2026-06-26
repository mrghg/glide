#!/usr/bin/env python
"""Generate CH4 validation timeseries from FLEXPART / NAME footprints × EDGAR flux.

For each site we read the monthly ``inert`` (passive-tracer) footprints, multiply by
the regridded EDGAR CH4 flux, and integrate over the domain to get an hourly CH4
mole-fraction *enhancement* (ppb):

    ch4_ppb(t) = 1e9 · Σ_ij  srr(t, i, j) · flux(i, j)

The footprint ``srr`` ("source-receptor relationship") is already in
``(mol/mol)/(mol m-2 s-1)`` and the EDGAR flux in ``mol m-2 s-1``, so the product is
a mole fraction (mol/mol) — no extra area weighting (STILT/Lin 2003 convention). The
EDGAR flux is the 2024 annual total, so all temporal structure comes from transport.

Both models' footprints and the EDGAR flux share the **EUROPE 293×391 grid**
(``data/EDGAR_CH4_2024_MHD_grid.nc``), so no regridding is needed; we assert the grid
matches before multiplying (a tiny float mismatch would otherwise silently zero the
xarray product — hence we work in NumPy with an explicit grid check).

One CSV per (site, model, met-driver) is written to ``data/validation-timeseries/``.
The commented header records the release lon/lat and height (m AGL, from the folder
name, e.g. ``BSD-248magl`` → 248 m) so GLIDE can be run with matching release
parameters to reproduce the timeseries. NAME comes under two met drivers (UMG, UKV);
both are emitted where available. The higher-resolution ``EUROPE-6km`` NAME variant
is skipped (different grid).

Footprints live outside the repo (too large); only the small timeseries are stored.

Usage (defaults target this machine's shared archive + 2024):
    python scripts/make_validation_timeseries.py
    python scripts/make_validation_timeseries.py --sites MHD-10magl,BSD-248magl   # subset
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# SITE-HEIGHTmagl, tolerating '-' or '_' before the height (e.g. CGR_10magl).
_LABEL_RE = re.compile(r"^(?P<site>.+)[-_](?P<height>\d+)magl$")
GRID_ATOL_DEG = 1e-3  # footprint grid must match the EDGAR grid to this tolerance
PPB = 1.0e9


def parse_label(label: str) -> tuple[str, float]:
    """`BSD-248magl` -> ('BSD', 248.0). Height is metres above ground level."""
    m = _LABEL_RE.match(label)
    if not m:
        raise ValueError(f"cannot parse site/height from folder name {label!r}")
    return m.group("site"), float(m.group("height"))


def parse_filename(label: str, basename: str) -> dict | None:
    """Parse the footprint filename fields after stripping the site-label prefix.

    `<label>_<model>_<met>_<domain>_<species>_<YYYYMM>.nc`. Stripping the label first
    is robust to underscores inside the label (CGR_10magl). Returns None if it doesn't
    match the expected layout.
    """
    prefix = label + "_"
    if not basename.startswith(prefix) or not basename.endswith(".nc"):
        return None
    fields = basename[len(prefix):-len(".nc")].split("_")
    if len(fields) != 5:
        return None
    model, met, domain, species, yyyymm = fields
    return {"model": model, "met": met, "domain": domain, "species": species, "yyyymm": yyyymm}


def load_flux(edgar_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load the EDGAR flux as float32 plus its reference lat/lon arrays and units."""
    ds = xr.open_dataset(edgar_path)
    flux = np.nan_to_num(ds["flux"].values, nan=0.0).astype(np.float32)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    units = ds["flux"].attrs.get("units", "mol m-2 s-1")
    ds.close()
    return flux, lat, lon, units


def enhancement_for_file(
    path: Path, flux: np.ndarray, ref_lat: np.ndarray, ref_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float | None]:
    """Hourly ppb enhancement for one monthly footprint file (+ release metadata)."""
    ds = xr.open_dataset(path)
    try:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        if lat.shape != ref_lat.shape or lon.shape != ref_lon.shape or not (
            np.allclose(lat, ref_lat, atol=GRID_ATOL_DEG)
            and np.allclose(lon, ref_lon, atol=GRID_ATOL_DEG)
        ):
            raise ValueError(
                f"grid mismatch ({lat.size}x{lon.size}) vs EDGAR "
                f"({ref_lat.size}x{ref_lon.size}); skipping {path.name}"
            )
        srr = np.nan_to_num(ds["srr"].values, nan=0.0)  # (time, lat, lon), float32
        times = ds["time"].values
        rlon = float(np.asarray(ds["release_lon"].values).ravel()[0])
        rlat = float(np.asarray(ds["release_lat"].values).ravel()[0])
        rh = (
            float(np.asarray(ds["release_height"].values).ravel()[0])
            if "release_height" in ds
            else None
        )
    finally:
        ds.close()
    # Sum over the spatial dims in float64 for accuracy (many small cells).
    enh = (srr * flux[None, :, :]).sum(axis=(1, 2), dtype=np.float64) * PPB
    return times, enh, rlon, rlat, rh


def write_csv(
    out_path: Path,
    *,
    label: str,
    site: str,
    height_magl: float,
    model: str,
    met: str,
    domain: str,
    release_lon: float,
    release_lat: float,
    release_height_file: float | None,
    flux_units: str,
    edgar_name: str,
    times: np.ndarray,
    enh: np.ndarray,
    months: list[str],
) -> None:
    order = np.argsort(times)
    times = times[order]
    enh = enh[order]
    iso = np.datetime_as_string(times.astype("datetime64[s]"), unit="s")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rh_file = "n/a" if release_height_file is None else f"{release_height_file:.1f}"
    header = [
        "# GLIDE validation CH4 enhancement timeseries",
        f"# site: {site}",
        f"# inlet_label: {label}",
        f"# model: {model}",
        f"# met_model: {met}",
        f"# domain: {domain}",
        f"# release_lon_deg: {release_lon:.4f}",
        f"# release_lat_deg: {release_lat:.4f}",
        f"# release_height_magl: {height_magl:.1f}    # from folder name; use this for the GLIDE release",
        f"# release_height_in_file_m: {rh_file}    # footprint 'release_height' (above model ground)",
        f"# flux: EDGAR CH4 2024 annual total ({edgar_name}), units {flux_units}",
        "# method: ch4_enhancement_ppb = 1e9 * sum_ij( srr[t,i,j] * flux[i,j] );  srr units (mol/mol)/(mol m-2 s-1)",
        "# footprint_grid: EUROPE 293x391 (matches the EDGAR flux grid)",
        f"# coverage_months: {months[0]}..{months[-1]} ({len(months)} of 12)",
        f"# n_times: {len(times)}",
        f"# generated: {now} by scripts/make_validation_timeseries.py",
        "# columns: time (UTC, ISO 8601), ch4_enhancement_ppb",
        "time,ch4_enhancement_ppb",
    ]
    lines = [f"{t}Z,{v:.4f}" for t, v in zip(iso, enh)]
    out_path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def process_group(
    files: list[Path],
    *,
    label: str,
    model: str,
    met: str,
    domain: str,
    flux: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
    flux_units: str,
    edgar_name: str,
    out_dir: Path,
) -> str:
    """Build + write one CSV for a (label, model, met) group of monthly files."""
    site, height_magl = parse_label(label)
    all_times: list[np.ndarray] = []
    all_enh: list[np.ndarray] = []
    months: list[str] = []
    rlon = rlat = rh = None
    for path in files:
        info = parse_filename(label, path.name)
        try:
            times, enh, rlon_f, rlat_f, rh_f = enhancement_for_file(path, flux, ref_lat, ref_lon)
        except (ValueError, KeyError, OSError) as exc:
            print(f"    skip {path.name}: {exc}", file=sys.stderr)
            continue
        all_times.append(times)
        all_enh.append(enh)
        months.append(info["yyyymm"])
        rlon, rlat, rh = rlon_f, rlat_f, rh_f
    if not all_times:
        return f"  {label} {model} {met}: no usable files"

    times = np.concatenate(all_times)
    enh = np.concatenate(all_enh)
    out_path = out_dir / f"{label}_{model}_{met}.csv"
    write_csv(
        out_path, label=label, site=site, height_magl=height_magl, model=model, met=met,
        domain=domain, release_lon=rlon, release_lat=rlat, release_height_file=rh,
        flux_units=flux_units, edgar_name=edgar_name, times=times, enh=enh,
        months=sorted(months),
    )
    return (
        f"  wrote {out_path.name}  ({len(times)} h, {len(months)} mo, "
        f"lon={rlon:.3f} lat={rlat:.3f} z={height_magl:.0f}magl, "
        f"mean={np.nanmean(enh):.2f} max={np.nanmax(enh):.1f} ppb)"
    )


def discover_groups(model_root: Path, year: int, sites: set[str] | None) -> dict:
    """Map (site_label, met) -> sorted list of monthly standard-EUROPE inert files."""
    groups: dict[tuple[str, str], list[Path]] = {}
    if not model_root.is_dir():
        return groups
    for site_dir in sorted(model_root.iterdir()):
        label = site_dir.name
        if sites is not None and label not in sites:
            continue
        inert = site_dir / "inert"
        if not inert.is_dir():
            continue
        for path in inert.glob(f"*_{year}??.nc"):
            info = parse_filename(label, path.name)
            if info is None or info["species"] != "inert" or info["domain"] != "EUROPE":
                continue  # skip EUROPE-6km (different grid) and malformed names
            groups.setdefault((label, info["met"]), []).append(path)
    return {k: sorted(v) for k, v in groups.items()}


def main() -> None:
    home = Path(os.path.expanduser("~"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flexpart-root", type=Path, default=home / "shared/LPDM/fp_FLEXPART/EUROPE")
    ap.add_argument("--name-root", type=Path, default=home / "shared/LPDM/fp_NAME/EUROPE")
    ap.add_argument("--edgar", type=Path, default=Path("data/EDGAR_CH4_2024_MHD_grid.nc"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/validation-timeseries"))
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--sites", type=str, default=None, help="comma-separated site labels to restrict to (default: all)")
    args = ap.parse_args()

    sites = set(s.strip() for s in args.sites.split(",")) if args.sites else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flux, ref_lat, ref_lon, flux_units = load_flux(args.edgar)
    edgar_name = args.edgar.name
    print(f"EDGAR flux: {edgar_name}  grid {ref_lat.size}x{ref_lon.size}  units {flux_units}")

    n_written = 0
    for model, root in (("FLEXPART", args.flexpart_root), ("NAME", args.name_root)):
        groups = discover_groups(root, args.year, sites)
        print(f"\n{model}: {len(groups)} (site, met) groups under {root}")
        for (label, met), files in sorted(groups.items()):
            msg = process_group(
                files, label=label, model=model, met=met, domain="EUROPE",
                flux=flux, ref_lat=ref_lat, ref_lon=ref_lon, flux_units=flux_units,
                edgar_name=edgar_name, out_dir=args.out_dir,
            )
            print(msg)
            if msg.strip().startswith("wrote"):
                n_written += 1

    print(f"\nDone: {n_written} timeseries written to {args.out_dir}/")


if __name__ == "__main__":
    main()
