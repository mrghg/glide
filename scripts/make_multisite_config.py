#!/usr/bin/env python
"""Generate a GLIDE `multi_point_periodic` config from the validation-timeseries sites.

Scans ``data/validation-timeseries/`` for unique inlet labels, reads each inlet's release
location (lon / lat / height AGL) from its CSV header, and writes a multi-site config over
the SAME EUROPE domain + output grid as ``configs/example_mhd_january_periodic.yaml`` (which
matches the FLEXPART/NAME validation footprints, so the result is cell-for-cell comparable).

All sites share one release schedule -> they share met windows -> one fetch (and one
convection matrix, one field build, one Python step) feeds every site's particles for that
hour. The per-window fixed costs that dominate the wall are paid ONCE for all sites instead
of once per site.

    python scripts/make_multisite_config.py --n-releases 48 -o configs/multisite_validation_48h.yaml
    sbatch scripts/run_periodic_cuda.slurm configs/multisite_validation_48h.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
VTS_DIR = REPO / "data" / "validation-timeseries"

# EUROPE domain + output grid, identical to configs/example_mhd_january_periodic.yaml
# (these come from the FLEXPART validation file; do not diverge or the comparison breaks).
OUTPUT_GRID = {
    "lon_bounds": [-98.076, 39.556],
    "lat_bounds": [10.612, 79.174],
    "n_x": 391,
    "n_y": 293,
    "z_edges_m": [0.0, 40.0, 1000.0, 5000.0],  # bottom bin matches FLEXPART surface layer
    "n_time_bins": 1,
}
MET_DOMAIN = {"lon_bounds": [-100.0, 40.0], "lat_bounds": [10.0, 80.0], "alt_max_m": 15000.0}


def _parse_header(csv_path: Path) -> dict[str, str]:
    """Parse the leading ``# key: value`` comment header of a validation CSV."""
    meta: dict[str, str] = {}
    with open(csv_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" in body:
                key, _, val = body.partition(":")
                meta[key.strip()] = val.split("#")[0].strip()  # drop trailing inline comment
    return meta


def collect_sites() -> list[dict[str, object]]:
    """One site per unique inlet label, with location from its CSV header."""
    labels = sorted(
        {p.name.split("_FLEXPART")[0].split("_NAME")[0] for p in VTS_DIR.glob("*.csv")}
    )
    if not labels:
        raise SystemExit(f"No validation CSVs found in {VTS_DIR}")
    sites: list[dict[str, object]] = []
    for label in labels:
        meta = _parse_header(next(VTS_DIR.glob(f"{label}_*.csv")))
        sites.append(
            {
                "name": label,
                "lon": float(meta["release_lon_deg"]),
                "lat": float(meta["release_lat_deg"]),
                "alt_agl_m": float(meta["release_height_magl"]),
            }
        )
    return sites


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-releases", type=int, default=48, help="hourly release times per site")
    ap.add_argument("--start-time", default="2024-01-01T00:00:00Z")
    ap.add_argument("--length-seconds", type=int, default=432000, help="backward window (default 5 days)")
    ap.add_argument("--period-seconds", type=int, default=3600)
    ap.add_argument("--duration-seconds", type=int, default=3600)
    ap.add_argument("--n-particles", type=int, default=20000, help="particles per (site, time)")
    ap.add_argument(
        "--max-releases-per-batch",
        type=int,
        default=0,
        help="0 = one batch (n_sites × n_releases): max amortisation, tiny met cache (monotonic sweep)",
    )
    ap.add_argument("--zarr-store", default="~/data/arco-era5/EUROPE_*.zarr")
    ap.add_argument("--output-uri", default="outputs/icos-validation")  # matches notebooks/multisite_validation.ipynb
    ap.add_argument("-o", "--output-config", default="configs/multisite_validation_48h.yaml")
    args = ap.parse_args()

    sites = collect_sites()
    n_sites = len(sites)
    max_per_batch = args.max_releases_per_batch or (n_sites * args.n_releases)

    cfg = {
        "io": {"zarr_store": args.zarr_store, "output_uri": args.output_uri},
        "simulation": {
            "start_time": args.start_time,
            "length_seconds": args.length_seconds,
            "dt_seconds": 60,
            "device": "auto",
        },
        "release": {
            "kind": "multi_point_periodic",
            "sites": sites,
            "start_time": args.start_time,
            "period_seconds": args.period_seconds,
            "n_releases": args.n_releases,
            "duration_seconds": args.duration_seconds,
            "n_particles_per_release": args.n_particles,
            "seed": 42,
        },
        "turbulence": {
            "scheme": "hanna_1982",
            "max_substeps": 20,
            "meander": {"enabled": True, "coefficient": 0.16},
        },
        "convection": {"scheme": "emanuel_reduced", "emanuel": {"closure_c": 0.03}},
        "output_grid": OUTPUT_GRID,
        "met_domain": MET_DOMAIN,
        # One batch (default) = all sites × all times in a single engine pass: the per-window
        # fixed costs are paid once, and a single monotonic backward sweep needs only a tiny
        # met cache (no cross-batch re-fetch). Keep max_releases_per_batch a multiple of
        # n_sites if you split into batches.
        "batch": {"max_releases_per_batch": max_per_batch},
        "memory": {
            "met_cache_max_hours": 8,
            "met_cache_on_host": True,
            "met_prefetch": True,
            "log_every_steps": 200,
            "gc_every_steps": 0,
            "guard_check_every_steps": 10,
        },
    }

    header = (
        f"# GENERATED by scripts/make_multisite_config.py — DO NOT EDIT BY HAND.\n"
        f"# {n_sites} validation sites × {args.n_releases} hourly releases "
        f"= {n_sites * args.n_releases} footprints, EUROPE domain.\n"
        f"# Regenerate: python scripts/make_multisite_config.py "
        f"--n-releases {args.n_releases} -o {args.output_config}\n"
    )
    out_path = REPO / args.output_config
    out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))

    # Sanity-check it parses + expands (also runs the in-domain / unique-name validators).
    from lpdm.config import RunConfig

    parsed = RunConfig.from_yaml(out_path)
    batches = parsed.expand_to_batches()
    total = sum(len(b.releases) for b in batches)
    print(f"wrote {out_path}")
    print(
        f"  {n_sites} sites × {args.n_releases} releases = {total} releases "
        f"in {len(batches)} batch(es) of <= {max_per_batch}"
    )
    print(
        f"  ~{total * args.n_particles / 1e6:.1f}M particles per batch; "
        f"footprint store {total}×{OUTPUT_GRID['n_time_bins']}×"
        f"{len(OUTPUT_GRID['z_edges_m']) - 1}×{OUTPUT_GRID['n_y']}×{OUTPUT_GRID['n_x']}"
    )
    print(f"  launch: sbatch scripts/run_periodic_cuda.slurm {args.output_config}")


if __name__ == "__main__":
    main()
