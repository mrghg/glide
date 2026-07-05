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

Batch size and met-cache size are auto-chosen. The batch target is the "active window"
(the releases whose backward windows overlap the cursor at peak = ceil(length/period) x
n_sites): on the static GPU path every step processes the whole batch buffer, so batches
bigger than the active window just waste work stepping inactive particles, while smaller
ones trade device memory for a larger (host) met cache. Pass --gpu-memory-gib to cap the
batch to your GPU's memory; the only spec that matters for sizing is total GPU (device)
memory. Everything is printed after generation.

    # short run — fits one batch on any GPU:
    python scripts/make_multisite_config.py --n-releases 48 -o configs/multisite_validation_48h.yaml
    # long run — auto-splits into active-window batches (pass your GPU memory to cap):
    python scripts/make_multisite_config.py --n-releases 720 --gpu-memory-gib 96 -o configs/multisite_720.yaml
    sbatch scripts/run_periodic_cuda.slurm configs/multisite_720.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
VTS_DIR = REPO / "data" / "validation-timeseries"

# --- Batch memory model (for --gpu-memory-gib auto-sizing) -----------------
# The batch's DEVICE working set is particles + the footprint tensor; the met
# cache lives in HOST RAM (met_cache_on_host) so it does NOT count here.
#
# Per-particle persistent device state (float32 unless noted):
#   position (N,4)=16 B, turbulence u'/v'/w' =12 B, meander u/v =8 B,
#   alive mask (bool)=1 B, per-particle release metadata ~19 B  ≈ 56 B.
# The per-step kernels allocate transient (N,) tensors on top; a ~2x multiplier
# covers that peak. This is an ESTIMATE — the --memory-headroom-frac budget and
# the runtime memory guards are the real safety net, so it only needs to be in
# the right ballpark.
_PERSISTENT_BYTES_PER_PARTICLE_BASE = 56.0     # meander on; subtract 8 if off
_WORKING_SET_MULTIPLIER = 2.0
_BYTES_PER_GIB = float(1024 ** 3)

# Host met-cache footprint: the on-host met cache holds ~this many GiB per cached
# hour on the EUROPE domain (empirical: 192 h ≈ 50 GiB). Multi-batch runs need a
# cache spanning the overlapping backward windows, so this drives HOST RAM (SLURM
# --mem), NOT device memory. ~20 GiB more is fixed run overhead (python/torch,
# particle staging, prefetch buffers).
_MET_GIB_PER_HOUR = 0.27
_HOST_RESERVE_GIB = 20.0

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


def _bytes_per_particle(meander_enabled: bool) -> float:
    base = _PERSISTENT_BYTES_PER_PARTICLE_BASE - (0.0 if meander_enabled else 8.0)
    return base * _WORKING_SET_MULTIPLIER


def _footprint_bytes_per_release() -> float:
    n_z = len(OUTPUT_GRID["z_edges_m"]) - 1
    return float(OUTPUT_GRID["n_time_bins"] * n_z * OUTPUT_GRID["n_y"] * OUTPUT_GRID["n_x"] * 4)


def _device_bytes_per_release(n_particles: int, meander_enabled: bool) -> float:
    return n_particles * _bytes_per_particle(meander_enabled) + _footprint_bytes_per_release()


def _choose_batch_size(
    *,
    total_releases: int,
    n_sites: int,
    length_seconds: int,
    period_seconds: int,
    n_particles: int,
    meander_enabled: bool,
    gpu_memory_gib: float | None,
    headroom_frac: float,
) -> tuple[int, str]:
    """Pick max_releases_per_batch (a multiple of n_sites) and explain why.

    The compute-optimal target is the ACTIVE WINDOW — the releases whose backward
    windows overlap the cursor at peak, ``ceil(length/period) * n_sites``. Making
    batches bigger than this just steps ever-more inactive particles each step (the
    static GPU path processes the whole buffer); making them smaller trades device
    memory for a larger met cache. So we target the active window, then shrink it
    only if it would exceed the GPU-memory budget.
    """

    per_release = _device_bytes_per_release(n_particles, meander_enabled)
    active_times = math.ceil(length_seconds / period_seconds)
    active_window = min(total_releases, (active_times + 1) * n_sites)  # +1 block edge slack

    if gpu_memory_gib is None:
        batch = active_window
        why = (
            f"active window ({active_times}+1 time-blocks x {n_sites} sites); "
            "pass --gpu-memory-gib to cap it to your GPU"
        )
    else:
        budget_bytes = gpu_memory_gib * headroom_frac * _BYTES_PER_GIB
        mem_cap = int(budget_bytes // per_release)
        mem_cap = (mem_cap // n_sites) * n_sites  # whole time-blocks
        if mem_cap < active_window:
            batch = max(n_sites, mem_cap)
            why = (
                f"GPU-memory-capped ({gpu_memory_gib:.0f} GiB x {headroom_frac:.0%} / "
                f"{per_release / 1e6:.2f} MB per release); below the active window "
                f"({active_window}) so a larger met cache is used"
            )
        else:
            batch = active_window
            why = f"active window ({active_window}); fits the GPU-memory budget"

    batch = max(n_sites, min(batch, total_releases))
    batch = (batch // n_sites) * n_sites or n_sites
    return batch, why


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
        help="0 = auto (target the active window, capped by --gpu-memory-gib). "
        "Set explicitly to override the auto-sizing. Kept a multiple of n_sites.",
    )
    ap.add_argument(
        "--gpu-memory-gib",
        type=float,
        default=None,
        help="Device (GPU) memory available for the run, in GiB — the ONLY spec that "
        "caps batch size. Find it with `nvidia-smi --query-gpu=memory.total --format=csv`. "
        "e.g. GH200~96, A100=40/80, H100=80, L4/RTX4090=24. Omit to target the active "
        "window and just print the memory it needs.",
    )
    ap.add_argument(
        "--memory-headroom-frac",
        type=float,
        default=0.6,
        help="Fraction of --gpu-memory-gib to use for the batch working set (rest covers "
        "CUDA context, compile workspace, fragmentation). Default 0.6.",
    )
    ap.add_argument(
        "--met-cache-max-hours",
        type=int,
        default=0,
        help="0 = auto (from batch geometry). Override to control host-RAM met-cache size.",
    )
    ap.add_argument(
        "--host-memory-gib",
        type=float,
        default=None,
        help="Host RAM budget in GiB (your SLURM --mem). The on-host met cache must fit "
        "it; if the geometry-ideal cache is larger it's CAPPED to fit (accepting some "
        "cross-batch met re-fetch) and a warning is printed. Omit to size for the geometry "
        "and just print the host RAM required.",
    )
    ap.add_argument("--zarr-store", default="~/data/arco-era5/EUROPE_*.zarr")
    ap.add_argument("--output-uri", default="outputs/icos-validation")  # matches notebooks/multisite_validation.ipynb
    ap.add_argument("-o", "--output-config", default="configs/multisite_validation_48h.yaml")
    args = ap.parse_args()

    sites = collect_sites()
    n_sites = len(sites)
    total_releases = n_sites * args.n_releases
    meander_enabled = True  # set in the turbulence block below

    # --- Batch sizing -------------------------------------------------------
    if args.max_releases_per_batch:
        max_per_batch = max(n_sites, (args.max_releases_per_batch // n_sites) * n_sites)
        batch_why = "explicit --max-releases-per-batch"
    else:
        max_per_batch, batch_why = _choose_batch_size(
            total_releases=total_releases,
            n_sites=n_sites,
            length_seconds=args.length_seconds,
            period_seconds=args.period_seconds,
            n_particles=args.n_particles,
            meander_enabled=meander_enabled,
            gpu_memory_gib=args.gpu_memory_gib,
            headroom_frac=args.memory_headroom_frac,
        )
    n_batches = math.ceil(total_releases / max_per_batch)

    # --- Met cache: single monotonic sweep needs almost nothing; multi-batch
    # runs re-read the overlapping backward windows, so size the host cache above
    # the reuse threshold (span + advance) to avoid cross-batch re-fetch thrash.
    span_hours = args.length_seconds / 3600.0
    advance_hours = (max_per_batch / n_sites) * (args.period_seconds / 3600.0)
    cache_capped = False
    if args.met_cache_max_hours:
        met_cache_hours = args.met_cache_max_hours
    elif n_batches <= 1:
        met_cache_hours = 8
    else:
        met_cache_hours = int(math.ceil(span_hours + advance_hours)) + 6

    # Cap the on-host cache to the declared host-RAM budget (SLURM --mem). A smaller
    # cache is still CORRECT — it just re-fetches the cross-batch overlap from zarr.
    if args.host_memory_gib is not None and not args.met_cache_max_hours:
        affordable_hours = int(max(8.0, (args.host_memory_gib - _HOST_RESERVE_GIB) / _MET_GIB_PER_HOUR))
        if met_cache_hours > affordable_hours:
            met_cache_hours = affordable_hours
            cache_capped = n_batches > 1

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
        # Batch size is auto-chosen to the "active window" (releases whose backward
        # windows overlap at peak) — the compute-optimal target on the static GPU
        # path (bigger batches just step more inactive particles each step). Capped
        # by --gpu-memory-gib. Kept a multiple of n_sites so each batch is whole
        # time-blocks × all sites.
        "batch": {"max_releases_per_batch": max_per_batch},
        "memory": {
            "met_cache_max_hours": met_cache_hours,
            "met_cache_on_host": True,
            "met_prefetch": True,
            "log_every_steps": 200,
            "gc_every_steps": 0,
            "guard_check_every_steps": 10,
        },
    }

    gpu_flag = f" --gpu-memory-gib {args.gpu_memory_gib:g}" if args.gpu_memory_gib else ""
    header = (
        f"# GENERATED by scripts/make_multisite_config.py — DO NOT EDIT BY HAND.\n"
        f"# {n_sites} validation sites × {args.n_releases} hourly releases "
        f"= {total_releases} footprints, EUROPE domain.\n"
        f"# Batching: {n_batches} batch(es) of <= {max_per_batch} ({batch_why}); "
        f"met cache {met_cache_hours} h.\n"
        f"# Regenerate: python scripts/make_multisite_config.py "
        f"--n-releases {args.n_releases}{gpu_flag} -o {args.output_config}\n"
    )
    out_path = REPO / args.output_config
    out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))

    # Sanity-check it parses + expands (also runs the in-domain / unique-name validators).
    from lpdm.config import RunConfig

    parsed = RunConfig.from_yaml(out_path)
    batches = parsed.expand_to_batches()
    total = sum(len(b.releases) for b in batches)

    # Sizing report.
    per_release_mb = _device_bytes_per_release(args.n_particles, meander_enabled) / 1e6
    batch_device_gib = max_per_batch * per_release_mb / 1024.0
    host_cache_gib = met_cache_hours * _MET_GIB_PER_HOUR
    host_total_gib = host_cache_gib + _HOST_RESERVE_GIB  # cache + run overhead; drives SLURM --mem

    print(f"wrote {out_path}")
    print(
        f"  {n_sites} sites x {args.n_releases} releases = {total} releases "
        f"in {len(batches)} batch(es) of <= {max_per_batch}  [{batch_why}]"
    )
    print(
        f"  DEVICE (GPU): ~{batch_device_gib:.1f} GiB/batch "
        f"(~{max_per_batch * args.n_particles / 1e6:.0f}M particles + footprint {total}-release "
        f"store {OUTPUT_GRID['n_time_bins']}x{len(OUTPUT_GRID['z_edges_m']) - 1}x"
        f"{OUTPUT_GRID['n_y']}x{OUTPUT_GRID['n_x']})"
    )
    print(
        f"  HOST (RAM / SLURM --mem): ~{host_total_gib:.0f} GiB "
        f"(met cache {met_cache_hours} h ~{host_cache_gib:.0f} GiB + ~{_HOST_RESERVE_GIB:.0f} GiB overhead"
        + ("" if n_batches > 1 else "; single monotonic sweep") + ")"
    )
    if cache_capped:
        print(
            f"  WARNING: met cache capped to {met_cache_hours} h to fit --host-memory-gib="
            f"{args.host_memory_gib:.0f}; ideal is {int(math.ceil(span_hours + advance_hours)) + 6} h. "
            "The run is still correct but re-fetches the cross-batch met overlap (slower I/O). "
            "Raise --mem / --host-memory-gib, or use smaller --max-releases-per-batch."
        )
    if args.gpu_memory_gib is None:
        print(
            f"  NOTE: batch sized for the active window assuming it fits GPU memory "
            f"(~{batch_device_gib:.1f} GiB). Set --gpu-memory-gib if your GPU is smaller."
        )
    print(f"  -> set SLURM --mem >= {math.ceil(host_total_gib / 16) * 16} GiB")
    print(f"  launch: sbatch scripts/run_periodic_cuda.slurm {args.output_config}")


if __name__ == "__main__":
    main()
