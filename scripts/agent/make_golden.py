#!/usr/bin/env python
"""Make a golden fixture: run a config with its fixed seed and freeze the outputs.

Usage (from the repository root, GPU environment sourced for ``--device cuda``)::

    python scripts/agent/make_golden.py --config configs/smoke_mhd_single_release.yaml \
        --device cpu
    python scripts/agent/make_golden.py --config configs/ab_multisite_perf.yaml \
        --device cuda --large
    # reuse a run already made at this rev (e.g. by benchmark.py):
    python scripts/agent/make_golden.py --config configs/ab_multisite_perf.yaml \
        --device cuda --large --from-output <run dir>

The fixture goes to ``tests/golden/<config-stem>/<device>/``. With ``--large`` only
``manifest.json`` goes there and the data (footprint, endpoint particles, run
metadata) go to ``<outputs>/golden/<config-stem>/<device>/``, where ``<outputs>``
is ``$GLIDE_AGENT_OUTPUTS`` or the main checkout's ``outputs/``. The seed is the
config's ``release.seed``; the manifest records it with the git rev, the config
hashes, the device, torch/CUDA versions, the GPU, the met stores' provenance and
SHA-256 checksums of the data files. See docs/VALIDATION.md, "Golden fixtures".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_common as gc  # noqa: E402


def build_fixture(
    *,
    config: Path,
    device: str,
    run_dir: Path,
    fixture_dir: Path,
    data_dir: Path,
    run_record: gc.RunRecord | None,
    overwrite: bool,
) -> dict:
    run_dir = run_dir.resolve()
    meta = json.loads((run_dir / gc.METADATA_FILE).read_text())
    if meta.get("runtime", {}).get("status") not in (None, "complete", "completed", "ok"):
        raise SystemExit(f"run status is {meta['runtime'].get('status')!r}, not complete")
    run_device = meta.get("runtime", {}).get("device")
    if run_device and not str(run_device).startswith(device):
        raise SystemExit(f"run was on {run_device!r}, fixture asked for {device!r}")

    for d in {fixture_dir, data_dir}:
        exists = (d / gc.MANIFEST_FILE).exists() or (d / gc.FOOTPRINT_FILE).exists()
        if exists and not overwrite:
            raise SystemExit(f"fixture exists in {d}; pass --overwrite to replace it")
        d.mkdir(parents=True, exist_ok=True)

    footprint, release_time = gc.read_run_footprint(run_dir)
    np.savez_compressed(
        data_dir / gc.FOOTPRINT_FILE, footprint=footprint, release_time=release_time
    )
    shutil.copyfile(run_dir / gc.ENDPOINT_FILE, data_dir / gc.ENDPOINT_FILE)
    # The copy of run_metadata.json names the run's files by portable locations,
    # so no machine-specific path is committed with a small fixture.
    meta_copy = json.loads(json.dumps(meta))
    for key, value in meta_copy.get("outputs", {}).items():
        if isinstance(value, str) and value.startswith("/"):
            meta_copy["outputs"][key] = gc.portable_location(Path(value))
    io = meta_copy.get("config", {}).get("io", {})
    if isinstance(io.get("output_uri"), str) and io["output_uri"].startswith("/"):
        io["output_uri"] = gc.portable_location(Path(io["output_uri"]))
    (data_dir / gc.METADATA_FILE).write_text(json.dumps(meta_copy, indent=2) + "\n")

    per_release = footprint.reshape(footprint.shape[0], -1).astype(np.float64).sum(axis=1)
    files = {
        name: {
            "sha256": gc.sha256_file(data_dir / name),
            "bytes": (data_dir / name).stat().st_size,
        }
        for name in (gc.FOOTPRINT_FILE, gc.ENDPOINT_FILE, gc.METADATA_FILE)
    }
    manifest: dict = {
        "fixture": f"{config.stem}/{device}",
        "created_utc": gc.utc_now(),
        "config": gc.portable_location(config),
        **gc.config_hashes(config, meta.get("config")),
        "device": device,
        "runtime_device": run_device,
        "seed": meta.get("config", {}).get("release", {}).get("seed"),
        **gc.git_info(),
        **gc.torch_info(gpu=device == "cuda"),
        "python": sys.version.split()[0],
        "data_location": "." if data_dir == fixture_dir else gc.portable_location(data_dir),
        "files": files,
        "footprint": {
            "shape": list(footprint.shape),
            "dims": ["release", "time_ago", "z_bin", "latitude", "longitude"],
            "dtype": str(footprint.dtype),
            "nonzero_cells": int(np.count_nonzero(footprint)),
            "total": float(per_release.sum()),
            "per_release_total_min": float(per_release.min()),
            "per_release_total_max": float(per_release.max()),
            "units": meta.get("footprint_units", {}).get("units"),
            "density_weighting": meta.get("footprint_units", {}).get("density_weighting"),
        },
        "n_releases": int(footprint.shape[0]),
        "n_particles_per_release": meta.get("config", {})
        .get("release", {})
        .get("n_particles_per_release"),
        "met_stores": gc.met_provenance(meta),
    }
    if run_record is not None:
        manifest["run"] = {
            "wall_seconds": round(run_record.wall_seconds, 1),
            "max_rss_kb": run_record.max_rss_kb,
            "env": run_record.env,
        }
    else:
        manifest["run"] = {
            "from_output": gc.portable_location(run_dir),
            "env_when_fixed": {k: os.environ[k] for k in gc.RECORDED_ENV if k in os.environ},
        }
    peak = meta.get("runtime", {})
    for key in ("peak_device_allocated_bytes", "peak_device_reserved_bytes", "steps"):
        if key in peak:
            manifest["run"][key] = peak[key]

    text = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    (fixture_dir / gc.MANIFEST_FILE).write_text(text)
    if data_dir != fixture_dir:
        (data_dir / gc.MANIFEST_FILE).write_text(text)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument(
        "--large",
        action="store_true",
        help="keep the data under the outputs directory; commit only the manifest",
    )
    p.add_argument("--fixture-dir", type=Path, help="default tests/golden/<stem>/<device>")
    p.add_argument("--data-dir", type=Path, help="with --large: default <outputs>/golden/...")
    p.add_argument(
        "--run-dir",
        type=Path,
        help="where the model writes its run (default <outputs>/golden-runs/<stem>-<device>-<utc>)",
    )
    p.add_argument(
        "--from-output",
        type=Path,
        help="build from an existing run directory made at this rev instead of running",
    )
    p.add_argument("--phase-timers", action="store_true", help="set GLIDE_PHASE_TIMERS=1")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    config = args.config.resolve()
    stem = config.stem
    fixture_dir = (args.fixture_dir or gc.GOLDEN_ROOT / stem / args.device).resolve()
    if args.large:
        data_dir = (args.data_dir or gc.outputs_root() / "golden" / stem / args.device).resolve()
    else:
        data_dir = fixture_dir

    record = None
    if args.from_output:
        run_dir = args.from_output
    else:
        run_dir = args.run_dir or (
            gc.outputs_root()
            / "golden-runs"
            / f"{stem}-{args.device}-{gc.utc_now().replace(':', '')}"
        )
        record = gc.run_glide(config, args.device, run_dir, phase_timers=args.phase_timers)

    m = build_fixture(
        config=config,
        device=args.device,
        run_dir=run_dir,
        fixture_dir=fixture_dir,
        data_dir=data_dir,
        run_record=record,
        overwrite=args.overwrite,
    )
    size = sum(f["bytes"] for f in m["files"].values())
    print(f"fixture {m['fixture']} at {fixture_dir}")
    print(f"  data in {data_dir} ({size / 1e6:.2f} MB)")
    print(f"  git {m['git_rev'][:12]} (src dirty: {m['src_dirty']}), torch {m['torch_version']}")
    print(f"  footprint {m['footprint']['shape']} total {m['footprint']['total']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
