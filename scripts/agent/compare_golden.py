#!/usr/bin/env python
"""Compare a fresh GLIDE run (or an existing run directory) with a golden fixture.

Usage (from the repository root, GPU environment sourced for a CUDA fixture)::

    python scripts/agent/compare_golden.py tests/golden/smoke_mhd_single_release/cpu
    python scripts/agent/compare_golden.py tests/golden/ab_multisite_perf/cuda
    python scripts/agent/compare_golden.py tests/golden/smoke_mhd_single_release/cuda \
        --output <existing run dir>

Without ``--output`` it runs the fixture's config on the fixture's device, at the
current checkout, into ``<outputs>/golden-compare/``. For every release it prints
the maximum absolute and relative differences of the footprint and the
difference of the integrated sensitivity (the release's footprint total), then
pass/fail. Exit status 0 on pass, 1 on fail.

Default criteria (``docs/VALIDATION.md``, "Golden fixtures"; QUEUE.md A0):

* CPU fixtures: bit-identical footprint and endpoint particles.
* CUDA fixtures: per footprint cell |a - b| <= rtol |b| + atol_frac max|b|
  (rtol 1e-5, atol_frac 1e-7, max|b| over the release), and each release's
  total within total_rtol = 1e-6 relative. CUDA ``scatter_add_`` sums the
  dry-air-weighted contributions in a varying order, so same-seed runs differ
  in the last bits (decided 24 Nov 2026). Endpoint particles are reported but
  not judged unless ``--endpoints exact``.

``--exact`` or ``--rtol/--atol-frac/--total-rtol`` override the defaults. The
largest value of each criterion is always printed, so a drift towards the
limit is visible before it fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_common as gc  # noqa: E402

CUDA_DEFAULTS = {"rtol": 1e-5, "atol_frac": 1e-7, "total_rtol": 1e-6}


@dataclass
class Tolerance:
    exact: bool
    rtol: float = 0.0
    atol_frac: float = 0.0
    total_rtol: float = 0.0

    def describe(self) -> str:
        if self.exact:
            return "bit-identical"
        return (
            f"|a-b| <= {self.rtol:g}*|b| + {self.atol_frac:g}*max|b| per cell; "
            f"release totals within {self.total_rtol:g} relative"
        )


def compare_footprints(a: np.ndarray, b: np.ndarray, tol: Tolerance) -> dict:
    """Compare fresh ``a`` with fixture ``b``, both [release, ...]. Pure numpy."""
    if a.shape != b.shape:
        return {"pass": False, "error": f"shape {a.shape} != fixture {b.shape}", "releases": []}
    n = a.shape[0]
    a2 = a.reshape(n, -1)
    b2 = b.reshape(n, -1)
    rows = []
    for r in range(n):
        ar = a2[r].astype(np.float64)
        br = b2[r].astype(np.float64)
        diff = np.abs(ar - br)
        absb = np.abs(br)
        maxb = float(absb.max()) if absb.size else 0.0
        nz = absb > 0
        max_rel = float((diff[nz] / absb[nz]).max()) if nz.any() else 0.0
        new_cells = int(np.count_nonzero((~nz) & (ar != 0)))
        ta, tb = float(ar.sum()), float(br.sum())
        total_rel = abs(ta - tb) / abs(tb) if tb != 0 else (0.0 if ta == 0 else np.inf)
        bound = tol.rtol * absb + tol.atol_frac * maxb
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bound > 0, diff / bound, np.where(diff > 0, np.inf, 0.0))
        cell_ratio = float(ratio.max()) if ratio.size else 0.0
        bitwise_equal = bool(
            np.array_equal(a2[r].view(np.uint8), b2[r].view(np.uint8))
            if a2.dtype == b2.dtype
            else False
        )
        n_diff = int(np.count_nonzero(diff))
        if tol.exact:
            ok = bitwise_equal
        else:
            ok = cell_ratio <= 1.0 and total_rel <= tol.total_rtol
        rows.append(
            {
                "release": r,
                "max_abs": float(diff.max()) if diff.size else 0.0,
                "max_rel": max_rel,
                "max_abs_b": maxb,
                "cells_differing": n_diff,
                "cells_new_nonzero": new_cells,
                "total": ta,
                "total_fixture": tb,
                "total_rel": total_rel,
                "cell_ratio": cell_ratio,
                "bitwise_equal": bitwise_equal,
                "pass": ok,
            }
        )
    worst = {
        "max_abs": max(x["max_abs"] for x in rows),
        "max_rel": max(x["max_rel"] for x in rows),
        "total_rel": max(x["total_rel"] for x in rows),
        "cell_ratio": max(x["cell_ratio"] for x in rows),
        "cells_differing": sum(x["cells_differing"] for x in rows),
        "releases_bitwise_equal": sum(x["bitwise_equal"] for x in rows),
    }
    return {"pass": all(x["pass"] for x in rows), "releases": rows, "worst": worst}


def compare_endpoints(fresh: Path, fixture: Path) -> dict:
    import pandas as pd

    a = pd.read_parquet(fresh)
    b = pd.read_parquet(fixture)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return {
            "equal": False,
            "error": f"columns/rows {list(a.columns)}x{len(a)} != {list(b.columns)}x{len(b)}",
        }
    cols = {}
    for c in a.columns:
        av, bv = a[c].to_numpy(), b[c].to_numpy()
        if np.issubdtype(av.dtype, np.number):
            d = np.abs(av.astype(np.float64) - bv.astype(np.float64))
            cols[c] = {"max_abs": float(np.nanmax(d)) if d.size else 0.0}
        cols[c] = {**cols.get(c, {}), "equal": bool(np.array_equal(av, bv))}
    return {"equal": all(v["equal"] for v in cols.values()), "columns": cols}


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def report(result: dict, tol: Tolerance, max_rows: int) -> str:
    out = []
    if "error" in result:
        return f"FOOTPRINT ERROR: {result['error']}"
    rows = result["releases"]
    shown = rows
    if len(rows) > max_rows:
        key = "cell_ratio" if not tol.exact else "max_abs"
        shown = sorted(rows, key=lambda x: (x[key], x["total_rel"]), reverse=True)[:max_rows]
        out.append(f"({len(rows)} releases; showing the {max_rows} worst by {key})")
    hdr = (
        f"{'release':>7} {'max_abs':>10} {'max_rel':>10} {'total':>12} "
        f"{'total_rel':>10} {'cell/tol':>9} {'n_diff':>8} {'result':>6}"
    )
    out.append(hdr)
    for x in shown:
        cr = "-" if tol.exact else f"{x['cell_ratio']:.3f}"
        out.append(
            f"{x['release']:>7} {_fmt(x['max_abs']):>10} {_fmt(x['max_rel']):>10} "
            f"{x['total']:>12.6g} {_fmt(x['total_rel']):>10} {cr:>9} "
            f"{x['cells_differing']:>8} {'pass' if x['pass'] else 'FAIL':>6}"
        )
    w = result["worst"]
    out.append("")
    out.append("largest over all releases:")
    out.append(f"  max |a-b|                       {_fmt(w['max_abs'])}")
    out.append(f"  max |a-b|/|b| (cells b != 0)    {_fmt(w['max_rel'])}")
    out.append(
        f"  max release-total relative diff {_fmt(w['total_rel'])}"
        + ("" if tol.exact else f"   (limit {tol.total_rtol:g})")
    )
    if not tol.exact:
        out.append(f"  max cell |a-b| / bound          {w['cell_ratio']:.4f}   (limit 1)")
    out.append(
        f"  cells differing                 {w['cells_differing']}; "
        f"releases bit-identical {w['releases_bitwise_equal']}/{len(rows)}"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("fixture", type=Path, help="fixture directory (holds manifest.json)")
    p.add_argument("--output", type=Path, help="compare this run directory; no new run")
    p.add_argument("--fixture-data", type=Path, help="override the fixture's data location")
    p.add_argument("--run-dir", type=Path, help="where to write the fresh run")
    p.add_argument("--exact", action="store_true", help="require bit-identical output")
    p.add_argument("--rtol", type=float)
    p.add_argument("--atol-frac", type=float)
    p.add_argument("--total-rtol", type=float)
    p.add_argument(
        "--endpoints",
        choices=["auto", "exact", "report"],
        default="auto",
        help="judge endpoint particles (auto: exact for bit-identical comparisons)",
    )
    p.add_argument("--max-rows", type=int, default=20)
    p.add_argument("--json", type=Path, help="also write the full result as JSON")
    args = p.parse_args(argv)

    fixture_dir = args.fixture.resolve()
    manifest = json.loads((fixture_dir / gc.MANIFEST_FILE).read_text())
    data_dir = gc.fixture_data_dir(fixture_dir, manifest, args.fixture_data)
    device = manifest["device"]

    for name, info in manifest["files"].items():
        path = data_dir / name
        if not path.exists():
            raise SystemExit(f"fixture data missing: {path}")
        if gc.sha256_file(path) != info["sha256"]:
            raise SystemExit(f"fixture data corrupt (sha256 mismatch): {path}")

    overrides = {
        k: v
        for k, v in {
            "rtol": args.rtol,
            "atol_frac": args.atol_frac,
            "total_rtol": args.total_rtol,
        }.items()
        if v is not None
    }
    if args.exact or (device == "cpu" and not overrides):
        tol = Tolerance(exact=True)
    else:
        tol = Tolerance(exact=False, **{**CUDA_DEFAULTS, **overrides})
    endpoints_mode = args.endpoints
    if endpoints_mode == "auto":
        endpoints_mode = "exact" if tol.exact else "report"

    config = gc.resolve_location(manifest["config"])
    if args.output:
        run_dir = args.output.resolve()
    else:
        run_dir = args.run_dir or (
            gc.outputs_root()
            / "golden-compare"
            / f"{config.stem}-{device}-{gc.utc_now().replace(':', '')}"
        )
        gc.run_glide(config, device, run_dir)

    meta = json.loads((run_dir / gc.METADATA_FILE).read_text())
    hashes = gc.config_hashes(config, meta.get("config"))
    git = gc.git_info()
    tinfo = gc.torch_info(gpu=device == "cuda")
    print(f"fixture   {manifest['fixture']}  ({fixture_dir})")
    print(f"  data    {data_dir}")
    print(
        f"  made at {manifest['git_rev'][:12]} on {manifest['device']}, torch "
        f"{manifest['torch_version']}"
        + (f", {manifest['gpu_name']}" if manifest.get("gpu_name") else "")
    )
    print(f"fresh run {run_dir}")
    print(
        f"  at      {git['git_rev'][:12]} (src dirty: {git['src_dirty']}), torch "
        f"{tinfo['torch_version']}" + (f", {tinfo['gpu_name']}" if tinfo.get("gpu_name") else "")
    )
    notes = []
    for key in ("config_file_sha256", "resolved_config_sha256"):
        if key in manifest and key in hashes and manifest[key] != hashes[key]:
            notes.append(f"{key} differs from the fixture's")
    if manifest.get("torch_version") != tinfo["torch_version"]:
        notes.append("torch version differs")
    if manifest.get("met_stores") != gc.met_provenance(meta):
        notes.append("met store provenance differs")
    run_device = str(meta.get("runtime", {}).get("device", ""))
    if not run_device.startswith(device):
        notes.append(f"run device {run_device!r} is not the fixture's {device!r}")
    for n in notes:
        print(f"  NOTE: {n}")
    print(f"tolerance {tol.describe()}")
    print()

    a, rt_a = gc.read_run_footprint(run_dir)
    b, rt_b = gc.read_fixture_footprint(data_dir)
    result = compare_footprints(a, b, tol)
    if "error" not in result and not np.array_equal(rt_a, rt_b):
        result["pass"] = False
        result["error_release_time"] = "release times differ"
        print("FAIL: release times differ from the fixture")
    print(report(result, tol, args.max_rows))

    ep = compare_endpoints(run_dir / gc.ENDPOINT_FILE, data_dir / gc.ENDPOINT_FILE)
    print()
    if "error" in ep:
        print(f"endpoint particles: {ep['error']}")
    else:
        cols = ", ".join(
            f"{c} max|d| {_fmt(v['max_abs'])}" if "max_abs" in v else f"{c} equal={v['equal']}"
            for c, v in ep["columns"].items()
        )
        print(f"endpoint particles: {'identical' if ep['equal'] else 'differ'} ({cols})")
        print(f"  judged: {endpoints_mode}")
    ok = bool(result["pass"])
    if endpoints_mode == "exact":
        ok = ok and bool(ep.get("equal"))
    print()
    print(f"RESULT: {'PASS' if ok else 'FAIL'} ({tol.describe()})")
    if args.json:
        args.json.write_text(
            json.dumps(
                {"pass": ok, "tolerance": tol.__dict__, "footprint": result, "endpoints": ep},
                indent=2,
            )
            + "\n"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
