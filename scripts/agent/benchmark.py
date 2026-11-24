#!/usr/bin/env python
"""Benchmark a config: run it with GLIDE_PHASE_TIMERS=1 and print a Markdown report.

Usage (from the repository root, GPU environment sourced)::

    python scripts/agent/benchmark.py --config configs/ab_multisite_perf.yaml --device cuda
    # re-report a finished run from its log (no model run):
    python scripts/agent/benchmark.py --from-log <run dir>.log

Reports footprints per second (releases written / process wall time, start-up,
compilation and output writing included), the model's phase table, the process
peak RSS (``/usr/bin/time -v``) and the peak GPU memory from
``run_metadata.json``. The phase timers do not synchronise the GPU unless
``GLIDE_PHASE_TIMERS_SYNC=1``, so GPU phases are under-counted and the residual
holds the rest; compare like with like. ``--json`` also writes the numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_common as gc  # noqa: E402

_HEADER = re.compile(
    r"GLIDE phase timers\s+\S\s+total wall\s+(?P<wall>[\d.]+)s\s+\(met fetches:\s*(?P<fetches>\d+)\)"
)
_ROW = re.compile(
    r"^\s+(?P<name>\S+)\s+(?P<total>[\d.]+)\s+(?P<pct>[\d.]+)%"
    r"(?:\s+(?P<calls>\d+)\s+(?P<ms>[\d.]+))?\s*$"
)


def parse_phase_table(text: str) -> dict | None:
    """The last phase table in a GLIDE log, or None if there is none."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if _HEADER.search(line):
            start = i
    if start is None:
        return None
    m = _HEADER.search(lines[start])
    table = {
        "total_wall_s": float(m.group("wall")),
        "met_fetches": int(m.group("fetches")),
        "gpu_synced": "no CUDA sync" not in lines[start],
        "phases": [],
    }
    for line in lines[start + 1 :]:
        if line.startswith("=="):
            break
        r = _ROW.match(line)
        if not r or r.group("name") == "phase":
            continue
        table["phases"].append(
            {
                "phase": r.group("name"),
                "total_s": float(r.group("total")),
                "pct_wall": float(r.group("pct")),
                "calls": int(r.group("calls")) if r.group("calls") else None,
                "ms_per_call": float(r.group("ms")) if r.group("ms") else None,
            }
        )
    return table


def _elapsed_seconds(value: str) -> float:
    parts = [float(x) for x in value.split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def summarise(log_path: Path, output_dir: Path | None, wall_seconds: float | None) -> dict:
    text = log_path.read_text(errors="replace")
    tv = gc.parse_time_v(text)
    table = parse_phase_table(text)
    meta = {}
    if output_dir is not None and (output_dir / gc.METADATA_FILE).exists():
        meta = json.loads((output_dir / gc.METADATA_FILE).read_text())
    wall = wall_seconds
    key = "Elapsed (wall clock) time (h:mm:ss or m:ss)"
    if key in tv:
        wall = _elapsed_seconds(tv[key])
    sched = meta.get("schedule", {})
    runtime = meta.get("runtime", {})
    n_rel = sched.get("n_releases")
    out = {
        "log": str(log_path),
        "config": meta.get("config", {}).get("io", {}).get("output_uri"),
        "device": runtime.get("device"),
        "n_releases": n_rel,
        "n_batches": sched.get("n_batches"),
        "n_particles_per_release": meta.get("config", {})
        .get("release", {})
        .get("n_particles_per_release"),
        "steps": runtime.get("steps"),
        "process_wall_s": wall,
        "footprints_per_s": (n_rel / wall) if (n_rel and wall) else None,
        "max_rss_gb": int(tv["Maximum resident set size (kbytes)"]) / 1e6
        if "Maximum resident set size (kbytes)" in tv
        else None,
        "peak_device_allocated_gib": runtime.get("peak_device_allocated_bytes", 0) / 2**30
        if runtime.get("peak_device_allocated_bytes") is not None
        else None,
        "peak_device_reserved_gib": runtime.get("peak_device_reserved_bytes", 0) / 2**30
        if runtime.get("peak_device_reserved_bytes") is not None
        else None,
        "phase_table": table,
    }
    return out


def to_markdown(s: dict, title: str = "GLIDE benchmark") -> str:
    def f(v, fmt):
        return "n/a" if v is None else format(v, fmt)

    lines = [f"### {title}", ""]
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    lines.append(f"| device | {s.get('device') or 'n/a'} |")
    lines.append(
        f"| releases (footprints) | {s.get('n_releases')} in {s.get('n_batches')} batch(es), "
        f"{s.get('n_particles_per_release')} particles each |"
    )
    lines.append(f"| model steps | {s.get('steps')} |")
    lines.append(f"| process wall time | {f(s.get('process_wall_s'), '.1f')} s |")
    lines.append(f"| **footprints per second** | **{f(s.get('footprints_per_s'), '.4f')}** |")
    lines.append(f"| peak host RSS | {f(s.get('max_rss_gb'), '.1f')} GB |")
    lines.append(
        f"| peak GPU allocated / reserved | {f(s.get('peak_device_allocated_gib'), '.2f')} / "
        f"{f(s.get('peak_device_reserved_gib'), '.2f')} GiB |"
    )
    t = s.get("phase_table")
    lines.append("")
    if not t:
        lines.append("No phase table in the log (was GLIDE_PHASE_TIMERS=1 set?).")
        return "\n".join(lines) + "\n"
    sync = "GPU synchronised" if t["gpu_synced"] else "no CUDA sync: GPU phases under-counted"
    lines.append(
        f"Phase timers: total wall {t['total_wall_s']:.1f} s, "
        f"{t['met_fetches']} met fetches ({sync})."
    )
    lines.append("")
    lines.append("| phase | total (s) | % wall | calls | ms/call |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for ph in t["phases"]:
        calls = "" if ph["calls"] is None else str(ph["calls"])
        ms = "" if ph["ms_per_call"] is None else f"{ph['ms_per_call']:.3f}"
        lines.append(
            f"| {ph['phase']} | {ph['total_s']:.2f} | {ph['pct_wall']:.1f} | {calls} | {ms} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, help="run directory (default under <outputs>/bench)")
    p.add_argument("--from-log", type=Path, help="report an existing run's log instead")
    p.add_argument("--json", type=Path, help="also write the numbers as JSON")
    args = p.parse_args(argv)

    if args.from_log:
        log = args.from_log
        out_dir = log.with_name(log.name.removesuffix(".log"))
        s = summarise(log, out_dir if out_dir.is_dir() else None, None)
        title = f"GLIDE benchmark ({out_dir.name})"
    else:
        if not (args.config and args.device):
            p.error("--config and --device are required unless --from-log is given")
        out_dir = args.output or (
            gc.outputs_root()
            / "bench"
            / f"{args.config.stem}-{args.device}-{gc.utc_now().replace(':', '')}"
        )
        rec = gc.run_glide(args.config.resolve(), args.device, out_dir, phase_timers=True)
        s = summarise(rec.log_path, rec.output_dir, rec.wall_seconds)
        s["env"] = rec.env
        s.update(gc.git_info())
        title = f"GLIDE benchmark: {args.config.stem} on {args.device}"
    md = to_markdown(s, title)
    print(md)
    if args.json:
        args.json.write_text(json.dumps(s, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
