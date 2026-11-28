#!/usr/bin/env python
"""Task A4, item 3: where a met window fetch spends its time, and whether
independent windows can be fetched concurrently.

    python scripts/validation/a4/profile_met_fetch.py --config configs/ab_multisite_perf.yaml \
        [--hours 6] [--workers 1 2 4]

Builds the reader as the runtime does (same channels and AGL grid), then:

1. fetches ``--hours`` consecutive windows sequentially, walking backward like the
   runtime (each new window processes one new hour; the other comes from the
   reader's hour cache), and reports wall and process CPU time per window, with
   cProfile's top functions (cumulative) for the whole sequence;
2. for each worker count, fetches the same number of *other* windows (so no
   hour is cached) with that many threads, each worker taking every k-th
   window, and reports the throughput (windows per second).

Reads only the met stores the config names. CPU only; no GPU.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from lpdm.config import RunConfig  # noqa: E402
from lpdm.convection import get_scheme as get_convection_scheme  # noqa: E402
from lpdm.main import (  # noqa: E402
    _convection_scheme_kwargs,
    _density_weighting_met_keys,
    _met_domain_bounds,
    _met_interval,
    _scheme_kwargs,
)
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, TimeBounds  # noqa: E402
from lpdm.turbulence import get_scheme  # noqa: E402


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def make_reader(cfg):
    scheme = get_scheme(cfg.turbulence.scheme, **_scheme_kwargs(cfg))
    conv = get_convection_scheme(cfg.convection.scheme, **_convection_scheme_kwargs(cfg))
    channels = tuple(
        dict.fromkeys(
            (
                "u",
                "v",
                "w",
                "blh",
                "sp",
                *scheme.required_met_keys(),
                *conv.required_met_keys(),
                *_density_weighting_met_keys(cfg),
            )
        )
    )
    return ArcoEra5ZarrReader(
        zarr_store=cfg.io.zarr_store,
        channel_names=channels,
        device="cpu",
        agl_levels_m=cfg.met_domain.resolve_agl_levels(),
    )


def fetch(reader, cfg, start):
    return reader.fetch_hourly_window(
        BoundingBoxRequest(
            spatial=_met_domain_bounds(cfg),
            time=TimeBounds(start=start, end=start + _met_interval(reader)),
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--hours", type=int, default=6)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    args = p.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    reader = make_reader(cfg)
    step = _met_interval(reader)
    t_top = reader.floor_to_window_start(cfg.simulation.start_time) - step

    # 1. sequential, backward, as the runtime walks
    prof = cProfile.Profile()
    rows = []
    for i in range(args.hours):
        t = t_top - i * step
        w0, c0 = time.perf_counter(), cpu_s()
        prof.enable()
        win = fetch(reader, cfg, t)
        prof.disable()
        rows.append((t, time.perf_counter() - w0, cpu_s() - c0))
    shape = tuple(win.hour_start.shape)
    print(f"window tensor {shape} float32 = {win.hour_start.numel() * 4 / 1e6:.0f} MB per hour")
    print("sequential backward fetches (window start, wall s, process CPU s):")
    for t, w, c in rows:
        print(f"  {t:%Y-%m-%d %H:%M}  {w:6.2f}  {c:6.2f}")
    steady = rows[1:]
    print(
        f"steady state (one new hour per window): wall {sum(r[1] for r in steady) / len(steady):.2f} s, "
        f"CPU {sum(r[2] for r in steady) / len(steady):.2f} s per window"
    )
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())

    # 2. concurrent fetches of uncached windows
    base = t_top - (args.hours + 2) * step
    offset = 0
    for k in args.workers:
        reader_k = make_reader(cfg)
        n = 2 * k if k > 1 else 2
        starts = [base - (offset + 2 * j) * step for j in range(n)]  # every other window
        offset += 2 * n + 2
        w0, c0 = time.perf_counter(), cpu_s()
        with ThreadPoolExecutor(max_workers=k) as ex:
            list(ex.map(lambda t, r=reader_k: fetch(r, cfg, t), starts))
        wall, cpu = time.perf_counter() - w0, cpu_s() - c0
        print(
            f"workers={k}: {n} windows (2 new hours each) in {wall:.1f} s wall, {cpu:.1f} s CPU "
            f"-> {n / wall:.3f} windows/s, {2 * n / wall:.3f} hours/s"
        )


if __name__ == "__main__":
    main()
