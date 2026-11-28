#!/usr/bin/env python
"""Task A4: time the wind gather alone on the GPU, to separate the effect of the
16-bit volume from that of the gather implementation.

    python scripts/validation/a4/gather_microbench.py [--n 1440000] [--repeats 200]

Volume: two [3, 40, 274, 551] wind stacks (start and end of a window, the
EUROPE domain on the 40-level AGL grid), random values; N random particle
coordinates in [-1, 1]^3. Times, with CUDA events, one RK2 evaluation's worth of
sampling (both stacks at N points) for:

- grid_sample on float32 stacks (main's and A4's default path);
- trilinear_sample on float32 stacks;
- trilinear_sample on bfloat16 stacks (simulation.wind_gather_dtype: bfloat16);

each eager and under torch.compile (the runtime compiles the whole step).
Run under the shared GPU lock.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from lpdm.gpu_engine import trilinear_sample  # noqa: E402


def grid_sample_pair(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor):
    grid = g.view(1, 1, 1, -1, 3)
    return [
        torch.nn.functional.grid_sample(v.unsqueeze(0), grid, align_corners=True).view(3, -1)
        for v in (a, b)
    ]


def trilinear_pair(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor):
    return trilinear_sample((a, b), g)


def time_it(fn, *args, repeats: int) -> float:
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_440_000)
    p.add_argument("--repeats", type=int, default=200)
    args = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn(3, 40, 274, 551, device=dev)
    b = torch.randn(3, 40, 274, 551, device=dev)
    g = torch.rand(args.n, 3, device=dev) * 2.0 - 1.0
    ah, bh = a.to(torch.bfloat16), b.to(torch.bfloat16)
    cases = [
        ("grid_sample float32", grid_sample_pair, a, b),
        ("trilinear_sample float32", trilinear_pair, a, b),
        ("trilinear_sample bfloat16", trilinear_pair, ah, bh),
    ]
    print(f"N = {args.n} particles, two [3, 40, 274, 551] stacks; ms per (start + end) sample")
    print("| gather | eager (ms) | compiled (ms) |")
    print("| --- | ---: | ---: |")
    for name, fn, x, y in cases:
        eager = time_it(fn, x, y, g, repeats=args.repeats)
        comp = torch.compile(fn, dynamic=False)
        compiled = time_it(comp, x, y, g, repeats=args.repeats)
        print(f"| {name} | {eager:.3f} | {compiled:.3f} |", flush=True)


if __name__ == "__main__":
    main()
