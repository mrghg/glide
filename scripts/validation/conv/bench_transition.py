"""Task CONV: cost of the transition matrices per convection event, by application.

    PYTHONPATH=<worktree>/src .venv/bin/python scripts/validation/conv/bench_transition.py --device cpu

Synthetic columns (U occupied firing columns x Z levels, the sizes an event
meets), the default closure's M_b; times `transition_matrices` (backward) for
`cfl_scaled`, `exponential` and `substeps` (n = 4) after a warm-up, median of
repeats. On CUDA it synchronises around each call and also reports whether the
exponential path issues any device-to-host copy (torch profiler, memcpy DtoH).
"""

from __future__ import annotations

import argparse
import time

import torch

import lpdm
from lpdm.convection import emanuel as em


def columns(u: int, z: int, device: torch.device) -> em.ColumnConvection:
    p = torch.linspace(101000.0, 15000.0, z, device=device).expand(u, z).contiguous()
    g = torch.Generator(device="cpu").manual_seed(0)
    lcl = torch.randint(z // 6, z // 3, (u,), generator=g).to(device)
    lnb = (lcl + torch.randint(z // 6, z // 2, (u,), generator=g).to(device)).clamp(max=z - 1)
    zeros = torch.zeros(u, device=device)
    return em.ColumnConvection(
        t=p,
        q=p,
        pressure_pa=p,
        height_m=p,
        ascending=True,
        i_lcl=lcl,
        i_lnb=lnb,
        cape=zeros,
        dtv_lcl_plus_1=zeros,
        cloud_depth_m=zeros,
        fires=torch.ones(u, dtype=torch.bool, device=device),
        m_b=torch.full((u,), 0.17, dtype=torch.float64, device=device),
        interval_s=3600.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    print(f"lpdm: {lpdm.__file__}; device {dev}; threads {args.threads}", flush=True)
    for u, z in ((256, 40), (2048, 40), (2048, 80), (2048, 137)):
        cols = columns(u, z, dev)
        idx = torch.arange(u, device=dev)
        row = []
        for app in ("cfl_scaled", "exponential", "substeps"):
            s = em.EmanuelReducedConvection(application=app)
            s.transition_matrices(cols, idx)
            ts = []
            for _ in range(args.repeats):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                tic = time.perf_counter()
                s.transition_matrices(cols, idx)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - tic)
            row.append(f"{app} {sorted(ts)[len(ts) // 2] * 1e3:8.1f} ms")
        print(f"U={u:5d} Z={z:4d}: " + " | ".join(row), flush=True)
    if dev.type == "cuda":
        cols = columns(512, 40, dev)
        idx = torch.arange(512, device=dev)
        s = em.EmanuelReducedConvection(application="exponential")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            s.transition_matrices(cols, idx)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events()]
        dtoh = [n for n in names if "DtoH" in n or "Memcpy DtoH" in n]
        print(f"exponential path: {len(dtoh)} device-to-host copies ({sorted(set(dtoh))})")


if __name__ == "__main__":
    main()
