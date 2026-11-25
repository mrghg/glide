"""Per-step cost of the Hanna step before/after the sub-step fix (CPU proxy).

Usage: PYTHONPATH=<src to measure> cost.py <label> [n_particles] [max_substeps]

Runs the neutral synthetic window of proto.py (h 1,500 m, all particles in
the BL, so k = ceil(60/(0.5*30)) = 4 at dt=60 s) on three paths:
  dynamic : CPU path (masked, variable count)
  static  : static path, eager, variable count (max_k)
  graph   : static path with the fixed graph trip count (n_substeps =
            max_substeps), run eagerly -- the work a CUDA-graph replay does
and reports wall time per step and the number of aten ops dispatched per step
(torch.profiler), a proxy for the kernels a captured graph contains. Appends a
JSON line to $WM_OUT/cost.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(int(os.environ.get("WM_THREADS", "8")))

from proto import _hanna_met_window  # noqa: E402

from lpdm.gpu_engine import GPUEngine  # noqa: E402
from lpdm.turbulence import HannaScheme  # noqa: E402

OUT = os.environ.get("WM_OUT", "outputs/agents/WM")


def bench(path: str, n: int, max_substeps: int, steps: int = 20):
    torch.manual_seed(0)
    engine = GPUEngine(device="cpu")
    scheme = HannaScheme(
        meander_enabled=False,
        static_substeps=(path != "dynamic"),
        max_substeps=max_substeps,
        substep_c=float(os.environ.get("WM_C", "0.5")),
    )
    if path == "graph":
        scheme._maybe_graph_compile = lambda engine: scheme._step_core
    met = _hanna_met_window(blh_m=1500.0, ustar=0.4, shf=0.0)
    p = torch.zeros(n, 4)
    p[:, 1] = 45.0
    p[:, 2] = torch.rand(n) * 1500.0
    p[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    def one():
        nonlocal p, state
        p, state = scheme.step(p, state, met, 0.5, 60.0, active, engine)

    for _ in range(3):
        one()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    per_step = (time.perf_counter() - t0) / steps
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        one()
    n_ops = sum(
        e.count for e in prof.key_averages() if e.key.startswith("aten::") and "empty" not in e.key
    )
    return per_step, n_ops


if __name__ == "__main__":
    label = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    for path in ("dynamic", "static", "graph"):
        per_step, n_ops = bench(path, n, ms)
        rec = dict(label=label, path=path, n=n, max_substeps=ms, substep_c=os.environ.get("WM_C", "0.5"),
                   ms_per_step=round(per_step * 1e3, 2), aten_ops_per_step=n_ops)  # fmt: skip
        print(json.dumps(rec))
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "cost.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
