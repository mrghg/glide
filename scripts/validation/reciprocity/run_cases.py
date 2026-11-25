"""Run the reciprocity cases and their companions and save the results.

    .venv/bin/python scripts/validation/reciprocity/run_cases.py OUT_DIR \
        [--cases NAME ...] [--teeth VARIANT ...] [--n-substeps N] [--seed S]

Each case of ``tests/test_reciprocity.py::CASES`` is run through the test's own
procedure (``reciprocity``); each companion (``TEETH_CASES``) with its mutation.
For every run the script writes ``<name>.json`` (the summary the test asserts on,
the case, the timings) and ``<name>.npz`` (both maps, the tested cells and the
95% noise ranges) to OUT_DIR, and prints one line per run. CPU only; about
4 minutes per case at the test's counts on 16 threads (more on a loaded node).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))

import test_reciprocity as tr  # noqa: E402


def _save(out: Path, name: str, res: tr.Result, extra: dict) -> None:
    summary = {**res.summary(), "passes": tr._passes(res), **extra}
    summary["case"] = dataclasses.asdict(res.case)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    np.savez_compressed(
        out / f"{name}.npz", backward=res.backward, forward=res.forward, tested=res.tested,
        lo=res.lo, hi=res.hi, sum_lo=res.sum_lo, sum_hi=res.sum_hi,
        z_edges=np.asarray(res.case.grid.z_edges),
    )  # fmt: skip
    print(name, json.dumps({k: summary[k] for k in ("frac_agree", "sum_ratio", "sum_range_rel",
                                                    "n_tested", "passes")}), flush=True)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cases", nargs="*", default=list(tr.CASES))
    ap.add_argument("--teeth", nargs="*", default=list(tr.TEETH_CASES))
    ap.add_argument("--n-substeps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--figure", type=Path, default=None,
                    help="also draw the Tier 0 reciprocity figure into this directory "
                    "(reuses this process's runs; default counts and seed only)")  # fmt: skip
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)

    def case(name: str) -> tr.Case:
        c = dataclasses.replace(tr.CASES[name], seed=a.seed)
        return c if a.n_substeps is None else dataclasses.replace(c, n_substeps=a.n_substeps)

    suffix = ("" if a.n_substeps is None else f"_n{a.n_substeps}") + (
        "" if a.seed == 0 else f"_s{a.seed}"
    )
    for name in a.cases:
        t0 = time.perf_counter()
        res = tr.reciprocity(case(name))
        _save(a.out, name + suffix, res, {"seconds": time.perf_counter() - t0})
    for variant in a.teeth:
        name = tr.TEETH_CASES[variant]
        t0 = time.perf_counter()
        res = tr.reciprocity(case(name), backward_variant=variant)
        _save(a.out, f"teeth_{variant}_{name}{suffix}", res,
              {"seconds": time.perf_counter() - t0, "variant": variant})  # fmt: skip
    if a.figure is not None:
        sys.path.insert(0, str(REPO / "scripts" / "validation"))
        from figures import tier0

        print("figure:", tier0.reciprocity(a.figure), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
