"""PR #73's reciprocity cases (tests/test_reciprocity.py, its own procedure and
counts), with the review's extra requirement: the per-cell test restricted to
cells that at least 50 distinct FORWARD particles reach (as well as the 50
backward particles of the selection run). Env GLIDE_WT = the source tree
(its HannaScheme defaults are the candidate under test).
Usage: run_a6.py OUT.json [--cases ...] [--nsub N] [--seed S] [--threads T]"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

WT = os.environ["GLIDE_WT"]
sys.path.insert(0, WT + "/tests")
import lpdm  # noqa: E402

assert lpdm.__file__.startswith(WT), lpdm.__file__
import test_reciprocity as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument(
    "--cases", nargs="*", default=["stable_surface", "stable_below_h", "stable_above_h"]
)
ap.add_argument("--nsub", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
torch.set_num_threads(a.threads)
s = R.HannaScheme()
info = dict(
    wt=WT,
    mode=getattr(s, "DRIFT_GRADIENT_MODE", "n/a"),
    form=getattr(s, "SIGMA_W_FLOOR_FORM", "n/a"),
    sigma_min=__import__("lpdm.turbulence.hanna", fromlist=["x"]).SIGMA_MIN_M_S,
)
print(info, flush=True)
results = {"info": info}
for name in a.cases:
    case = dataclasses.replace(R.CASES[name], seed=a.seed, n_substeps=a.nsub)
    t0 = time.perf_counter()
    res = R.reciprocity(case)
    rr = case.receptor_row
    cf = R.forward_contributions(case, "production")[:, :, rr, :]
    nf = (cf > 0).sum(1).numpy()  # distinct forward particles per [source row, k]
    t = res.tested
    t50 = t & (nf >= 50)
    agree = res.agree
    summ = res.summary()
    summ.update(
        passes=bool(R._passes(res)),
        n_tested_nf50=int(t50.sum()),
        frac_agree_nf50=float(agree[t50].mean()) if t50.any() else None,
        nf_tested_quantiles=np.percentile(nf[t], [0, 10, 50, 90, 100]).tolist(),
        sum_ratio_nf50=float(res.backward[t50].sum() / res.forward[t50].sum())
        if t50.any()
        else None,
        # row totals over the tested cells, backward / forward, per source row
        row_ratio=[
            float(res.backward[r][t[r]].sum() / res.forward[r][t[r]].sum()) if t[r].any() else None
            for r in range(case.grid.n_rows)
        ],
        seconds=time.perf_counter() - t0,
    )
    results[name] = summ
    print(name, json.dumps({k: summ[k] for k in ("n_tested", "frac_agree", "sum_ratio", "sum_ok",
                                                 "passes", "n_tested_nf50", "frac_agree_nf50")}),
          flush=True)  # fmt: skip
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
