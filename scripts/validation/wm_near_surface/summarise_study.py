"""Tables from the fixcount_study.py JSON lines (decision 0027).

    python scripts/validation/wm_near_surface/summarise_study.py <study dir> [> tables.md]

For each harness, regime, outer step, count and variant: the mean over seeds of
each occupancy metric, the standard deviation across seeds ("sd"), the standard
error of the mean (sd/√seeds, "se") and the number of seeds. For the open top
the reference is the dt = 1.875 s run; "Δref" is the difference of the seed
means from it, with the combined standard error.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REGIMES = ("stable", "neutral", "convective")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(r: dict) -> tuple:
    return (r["harness"], r["regime"], r["dt"], r["n"], r["variant"], r.get("adapt_c"))


def group(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def stats(runs: list[dict], metric: str) -> tuple[float, float, float, int]:
    v = np.array([r["metrics"][metric] for r in runs])
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return float(v.mean()), sd, sd / math.sqrt(v.size), int(v.size)


def fmt(mean: float, se: float) -> str:
    return f"{mean:.3f} ± {se:.3f}"


def table(groups: dict, harness: str, metrics: list[str], ref_dt: float | None = None) -> str:
    keys = sorted(
        (k for k in groups if k[0] == harness),
        key=lambda k: (REGIMES.index(k[1]), -k[2], k[3] or 0, k[4]),
    )
    head = ["regime", "dt (s)", "n", "variant", "seeds"]
    for m in metrics:
        head += [f"{m} (mean ± se)", f"{m} sd"]
        if ref_dt is not None:
            head.append(f"{m} Δref")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in keys:
        runs = groups[k]
        row = [k[1], f"{k[2]:g}", str(k[3]), k[4] + (f" c={k[5]:g}" if k[5] else ""), ""]
        for m in metrics:
            mean, sd, se, n = stats(runs, m)
            row[4] = str(n)
            row += [fmt(mean, se), f"{sd:.3f}"]
            if ref_dt is not None:
                ref = groups.get((harness, k[1], ref_dt, 1, "fixed", None))
                if ref:
                    rm, _, rse, _ = stats(ref, m)
                    row.append(f"{mean - rm:+.3f} ± {math.hypot(se, rse):.3f}")
                else:
                    row.append("")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def paired(rows_a: list[dict], rows_b: list[dict], metrics: list[str]) -> str:
    """Paired (same regime, dt, n, seed) differences a − b, mean ± se over seeds."""
    b_index = {(r["regime"], r["dt"], r["n"], r["seed"]): r for r in rows_b}
    diffs = defaultdict(lambda: defaultdict(list))
    for r in rows_a:
        o = b_index.get((r["regime"], r["dt"], r["n"], r["seed"]))
        if o is None:
            continue
        for m in metrics:
            diffs[(r["regime"], r["dt"], r["n"])][m].append(r["metrics"][m] - o["metrics"][m])
    head = ["regime", "dt (s)", "n", "pairs"] + [f"Δ{m} (mean ± se)" for m in metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (reg, dt, n), d in sorted(diffs.items(), key=lambda x: (REGIMES.index(x[0][0]), -x[0][1])):
        row = [reg, f"{dt:g}", str(n), str(len(next(iter(d.values()))))]
        for m in metrics:
            v = np.array(d[m])
            se = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else float("nan")
            row.append(fmt(float(v.mean()), float(se)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    d = Path(sys.argv[1])
    closed = load(d / "closed.jsonl")
    open_ = load(d / "open.jsonl")
    clamp = load(d / "clamp.jsonl")
    adaptive = load(d / "adaptive.jsonl")
    old = load(d / "old.jsonl")
    out = []
    out.append("## Closed column (lid every sub-step); 1 = well mixed\n")
    out.append(table(group(closed + adaptive + old), "closed", ["lo5", "z40", "z10"]))
    out.append("\n\n## Open top (free troposphere above h); reference dt = 1.875 s\n")
    out.append(table(group(open_), "open", ["lo5", "z40", "frac_bl", "lo5_bl"], ref_dt=1.875))
    out.append("\n\n## Basal clamp: variant clamp0 (zero gradient below z_ubl) − clamp off\n")
    out.append(table(group(clamp), "closed", ["lo5", "z40", "z10", "z2"]))
    out.append("\n\nPaired differences (same seed):\n")
    out.append(
        paired(
            [r for r in clamp if r["variant"] == "clamp0"],
            [r for r in clamp if r["variant"] == "fixed_clampoff"],
            ["lo5", "z40", "z10", "z2"],
        )
    )
    if adaptive:
        out.append("\n\n## Adaptivity: k_i = max(n, ceil(dt/(c T_Lw,i))) − fixed n (same seed)\n")
        out.append(
            paired(adaptive, [r for r in closed if r["variant"] == "fixed"], ["lo5", "z40", "z10"])
        )
        hist = defaultdict(lambda: np.zeros(65))
        for r in adaptive:
            hist[(r["regime"], r["n"])] += np.array(r["adaptive_k_hist"])
        out.append("\n\nAdaptive sub-step counts (share of particle-steps by k):\n")
        for (reg, n), h in sorted(hist.items()):
            share = {k: round(float(v / h.sum()), 3) for k, v in enumerate(h) if v}
            out.append(f"\n- {reg}, n = {n}: {share}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
