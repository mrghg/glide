"""Command line for the tracer readers and scoring.

Run from ``scripts/validation``::

    python -m tracers verify-datem            # reproduce mdl_stat/<exp>/stat[AT]?_000.txt
    python -m tracers bootstrap --seeds 20    # -b ranks over seeds
    python -m tracers rules                   # pair counts per assignment rule
    python -m tracers export-samples OUTDIR   # sample list and receptors for A11
    python -m tracers etex-match              # JRC vs DATEM ETEX-1, sample by sample
    python -m tracers atmes2-filter           # ATMES-II zero filter on the JRC file
    python -m tracers atmes2-hysplit          # ATMES-II statistics of DATEM's HYSPLIT example
    python -m tracers score MEAS PRED [opts]  # score any DATEM-format pair of files

Data paths come from ``$GLIDE_DATEM`` and ``$GLIDE_ETEX``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import atmes2, datem_stats, etex, experiments
from .readers import datem_root, read_datem_meas, read_etex_samples

#: (DATEM directory, tracer) pairs that B10 must reproduce, with OKC80 as an extra.
DATEM_CASES = [
    ("captex", 1),
    ("anatex", 1),
    ("anatex", 2),
    ("etex", 1),
    ("acurate", 1),
    ("okc80", 1),
]


def _options(args) -> datem_stats.StatOptions:
    return datem_stats.StatOptions(
        zero_percentile=args.zero,
        exclude_zero_pairs=args.exclude_zero_pairs,
        model_zero=args.model_zero,
        plume_only=args.plume,
        averaging=args.averaging,
        bootstrap=args.bootstrap,
        seed=args.seed,
        contingency_level=args.level,
    )


def verify_datem(root: Path | None = None) -> list[tuple[str, int, str, list]]:
    """Score DATEM's HYSPLIT predictions with run_stat.sh's options (-z0.0,
    zero-zero pairs kept, with and without -at) and compare every printed
    value with the posted file. Returns (exp, tracer, kind, differences)."""
    root = root or datem_root()
    out = []
    for exp, tr in DATEM_CASES:
        m = read_datem_meas(root / "exp_data" / exp / f"meas-t{tr}.txt")
        p = read_datem_meas(root / "mdl_data" / exp / f"ct{tr}_000.txt", experiment=exp)
        for kind, avg in (("A", None), ("T", "time")):
            s = datem_stats.score(m, p, datem_stats.StatOptions(averaging=avg))
            diffs = datem_stats.compare_with_stat_file(
                s, root / "mdl_stat" / exp / f"stat{kind}{tr}_000.txt"
            )
            out.append((exp, tr, kind, s.rank, diffs))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tracers", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-datem")
    b = sub.add_parser("bootstrap")
    b.add_argument("--seeds", type=int, default=20)
    sub.add_parser("rules")
    e = sub.add_parser("export-samples")
    e.add_argument("outdir", type=Path)
    sub.add_parser("etex-match")
    sub.add_parser("atmes2-hysplit")
    f = sub.add_parser("atmes2-filter")
    f.add_argument("--margin", type=int, default=2)
    f.add_argument("--last-interval", type=int, default=30)
    s = sub.add_parser("score")
    s.add_argument("meas", type=Path)
    s.add_argument("pred", type=Path)
    s.add_argument("-z", "--zero", type=float, default=0.0)
    s.add_argument("-x", "--exclude-zero-pairs", action="store_true")
    s.add_argument("-y", "--model-zero", action="store_true")
    s.add_argument("-p", "--plume", action="store_true")
    s.add_argument("-a", "--averaging", choices=["time", "space"], default=None)
    s.add_argument("-b", "--bootstrap", action="store_true")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("-l", "--level", type=float, default=1.0)
    args = ap.parse_args(argv)

    if args.cmd == "verify-datem":
        bad = 0
        for exp, tr, kind, rank, diffs in verify_datem():
            status = "identical" if not diffs else f"DIFFERS {diffs}"
            bad += bool(diffs)
            print(f"{exp:8s} t{tr} stat{kind}: rank {rank:.2f}  {status}")
        return 1 if bad else 0

    if args.cmd == "bootstrap":
        root = datem_root()
        for exp, tr in DATEM_CASES:
            m = read_datem_meas(root / "exp_data" / exp / f"meas-t{tr}.txt")
            p = read_datem_meas(root / "mdl_data" / exp / f"ct{tr}_000.txt", experiment=exp)
            r = [
                datem_stats.score(m, p, datem_stats.StatOptions(bootstrap=True, seed=k)).rank
                for k in range(args.seeds)
            ]
            print(
                f"{exp:8s} t{tr} -b over {args.seeds} seeds: mean {np.mean(r):.5f} "
                f"min {min(r):.5f} max {max(r):.5f}"
            )
        return 0

    if args.cmd == "rules":
        with pd.option_context("display.width", 250):
            print(experiments.rule_table().to_string())
        print(f"adopted: {experiments.ADOPTED_RULE.label}")
        return 0

    if args.cmd == "export-samples":
        args.outdir.mkdir(parents=True, exist_ok=True)
        samples, receptors = experiments.sample_list()
        samples.to_csv(args.outdir / "tracer_samples.csv", index=False)
        receptors.to_csv(args.outdir / "tracer_receptors.csv", index=False)
        print(samples.groupby("experiment").size().to_string())
        print(f"{len(samples)} samples, {len(receptors)} unique receptor windows")
        return 0

    if args.cmd == "etex-match":
        m = etex.match_jrc_datem()
        print(etex.summarise_match(m).to_string(index=False))
        odd = m[m["category"].isin(["value differs", "JRC code, value in DATEM", "DATEM only"])]
        print(odd.to_string(index=False))
        return 0

    if args.cmd == "atmes2-filter":
        j = read_etex_samples(1)
        k = atmes2.zero_filter(j, margin=args.margin, last_interval=args.last_interval)
        z = int((k & (j["conc_raw"] == 0)).sum())
        print(f"{int(k.sum())} pairs, {100 * z / k.sum():.1f}% measured zeros")
        print(atmes2.DEFINITIONS_NOTE)
        return 0

    if args.cmd == "atmes2-hysplit":
        # DATEM's HYSPLIT example predictions on the JRC samples: a check of the
        # pipeline, not a reference value (a different HYSPLIT run from ATMES-II's).
        j = read_etex_samples(1)
        p = read_datem_meas(datem_root() / "mdl_data" / "etex" / "ct1_000.txt", experiment="etex")
        prefixes = sorted({s.rstrip("0123456789") for s in j["site"].unique()})
        p["jrc_site"] = [etex.datem_to_jrc_code(s, prefixes) for s in p["site_num"]]
        key = p.set_index(["jrc_site", "start"])["conc_raw"] / 1000.0
        pred = pd.Series(
            key.reindex(pd.MultiIndex.from_arrays([j["site"], j["start"]])).to_numpy(),
            index=j.index,
        )
        for k, v in atmes2.score_against_jrc(j, pred).items():
            print(f"{k:20s} {v}" if isinstance(v, str) else f"{k:20s} {v:.4g}")
        return 0

    if args.cmd == "score":
        m = read_datem_meas(args.meas)
        p = read_datem_meas(args.pred, experiment=m["sample_id"].iloc[0].split("-")[0])
        st = datem_stats.score(m, p, _options(args))
        for k, v in st.as_dict().items():
            print(f"{k:28s} {v}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
