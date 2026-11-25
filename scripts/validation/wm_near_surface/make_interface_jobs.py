"""Job files for run_interface_jobs.sh (decision 0028 study).

    python make_interface_jobs.py <set> <outputs dir> <this branch src root> \
        [<PR #69 head src root> <main src root>]

Writes ``<outputs dir>/jobs_<set>.txt``; results go to ``<outputs dir>/*.jsonl``.
"""

import sys
from pathlib import Path

REG = ("stable", "neutral", "convective")


def main() -> None:
    name, out_dir, new = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    pr = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    old = Path(sys.argv[5]) if len(sys.argv) > 5 else None

    def job(src, out, **kw):
        args = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in kw.items())
        return f"{src} {args} --out {out_dir / out}"

    sets = {
        # h moving between steps: production against rescaling w', u', v' on a
        # side change.
        "hvar": [
            job(new, "hvar.jsonl", regime=r, harness="hvar", variant=v, n=16, seed=s)
            for r in REG
            for v in ("prod", "rescale")
            for s in (1, 2, 3)
        ],
        # Main study: this branch at n = 4/8/16/32, 5 seeds, three harnesses.
        "main": [
            job(new, f"{h}.jsonl", regime=r, harness=h, variant="prod", n=n, seed=s)
            for n in (32, 16, 8, 4)
            for h in ("open2h", "openbl", "closed")
            for r in REG
            for s in (1, 2, 3, 4, 5)
        ],
        # Mechanism isolation: one half of the interface at a time.
        "mech": [
            job(new, "mech_open2h.jsonl", regime=r, harness="open2h", variant=v, n=n, seed=s)
            for n in (64, 16, 4)
            for v in ("nocross", "nostencil")
            for r in REG
            for s in (1, 2, 3)
        ],
    }
    if pr is not None and old is not None:
        # The review's finding 1, re-measured: PR #69 head and main.
        sets["verify"] = [
            job(pr, "verify_pr_open2h.jsonl", regime=r, harness="open2h", n=n, seed=s)
            for n in (64, 32, 16, 8, 4)
            for r in REG
            for s in (1, 2, 3)
        ] + [
            job(old, "verify_main_open2h.jsonl", regime=r, harness="open2h", variant="old", seed=s)
            for r in REG
            for s in (1, 2, 3)
        ]
    path = out_dir / f"jobs_{name}.txt"
    path.write_text("\n".join(sets[name]) + "\n")
    print(path, len(sets[name]))


if __name__ == "__main__":
    main()
