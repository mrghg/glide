"""Regenerate the validation figures and the gallery page in one command.

    .venv/bin/python scripts/validation/figures/make_all.py              # all Tier 0 figures
    .venv/bin/python scripts/validation/figures/make_all.py --only terrain well_mixed
    .venv/bin/python scripts/validation/figures/make_all.py --gallery-only
    .venv/bin/python scripts/validation/figures/make_all.py --root DIR   # somewhere else

Figures go to ``<root>/figures/<physics tag>/tier0/`` and the page to
``<root>/gallery.md``; the default root is ``docs/validation``. CPU only, about eight
minutes, under 2 GB of memory. Figures of earlier physics tags already under
``<root>/figures`` are kept, so the gallery compares versions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from figures import core, gallery, tier0  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=core.REPO_ROOT / "docs" / "validation")
    ap.add_argument("--only", nargs="*", choices=sorted(tier0.BUILDERS), default=None)
    ap.add_argument("--gallery-only", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    if not a.gallery_only:
        out = a.root / "figures" / core.physics_tag() / "tier0"
        for fid in a.only or list(tier0.BUILDERS):
            t0 = time.perf_counter()
            path = tier0.BUILDERS[fid](out)
            print(f"{fid:20s} {time.perf_counter() - t0:6.1f} s  {path}", flush=True)
    print(f"gallery: {gallery.write(a.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
