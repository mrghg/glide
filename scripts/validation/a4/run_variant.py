#!/usr/bin/env python
"""Task A4: run GLIDE with a rounding-level variant of the particle positions.

    python scripts/validation/a4/run_variant.py --src <checkout>/src \
        --variant {plain,shifted_origin,ulp} -- --config <yaml> --device cpu \
        --output-uri <dir>

Runs ``lpdm.main.main`` from ``--src`` (put first on ``sys.path``), after one of
these patches, then passes the remaining arguments to it:

- ``plain``: none (the checkout as it is; ``GLIDE_POSITION_FRAME`` applies).
- ``shifted_origin`` (A4 code only): the local metric frame with its origin moved
  by (+7.31 deg, -3.17 deg) from the met-domain centre. The motion is the same in
  exact arithmetic; only the rounding of the positions differs. The spread
  between this and ``plain`` is a rounding-level reference.
- ``ulp``: every released particle's longitude and latitude moved by one float32
  ulp towards +inf, before anything else touches them. A second rounding-level
  reference that works on main as well.

Nothing here changes the model's code on disk.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--variant", choices=["plain", "shifted_origin", "ulp"], required=True)
    args, rest = p.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    sys.path.insert(0, args.src)

    import torch

    import lpdm.main as lm

    if args.variant == "shifted_origin":
        from lpdm.local_frame import HorizontalFrame

        def shifted(cfg):
            md = cfg.met_domain
            base = HorizontalFrame.local_metric(md.lon_bounds, md.lat_bounds)
            return HorizontalFrame(
                "local_metric", base.lon0 + 7.31, base.lat0 - 3.17, base.kx, base.ky
            )

        lm._position_frame = shifted
    elif args.variant == "ulp":
        original = lm.generate_batch_particles

        def nudged(*a, **k):
            batch = original(*a, **k)
            pos = batch.particles
            inf = torch.full_like(pos[:, :2], float("inf"))
            pos[:, :2] = torch.nextafter(pos[:, :2], inf)
            return batch

        lm.generate_batch_particles = nudged

    lm.main(rest)


if __name__ == "__main__":
    main()
