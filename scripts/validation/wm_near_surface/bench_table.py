"""Timing and exceedance table from benchmark runs (decision 0027).

    python scripts/validation/wm_near_surface/bench_table.py <bench dir> <label> [...]

Each label names ``<label>.bench.json`` (scripts/agent/benchmark.py --json) and
the run directory ``<label>/`` with its run_metadata.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def phase(table: dict, name: str) -> dict | None:
    for row in table["phases"]:
        if row["phase"] == name:
            return row
    return None


def main() -> None:
    root = Path(sys.argv[1])
    head = (
        "| run | wall (s) | footprints/s | step total (s) | step ms/step | step % wall "
        "| met_fetch (s) | BL sub-steps > 0.5 T_Lw | FT sub-steps > 0.5 T_Lw "
        "| FT sub-steps > T_Lw | FT share of sub-steps | worst ratio BL / FT |"
    )
    print(head)
    print("|" + "---|" * 12)
    for label in sys.argv[2:]:
        b = json.loads((root / f"{label}.bench.json").read_text())
        meta = json.loads((root / label / "run_metadata.json").read_text())
        t = b["phase_table"]
        step, fetch = phase(t, "step"), phase(t, "met_fetch")
        rt = meta.get("runtime", {})
        exc = rt.get("substep_exceedance")
        if exc:
            bl, ft = exc["boundary_layer"], exc["free_troposphere"]
            n_all = bl["substeps"] + ft["substeps"]
            cols = [
                f"{bl['exceeded_fraction']:.2%} of {bl['substeps']:.3g}",
                f"{ft['exceeded_fraction']:.2%} of {ft['substeps']:.3g}",
                f"{ft['over_T_L_fraction']:.2%}",
                f"{ft['substeps'] / n_all:.1%}",
                f"{bl['worst_sub_dt_over_T_Lw']:.2f} / {ft['worst_sub_dt_over_T_Lw']:.2f}",
            ]
        else:
            sat = rt.get("substep_saturation") or {}
            cols = [
                f"(cap reached: {sat.get('saturated_fraction', float('nan')):.2%} of "
                "particle-steps)",
                "",
                "",
                "",
                f"{sat.get('worst_sub_dt_over_T_Lw', float('nan')):.2f}",
            ]
        print(
            f"| {label} | {b['process_wall_s']:.0f} | {b['footprints_per_s']:.4f} "
            f"| {step['total_s']:.1f} | {step['ms_per_call']:.2f} | {step['pct_wall']:.1f} "
            f"| {fetch['total_s']:.0f} | " + " | ".join(cols) + " |"
        )


if __name__ == "__main__":
    main()
