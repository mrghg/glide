"""Explainer figure for footprint residence counting (docs/physics.md §7, decision 0029).

Panel (a): one particle's height against time over its first five 60 s steps from
a 10 m release, the 0-40 m layer shaded, the time it truly spends in the layer,
and where `endpoint` (all of each step at the step's end) and `trapezoid` (half
of each step at each end) put their deposits, with the totals each rule counts in
the layer. Panel (b): how `trapezoid` splits a half step between age bins (the
reviewer's step of ages 80-140 s against 120 s bins).

The path is a made-up smooth curve (what a particle's height might do), not a
GLIDE trajectory; the deposits and the age-bin split are counted by GLIDE's own
`lpdm.main._deposit_step` and `lpdm.main._age_bin_pieces` on a one-cell grid.

Usage: python scripts/validation/figures/residence_counting.py [out.png]
(default docs/img/residence-counting.png). CPU, a few seconds.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from lpdm.footprint_gridder import FootprintGridder  # noqa: E402
from lpdm.main import _age_bin_pieces, _deposit_step, _time_ago_index  # noqa: E402

DT = 60.0
N_STEPS = 5
LAYER_TOP = 40.0
# A made-up smooth height path (s, m) from a 10 m release: up through the layer
# top, back down, up again.
KNOTS_T = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
KNOTS_Z = [10, 22, 31, 52, 47, 26, 18, 33, 61, 64, 55]


def _counted_in_layer(position: str, z_nodes: np.ndarray) -> float:
    """Seconds GLIDE counts in the 0-40 m layer over the steps, with this rule."""

    grid = FootprintGridder(
        lon_bounds=(0.0, 1.0),
        lat_bounds=(0.0, 1.0),
        z_edges_m=(0.0, LAYER_TOP, 1000.0),
        n_time_bins=1,
        n_y=1,
        n_x=1,
        device="cpu",
        dtype=torch.float64,
    )
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    met = SimpleNamespace(metadata=SimpleNamespace(time_start=t0, time_end=t0 + timedelta(hours=1)))
    for k in range(N_STEPS):
        start = torch.tensor([[0.5, 0.5, z_nodes[k]]], dtype=torch.float64)
        end = torch.tensor([[0.5, 0.5, z_nodes[k + 1]]], dtype=torch.float64)
        age = torch.tensor([k * DT], dtype=torch.float32)
        _deposit_step(
            grid,
            position=position,
            start_xyz=start,
            end_xyz=end,
            active_mask=torch.ones(1, dtype=torch.bool),
            weights=torch.ones(1, dtype=torch.float64),
            elapsed_s=age,
            t_idx=_time_ago_index(age, 3600.0, 1),
            time_bin_s=3600.0,
            release_idx=torch.zeros(1, dtype=torch.int64),
            delta_s=DT,
            t_cursor=t0 + timedelta(seconds=1800),
            particle_density=None,
            met_window=met,
            t_alpha=0.5,
            engine=None,
        )
    return float(grid.tensor[0, 0, 0].sum())


def panel_rules(ax) -> None:
    path = PchipInterpolator(KNOTS_T, KNOTS_Z)
    t = np.linspace(0.0, N_STEPS * DT, 3001)
    z = path(t)
    t_nodes = np.arange(N_STEPS + 1) * DT
    z_nodes = path(t_nodes)
    inside = z < LAYER_TOP
    true_s = float(np.sum(inside[:-1]) * (t[1] - t[0]))

    ax.axhspan(0, LAYER_TOP, color="#d9ead3", zorder=0)
    ax.text(3, LAYER_TOP - 3, "0–40 m layer", va="top", fontsize=8, color="#38761d")
    ax.plot(t, z, color="0.35", lw=1.4, zorder=2, label="particle height (illustrative)")
    # The time truly spent in the layer, as a bar along the bottom.
    edges = np.flatnonzero(np.diff(inside.astype(int)))
    bounds = np.concatenate(([0], edges + 1, [len(t) - 1]))
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if inside[a]:
            ax.plot([t[a], t[b]], [-6, -6], color="#38761d", lw=5, solid_capstyle="butt")
    for tn in t_nodes:
        ax.axvline(tn, color="0.85", lw=0.8, zorder=1)

    # Deposits: marker area ~ seconds; filled when the deposit lands in the layer.
    scale = 2.2
    for k, (tn, zn) in enumerate(zip(t_nodes, z_nodes, strict=True)):
        if k > 0:
            ax.scatter(
                tn + 4,
                zn,
                s=scale * DT,
                marker="s",
                zorder=3,
                facecolor="#cc4125" if zn < LAYER_TOP else "white",
                edgecolor="#cc4125",
            )
        ax.scatter(
            tn - 4,
            zn,
            s=scale * DT / 2,
            marker="o",
            zorder=3,
            facecolor="#1155cc" if zn < LAYER_TOP else "white",
            edgecolor="#1155cc",
        )
    ep = _counted_in_layer("endpoint", z_nodes)
    tz = _counted_in_layer("trapezoid", z_nodes)
    ax.scatter(
        [],
        [],
        s=scale * DT,
        marker="s",
        facecolor="#cc4125",
        edgecolor="#cc4125",
        label=f"endpoint: Δt at each step's end → {ep:.0f} s in the layer",
    )
    ax.scatter(
        [],
        [],
        s=scale * DT / 2,
        marker="o",
        facecolor="#1155cc",
        edgecolor="#1155cc",
        label=f"trapezoid: Δt/2 at each end → {tz:.0f} s in the layer",
    )
    ax.plot([], [], color="#38761d", lw=5, label=f"true time in the layer: {true_s:.0f} s")
    ax.set_xlim(-10, N_STEPS * DT + 10)
    ax.set_ylim(-12, 108)
    ax.set_xticks(t_nodes)
    ax.set_xlabel("time before the measurement (s); steps of Δt = 60 s, release at 0")
    ax.set_ylabel("height (m)")
    ax.set_title("(a) where each rule counts a particle's residence", loc="left", fontsize=10)
    ax.legend(
        loc="upper left",
        fontsize=7.5,
        framealpha=0.95,
        ncol=1,
        title="filled: deposit lands in the layer",
        title_fontsize=7.5,
    )
    ax.annotate(
        "endpoint never counts\nthe first half step\nat the release (10 m)",
        xy=(0, z_nodes[0]),
        xytext=(18, 2),
        fontsize=7.5,
        color="#cc4125",
        arrowprops=dict(arrowstyle="->", color="#cc4125", lw=0.8),
    )
    print(f"panel (a): true {true_s:.1f} s, endpoint {ep:.1f} s, trapezoid {tz:.1f} s")


def panel_split(ax) -> None:
    width, age0 = 120.0, 80.0
    half = DT / 2
    rows = []
    for name, start in (
        ("start half (start position)", age0),
        ("end half (end position)", age0 + half),
    ):
        pieces = _age_bin_pieces(torch.tensor([start], dtype=torch.float32), half, width, 2)
        for idx, sec in pieces:
            s = float(sec if isinstance(sec, float) else sec[0])
            if s > 0:
                rows.append((name, int(idx[0]), s))
    colors = {0: "#e69138", 1: "#674ea7"}
    y = {"endpoint (whole step)": 2, "start half (start position)": 1, "end half (end position)": 0}
    ax.barh(2, DT, left=age0, height=0.5, color=colors[0], edgecolor="k", lw=0.6)
    ax.text(age0 + DT / 2, 2, "60 s → bin 0", ha="center", va="center", fontsize=8, color="w")
    cursor = {"start half (start position)": age0, "end half (end position)": age0 + half}
    for name, k, s in rows:
        ax.barh(y[name], s, left=cursor[name], height=0.5, color=colors[k], edgecolor="k", lw=0.6)
        ax.text(
            cursor[name] + s / 2,
            y[name],
            f"{s:.0f} s → bin {k}" if s >= 15 else f"{s:.0f} s",
            ha="center",
            va="center",
            fontsize=7.5,
            color="w",
        )
        cursor[name] += s
    ax.axvline(width, color="k", ls="--", lw=1)
    ax.text(width + 1, 2.55, "age-bin edge (120 s)", fontsize=7.5)
    ax.set_yticks(list(y.values()), list(y.keys()), fontsize=8)
    ax.set_xlim(60, 160)
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlabel("age of the step's interval (s since the release window's end)")
    ax.set_title(
        "(b) trapezoid splits each half step between age bins: 40 s / 20 s, exact",
        loc="left",
        fontsize=10,
    )
    totals = {0: 0.0, 1: 0.0}
    for _, k, s in rows:
        totals[k] += s
    print(f"panel (b): trapezoid bin 0 {totals[0]:.0f} s, bin 1 {totals[1]:.0f} s; endpoint 60/0")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/img/residence-counting.png")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), gridspec_kw={"height_ratios": [2.2, 1]})
    panel_rules(axes[0])
    panel_split(axes[1])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
