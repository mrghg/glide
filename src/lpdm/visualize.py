"""Visualization helpers for LPDM diagnostics.

- 3D trajectory visualization with Plotly.
- Altitude histogram relative to boundary layer height with Matplotlib.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_3d_trajectories(
    trajectories: np.ndarray,
    *,
    max_particles: int = 200,
    title: str = "LPDM Particle Trajectories",
) -> go.Figure:
    """Plot 3D trajectories.

    Args:
        trajectories: Array shaped (T, N, 3) in [x, y, z] order.
        max_particles: Plot at most this many particles for readability.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """

    arr = np.asarray(trajectories)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("trajectories must have shape (T, N, 3)")

    t_steps, n_particles, _ = arr.shape
    if t_steps < 2:
        raise ValueError("trajectories must include at least two time steps")

    show_n = min(max_particles, n_particles)
    fig = go.Figure()

    for i in range(show_n):
        xyz = arr[:, i, :]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                line={"width": 2},
                opacity=0.7,
                name=f"p{i}",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
        },
        template="plotly_white",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def plot_altitude_histogram_relative_to_blh(
    altitude_m: Iterable[float],
    blh_m: Iterable[float] | float,
    *,
    bins: int = 50,
    title: str = "Altitude Relative to Boundary Layer Height",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot histogram of z - BLH.

    Args:
        altitude_m: Particle altitudes in meters.
        blh_m: BLH values in meters; scalar or per-particle iterable.
        bins: Number of histogram bins.
        title: Plot title.

    Returns:
        Matplotlib Figure and Axes.
    """

    z = np.asarray(list(altitude_m), dtype=float)
    blh = np.asarray(blh_m, dtype=float)

    if z.ndim != 1:
        raise ValueError("altitude_m must be one-dimensional")
    if blh.ndim > 1:
        raise ValueError("blh_m must be a scalar or one-dimensional")

    if blh.ndim == 0:
        z_rel = z - float(blh)
    else:
        if blh.shape[0] != z.shape[0]:
            raise ValueError("When blh_m is an array, it must match altitude_m length")
        z_rel = z - blh

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(z_rel, bins=bins, color="#2a9d8f", alpha=0.85, edgecolor="white")
    ax.axvline(0.0, color="#e76f51", linestyle="--", linewidth=2, label="BLH")
    ax.set_title(title)
    ax.set_xlabel("Altitude relative to BLH (m)")
    ax.set_ylabel("Particle count")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    return fig, ax
