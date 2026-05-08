"""GPU-based footprint accumulation module.

Output units convention: each cell of the footprint tensor accumulates
`Σ over active particles in cell of (weight_i × dt_step_seconds)` where `weight_i`
is the per-particle weight set by the release generator (typically `1/n_particles`
for uniform releases). The raw value therefore has dimensionality
`(dimensionless mass fraction) × seconds`, i.e. residence time per unit released
mass.

Conversion to a physical sensitivity (e.g. ppm per (μmol/m²/s) emission flux)
requires additional factors not applied here: cell volume, mixed-layer thickness,
air density, and molar-mass conversions. See Lin et al. 2003 (STILT) or
Seibert & Frank 2004 for the standard recipes. Downstream code is responsible
for converting the raw accumulator to whichever physical sensitivity is needed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from lpdm.runtime import DEVICE


class FootprintGridder:
        """Allocate and own the footprint tensor on the configured device.

        Output dimensions are `(time_ago, z_bin, latitude, longitude)`. The vertical
        bin layout is controlled by `z_edges_m`, a strictly-ascending sequence of bin
        edges in metres above ground level (so `z_edges_m=(0, 40, 1000, 5000)` defines
        three bins: surface 0-40 m, mixed-layer 40-1000 m, free-troposphere 1000-5000 m).
        The number of vertical bins is `len(z_edges_m) - 1`. See the module docstring
        for the units convention used by `accumulate`.
        """

        def __init__(
                self,
                lon_bounds: tuple[float, float],
                lat_bounds: tuple[float, float],
                z_edges_m: Sequence[float],
                n_time_bins: int,
                n_y: int,
                n_x: int,
                *,
                device: torch.device | str = DEVICE,
                dtype: torch.dtype = torch.float32,
        ) -> None:
                z_edges_arr = np.asarray(list(z_edges_m), dtype=float)
                if z_edges_arr.ndim != 1 or z_edges_arr.size < 2:
                        raise ValueError("z_edges_m must be a 1D sequence with at least 2 values")
                if not np.all(np.diff(z_edges_arr) > 0):
                        raise ValueError("z_edges_m must be strictly ascending")

                n_z_bins = int(z_edges_arr.size - 1)
                if min(n_time_bins, n_y, n_x) <= 0 or n_z_bins <= 0:
                        raise ValueError("All footprint dimensions must be > 0")

                self.device = torch.device(device)
                self.dtype = dtype
                self.tensor = torch.zeros(
                        (n_time_bins, n_z_bins, n_y, n_x),
                        device=self.device,
                        dtype=self.dtype,
                )

                self.lon_min, self.lon_max = lon_bounds
                self.lat_min, self.lat_max = lat_bounds
                # Edge tensor stays on the gridder device so torch.bucketize avoids host syncs.
                self.z_edges = torch.tensor(z_edges_arr, dtype=self.dtype, device=self.device)
                # Convenience scalars for downstream code wanting overall vertical extent.
                self.z_min = float(z_edges_arr[0])
                self.z_max = float(z_edges_arr[-1])
                self.n_t = n_time_bins
                self.n_z = n_z_bins
                self.n_y = n_y
                self.n_x = n_x

        def accumulate(
                self,
                particles: torch.Tensor,
                active_mask: torch.Tensor,
                weights: torch.Tensor,
                t_idx: int,
                dt_seconds: float,
        ) -> None:
                """Accumulate weighted residence time in the footprint grid.

                Each in-bounds active particle adds `weights[i] * dt_seconds` to the cell
                it currently occupies at time bin `t_idx`. Inactive particles are skipped;
                out-of-bounds particles are silently dropped (no edge clamping); an
                out-of-range `t_idx` is a silent no-op. See module docstring for the units
                convention.

                Args:
                        particles: float tensor [N, 3] (lon, lat, alt).
                        active_mask: bool tensor [N].
                        weights: float tensor [N] of per-particle release weights.
                        t_idx: current time-bin index in [0, n_time_bins).
                        dt_seconds: timestep length over which `weights` apply.
                """
                if not torch.any(active_mask):
                        return
                        
                if not (0 <= t_idx < self.n_t):
                        return
                        
                act_p = particles[active_mask]
                act_w = weights[active_mask]

                # Horizontal: uniform fractional indices.
                x_frac = (act_p[:, 0] - self.lon_min) / max(1e-6, self.lon_max - self.lon_min) * (self.n_x)
                y_frac = (act_p[:, 1] - self.lat_min) / max(1e-6, self.lat_max - self.lat_min) * (self.n_y)
                x_idx = torch.floor(x_frac).long()
                y_idx = torch.floor(y_frac).long()

                # Vertical: bucketize against (potentially non-uniform) edges. With
                # right=False, bucketize returns the smallest i such that z < edges[i];
                # subtracting 1 gives the half-open bin index in [0, n_z-1] for in-range
                # values. Below the first edge → -1, at-or-above the last edge → n_z;
                # both are filtered by valid_mask below.
                z_idx = torch.bucketize(act_p[:, 2], self.z_edges, right=False) - 1
                
                # Filter out-of-bounds
                valid_mask = (
                        (x_idx >= 0) & (x_idx < self.n_x) &
                        (y_idx >= 0) & (y_idx < self.n_y) &
                        (z_idx >= 0) & (z_idx < self.n_z)
                )
                
                if not torch.any(valid_mask):
                        return
                        
                x_idx = x_idx[valid_mask]
                y_idx = y_idx[valid_mask]
                z_idx = z_idx[valid_mask]
                w = act_w[valid_mask]
                
                # We want to add w * dt_seconds to tensor[t_idx, z_idx, y_idx, x_idx]
                flat_idx = z_idx * (self.n_y * self.n_x) + y_idx * self.n_x + x_idx
                
                # Create a 1D view of the selected time bin spatial grid
                target_slice = self.tensor[t_idx].view(-1)
                target_slice.scatter_add_(0, flat_idx, w * dt_seconds)
