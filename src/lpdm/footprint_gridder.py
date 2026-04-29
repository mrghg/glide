"""GPU-based footprint accumulation module.

Implementation TODO:
- Initialize [time_ago, z_bin, y, x] footprint tensor
- Bin particles by altitude and horizontal index
- Accumulate weighted residence time with scatter_add_
"""

from __future__ import annotations

import torch

from lpdm.runtime import DEVICE


class FootprintGridder:
        """Allocate and own the footprint tensor on the configured device."""

        def __init__(
                self,
                lon_bounds: tuple[float, float],
                lat_bounds: tuple[float, float],
                z_bounds: tuple[float, float],
                n_time_bins: int,
                n_y: int,
                n_x: int,
                *,
                n_z_bins: int = 4,
                device: torch.device | str = DEVICE,
                dtype: torch.dtype = torch.float32,
        ) -> None:
                if min(n_time_bins, n_z_bins, n_y, n_x) <= 0:
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
                self.z_min, self.z_max = z_bounds
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
                
                Args:
                        particles: float tensor [N, 3] (lon, lat, alt)
                        active_mask: bool tensor [N]
                        weights: float tensor [N] (particle mass/weight)
                        t_idx: current time step bin index (0 to n_t-1)
                        dt_seconds: time spent in this bin
                """
                if not torch.any(active_mask):
                        return
                        
                if not (0 <= t_idx < self.n_t):
                        return
                        
                act_p = particles[active_mask]
                act_w = weights[active_mask]
                
                # Convert coords to fractional indices
                # (lon, lat, z) -> (x, y, z)
                x_frac = (act_p[:, 0] - self.lon_min) / max(1e-6, self.lon_max - self.lon_min) * (self.n_x)
                y_frac = (act_p[:, 1] - self.lat_min) / max(1e-6, self.lat_max - self.lat_min) * (self.n_y)
                z_frac = (act_p[:, 2] - self.z_min) / max(1e-6, self.z_max - self.z_min) * (self.n_z)
                
                x_idx = torch.floor(x_frac).long()
                y_idx = torch.floor(y_frac).long()
                z_idx = torch.floor(z_frac).long()
                
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
