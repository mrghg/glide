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

import os
from typing import Sequence

import numpy as np
import torch

from lpdm.runtime import DEVICE


class FootprintGridder:
        """Allocate and own the footprint tensor on the configured device.

        Output dimensions are `(n_releases, time_ago, z_bin, latitude, longitude)`.
        The leading dim was added in M5 stage 4 so a single batch can carry one
        footprint per release; for single-release runs `n_releases=1` and the
        leading axis is trivially size-1 (downstream code can squeeze it). The
        vertical bin layout is controlled by `z_edges_m`, a strictly-ascending
        sequence of bin edges in metres above ground level (so
        `z_edges_m=(0, 40, 1000, 5000)` defines three bins: surface 0-40 m,
        mixed-layer 40-1000 m, free-troposphere 1000-5000 m). The number of
        vertical bins is `len(z_edges_m) - 1`. See the module docstring for the
        units convention used by `accumulate`.
        """

        # Hard cap on the dense in-memory footprint tensor. Stops misconfigured
        # runs (large n_releases × n_time_bins × spatial grid) from hitting a
        # cryptic CPU/CUDA OOM during allocation. Set by env var
        # `LPDM_FOOTPRINT_MAX_GIB` if you genuinely need a bigger allocation;
        # the right long-term fix is streaming per-batch Zarr writes (flagged
        # as an M5 follow-up in CHECKPOINT.md).
        MAX_FOOTPRINT_GIB: float = 32.0

        def __init__(
                self,
                lon_bounds: tuple[float, float],
                lat_bounds: tuple[float, float],
                z_edges_m: Sequence[float],
                n_time_bins: int,
                n_y: int,
                n_x: int,
                *,
                n_releases: int = 1,
                device: torch.device | str = DEVICE,
                dtype: torch.dtype = torch.float32,
        ) -> None:
                z_edges_arr = np.asarray(list(z_edges_m), dtype=float)
                if z_edges_arr.ndim != 1 or z_edges_arr.size < 2:
                        raise ValueError("z_edges_m must be a 1D sequence with at least 2 values")
                if not np.all(np.diff(z_edges_arr) > 0):
                        raise ValueError("z_edges_m must be strictly ascending")

                n_z_bins = int(z_edges_arr.size - 1)
                if min(n_time_bins, n_y, n_x, n_releases) <= 0 or n_z_bins <= 0:
                        raise ValueError("All footprint dimensions must be > 0")

                self.device = torch.device(device)
                self.dtype = dtype

                # Pre-check tensor size against the cap (overridable via env) so a
                # misconfigured run fails with a clear message instead of OOM.
                bytes_per_element = torch.empty((), dtype=self.dtype).element_size()
                n_elements = int(n_releases) * int(n_time_bins) * n_z_bins * int(n_y) * int(n_x)
                requested_bytes = n_elements * bytes_per_element
                env_cap = os.environ.get("LPDM_FOOTPRINT_MAX_GIB")
                cap_gib = float(env_cap) if env_cap else self.MAX_FOOTPRINT_GIB
                cap_bytes = int(cap_gib * (1024 ** 3))
                if requested_bytes > cap_bytes:
                        raise MemoryError(
                                "Footprint tensor would require "
                                f"{requested_bytes / (1024 ** 3):.2f} GiB "
                                f"(shape (n_releases={n_releases}, n_time_bins={n_time_bins}, "
                                f"n_z={n_z_bins}, n_y={n_y}, n_x={n_x}), {self.dtype}). "
                                f"Cap is {cap_gib:.2f} GiB; raise it via LPDM_FOOTPRINT_MAX_GIB or "
                                "reduce n_time_bins / spatial resolution / n_releases. Note: the "
                                "FLEXPART comparison uses time-integrated footprints, so "
                                "n_time_bins=1 is usually the right choice for that workflow. "
                                "Streaming per-batch Zarr writes are an open M5 follow-up that "
                                "will lift this constraint for multi-batch runs."
                        )

                self.tensor = torch.zeros(
                        (n_releases, n_time_bins, n_z_bins, n_y, n_x),
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
                self.n_releases = int(n_releases)
                self.n_t = n_time_bins
                self.n_z = n_z_bins
                self.n_y = n_y
                self.n_x = n_x

        def accumulate(
                self,
                particles: torch.Tensor,
                active_mask: torch.Tensor,
                weights: torch.Tensor,
                t_idx: torch.Tensor,
                release_idx: torch.Tensor,
                dt_seconds: float,
        ) -> None:
                """Accumulate weighted residence time into the 5D footprint grid.

                Each in-bounds active particle adds `weights[i] * dt_seconds` to the
                cell it currently occupies at its own per-particle time bin `t_idx[i]`
                within its own release slice `release_idx[i]`. Inactive particles are
                skipped; out-of-bounds particles (horizontal, vertical, time, OR
                release) are silently dropped. See module docstring for the units
                convention.

                M5 stage 4 added the per-particle `release_idx` argument and the
                leading `n_releases` dim. Single-release runs pass a release_idx
                tensor of all zeros into a gridder with `n_releases=1`, which is
                bit-equivalent to stage 3's 4D scatter.

                Args:
                        particles: float tensor [N, 3] (lon, lat, alt).
                        active_mask: bool tensor [N].
                        weights: float tensor [N] of per-particle release weights.
                        t_idx: int64 tensor [N] of per-particle time-bin indices in
                                [0, n_time_bins). Out-of-range values are filtered.
                        release_idx: int64 tensor [N] of per-particle release indices
                                in [0, n_releases). Out-of-range values are filtered.
                        dt_seconds: timestep length over which `weights` apply.
                """
                if not torch.any(active_mask):
                        return

                act_p = particles[active_mask]
                act_w = weights[active_mask]
                act_t = t_idx[active_mask]
                act_r = release_idx[active_mask]

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
                # `act_p[:, 2]` is a non-contiguous column view; bucketize warns and
                # makes an internal copy unless we hand it a contiguous tensor.
                z_coord = act_p[:, 2].contiguous()
                z_idx = torch.bucketize(z_coord, self.z_edges, right=False) - 1

                # Filter out-of-bounds (release, horizontal, vertical, per-particle time).
                valid_mask = (
                        (act_r >= 0) & (act_r < self.n_releases) &
                        (x_idx >= 0) & (x_idx < self.n_x) &
                        (y_idx >= 0) & (y_idx < self.n_y) &
                        (z_idx >= 0) & (z_idx < self.n_z) &
                        (act_t >= 0) & (act_t < self.n_t)
                )

                if not torch.any(valid_mask):
                        return

                r_sel = act_r[valid_mask]
                t_sel = act_t[valid_mask]
                z_sel = z_idx[valid_mask]
                y_sel = y_idx[valid_mask]
                x_sel = x_idx[valid_mask]
                w = act_w[valid_mask]

                # 5D flat index: r*(T*Z*Y*X) + t*(Z*Y*X) + z*(Y*X) + y*X + x.
                # Equivalent to scattering into tensor[r, t, z, y, x] in one pass.
                stride_t = self.n_z * self.n_y * self.n_x
                stride_r = self.n_t * stride_t
                stride_z = self.n_y * self.n_x
                flat_idx = (
                        r_sel * stride_r
                        + t_sel * stride_t
                        + z_sel * stride_z
                        + y_sel * self.n_x
                        + x_sel
                )
                self.tensor.view(-1).scatter_add_(0, flat_idx, w * dt_seconds)
