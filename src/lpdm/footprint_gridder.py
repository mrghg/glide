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
