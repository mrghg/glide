"""NoConvection — default scheme that does nothing.

Use this when convection is disabled (the default) or for regression baselines
where bit-equivalence with the no-convection runtime path matters.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from lpdm.convection.base import (
	ConvectionScheme,
	ConvectionState,
	register_scheme,
)
from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors


@register_scheme
class NoConvection(ConvectionScheme):
	"""Pass-through scheme: particles are unchanged."""

	name: ClassVar[str] = "none"

	def required_met_keys(self) -> tuple[str, ...]:
		return ()

	def maybe_convect(
		self,
		particles: torch.Tensor,
		state: ConvectionState,
		met_window: HourlyMetTensors,
		*,
		t_alpha: float,
		dt_seconds: float,
		active_mask: torch.Tensor,
		engine: GPUEngine,
		generator: torch.Generator | None = None,
	) -> tuple[torch.Tensor, ConvectionState]:
		del met_window, t_alpha, dt_seconds, active_mask, engine, generator
		return particles, state
