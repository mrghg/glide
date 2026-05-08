"""Placeholder constant-OU turbulence scheme.

Reproduces the M0 placeholder behaviour exactly: vertical OU with hard-coded
`T_L = 300 s`, `sigma_w^2 = 1.0 m^2/s^2`, no horizontal turbulence. Kept as a
registered scheme for regression testing and reproducing M0 baselines while the
Hanna scheme is being validated.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors
from lpdm.turbulence.base import TurbulenceScheme, TurbulenceState, register_scheme


@register_scheme
class PlaceholderConstantOU(TurbulenceScheme):
	"""Constant-T_L, constant-sigma vertical OU. Matches the M0 placeholder behaviour."""

	name: ClassVar[str] = "placeholder_constant_ou"

	t_lagrangian_s: float = 300.0
	sigma_w2: float = 1.0

	def required_met_keys(self) -> tuple[str, ...]:
		return ()

	def initialize_state(
		self,
		n_particles: int,
		*,
		device: torch.device,
		dtype: torch.dtype,
	) -> TurbulenceState:
		return {"w_prime": torch.zeros(n_particles, device=device, dtype=dtype)}

	def step(
		self,
		particles: torch.Tensor,
		state: TurbulenceState,
		met_window: HourlyMetTensors,
		t_alpha: float,
		dt_seconds: float,
		active_mask: torch.Tensor,
		engine: GPUEngine,
	) -> tuple[torch.Tensor, TurbulenceState]:
		del met_window, t_alpha  # placeholder is met-independent

		if not bool(torch.any(active_mask)):
			return particles, state

		w_prime = state["w_prime"]
		w_prime_active = engine.update_langevin_velocity(
			w_prime[active_mask],
			t_lagrangian=self.t_lagrangian_s,
			sigma_w2=self.sigma_w2,
			dt_seconds=dt_seconds,
		)

		active_particles_out = engine.apply_vertical_turbulence(
			particles[active_mask],
			w_prime_active,
			dt_seconds=dt_seconds,
			backward=True,
		)
		active_particles_out = engine.reflect_surface(active_particles_out, z_surface=0.0)

		particles[active_mask] = active_particles_out
		w_prime[active_mask] = w_prime_active

		return particles, state
