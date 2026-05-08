"""Abstract base class and registry for turbulence parameterization schemes.

Schemes implement met-driven turbulent dispersion: per-particle sigma^2 and Lagrangian
timescale T_L from the local meteorology, the OU update of perturbation velocities,
the geometric application of those perturbations, and any boundary conditions tied
to vertical turbulence (e.g. surface reflection).

The runtime loop owns mean-wind advection; the scheme runs on each step after
advection. See `docs/turbulence.md` for the architecture rationale and per-scheme
math.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeAlias

import torch

from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors


TurbulenceState: TypeAlias = dict[str, torch.Tensor]


class TurbulenceScheme(ABC):
	"""Abstract interface for turbulence parameterizations."""

	name: ClassVar[str]

	@abstractmethod
	def required_met_keys(self) -> tuple[str, ...]:
		"""Logical met-variable keys this scheme reads from `HourlyMetTensors`.

		Names must match keys in `ArcoEra5ZarrReader.DEFAULT_VARIABLE_MAP`. The runtime
		cross-checks these at startup so missing fields fail loud rather than silently
		defaulting.
		"""

	@abstractmethod
	def initialize_state(
		self,
		n_particles: int,
		*,
		device: torch.device,
		dtype: torch.dtype,
	) -> TurbulenceState:
		"""Allocate per-particle state tensors for this scheme."""

	@abstractmethod
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
		"""Apply one turbulence step to the particle cloud.

		Args:
			particles: (N, 4) tensor [lon, lat, alt, weight] for the entire ensemble.
			state: per-particle turbulence state (e.g. perturbation velocities).
			met_window: bracketing hourly met fields for temporal interpolation.
			t_alpha: temporal interpolation weight in [0, 1] for met blending.
			dt_seconds: integration timestep length.
			active_mask: (N,) bool. Inactive particles must be left unmodified.
			engine: `GPUEngine` instance for primitives (OU update, displacement,
				surface reflection).

		Returns:
			(updated_particles, updated_state). Implementations may mutate inputs in
			place; callers must treat the return values as authoritative.
		"""


_REGISTRY: dict[str, type[TurbulenceScheme]] = {}


def register_scheme(cls: type[TurbulenceScheme]) -> type[TurbulenceScheme]:
	"""Decorator: register a `TurbulenceScheme` subclass by its `name` class attribute."""

	if not getattr(cls, "name", ""):
		raise TypeError(f"{cls.__name__} must define a non-empty class-level `name` attribute")
	existing = _REGISTRY.get(cls.name)
	if existing is not None and existing is not cls:
		raise ValueError(
			f"Turbulence scheme name {cls.name!r} already registered to {existing.__name__}"
		)
	_REGISTRY[cls.name] = cls
	return cls


def get_scheme(name: str, **kwargs: object) -> TurbulenceScheme:
	"""Construct a registered scheme by name. `kwargs` are forwarded to the constructor."""

	if name not in _REGISTRY:
		available = ", ".join(sorted(_REGISTRY)) or "<none>"
		raise KeyError(f"Unknown turbulence scheme {name!r}. Registered schemes: {available}")
	return _REGISTRY[name](**kwargs)


def list_schemes() -> tuple[str, ...]:
	"""Return the names of all registered schemes, sorted."""

	return tuple(sorted(_REGISTRY))
