"""Abstract base class + registry for deep-convection schemes.

Convection is a separate runtime stage from advection and turbulence. It is
called once per met-update interval (typically hourly for ARCO ERA5) rather
than per integration step, and it operates per-column rather than per-particle
— a particle in a convectively active column can jump non-locally to a level
sampled from the convective updraft / downdraft profile.

For backward LPDMs, the same mass-flux matrix is applied as for forward — the
matrix is constructed so that probabilistic redistribution preserves the
well-mixed criterion in either time direction (Forster, Stohl & Seibert 2007,
§3 + Fig 2). FLEXPART's two modes (forward / backward) differ only in how the
matrix is sampled column-by-column, not in the underlying physics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeAlias

import torch

from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors


# Per-scheme state container (e.g. the previous step's cloud-base mass flux,
# which Emanuel's scheme uses as a closure variable). Concrete schemes pick
# whichever keys they need; `NoConvection` returns an empty dict.
ConvectionState: TypeAlias = dict[str, torch.Tensor]


class ConvectionScheme(ABC):
	"""Abstract interface for deep-convection parameterisations."""

	name: ClassVar[str]

	@abstractmethod
	def required_met_keys(self) -> tuple[str, ...]:
		"""Logical met-variable keys this scheme reads from `HourlyMetTensors`.

		Names must match keys in `ArcoEra5ZarrReader.DEFAULT_VARIABLE_MAP`. The
		runtime cross-checks these at startup so missing fields fail loud rather
		than silently defaulting.
		"""

	def initialize_state(
		self,
		n_particles: int,
		*,
		device: torch.device,
		dtype: torch.dtype,
	) -> ConvectionState:
		"""Allocate any persistent state. Default: empty (schemes without state)."""

		del n_particles, device, dtype  # unused for stateless schemes
		return {}

	@abstractmethod
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
		"""Possibly redistribute particles vertically in convectively active columns.

		Called once per met-update interval (the runtime fires this whenever the
		hourly met window advances, NOT every integration timestep). Implementations
		decide which columns are convectively active and randomly displace particles
		in those columns according to a per-column mass-flux matrix.

		Args:
			particles: (N, 4) tensor [lon, lat, alt, weight].
			state: per-scheme state from `initialize_state` (or `{}`).
			met_window: bracketing hourly met fields; convection uses the
				time-interpolated profile at ``t_alpha``.
			t_alpha: temporal interpolation weight in [0, 1] for met blending.
			dt_seconds: integration-loop dt — only used as the closure timescale
				`τ_conv` for converting the steady-state mass flux into a
				per-call redistribution probability. The actual cadence at which
				this method is called is the met-update interval.
			active_mask: (N,) bool. Particles outside this mask must be unchanged.
			engine: `GPUEngine` instance — exposed for future use (currently
				only the device/dtype are read).
			generator: optional `torch.Generator` for reproducible random draws.

		Returns:
			`(updated_particles, updated_state)`. Implementations may mutate
			inputs in place; callers must treat the return values as authoritative.
		"""


_REGISTRY: dict[str, type[ConvectionScheme]] = {}


def register_scheme(cls: type[ConvectionScheme]) -> type[ConvectionScheme]:
	"""Decorator: register a `ConvectionScheme` subclass by its `name` attribute."""

	if not getattr(cls, "name", ""):
		raise TypeError(f"{cls.__name__} must define a non-empty class-level `name` attribute")
	existing = _REGISTRY.get(cls.name)
	if existing is not None and existing is not cls:
		raise ValueError(
			f"Convection scheme name {cls.name!r} already registered to {existing.__name__}"
		)
	_REGISTRY[cls.name] = cls
	return cls


def get_scheme(name: str, **kwargs: object) -> ConvectionScheme:
	"""Construct a registered scheme by name. `kwargs` forward to the constructor."""

	if name not in _REGISTRY:
		available = ", ".join(sorted(_REGISTRY)) or "<none>"
		raise KeyError(f"Unknown convection scheme {name!r}. Registered: {available}")
	return _REGISTRY[name](**kwargs)


def list_schemes() -> tuple[str, ...]:
	"""Return the names of all registered schemes, sorted."""

	return tuple(sorted(_REGISTRY))
