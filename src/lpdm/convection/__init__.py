"""Deep-convection schemes.

Convection redistributes particles vertically through cumulus updrafts/downdrafts
in unstable columns. This is fundamentally different from the OU turbulence in
``lpdm.turbulence``: convection produces non-local mass jumps (a particle in the
BL can be lofted to the tropopause in one event), runs on the met-update cadence
(hourly) rather than the integration step (60–300 s), and operates per column
rather than per particle.

The architecture mirrors ``lpdm.turbulence``: a `ConvectionScheme` ABC + a
name-keyed registry. Schemes register themselves with `@register_scheme`. The
runtime calls `scheme.maybe_convect(...)` whenever the met window advances.

Schemes:
- `NoConvection` — bit-equivalent to no-convection runs (default).
- `EmanuelReducedConvection` — reduced port of FLEXPART's Emanuel & Živković-
  Rothman (1999) scheme per Forster et al. (2007), with the same buoyancy-
  sorting mass-flux matrix and probabilistic particle redistribution.

See `docs/convection.md` for the algorithm and validation notes.
"""

from __future__ import annotations

from lpdm.convection.base import (
	ConvectionScheme,
	ConvectionState,
	get_scheme,
	list_schemes,
	register_scheme,
)
from lpdm.convection.no_convection import NoConvection

# Triggers @register_scheme on import so `get_scheme("emanuel_reduced")` works.
from lpdm.convection.emanuel import EmanuelReducedConvection  # noqa: F401

__all__ = [
	"ConvectionScheme",
	"ConvectionState",
	"EmanuelReducedConvection",
	"NoConvection",
	"get_scheme",
	"list_schemes",
	"register_scheme",
]
