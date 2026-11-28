"""Horizontal frame of the particle positions (task A4, roadmap item 7f).

The runtime holds each particle's horizontal position in a frame chosen once per
run and converts to longitude and latitude only where degrees are needed: at
release (in), at the footprint gridder and the endpoint and diagnostic outputs
(out), and for the two quantities that depend on latitude itself, the Coriolis
parameter and the length of a degree of longitude.

Two frames exist:

- ``local_metric`` (the runtime's frame): metres from the centre of the met
  domain, equirectangular with the x axis scaled by the cosine of the centre
  latitude::

      x = (lon - lon0) * 111320 cos(lat0)
      y = (lat - lat0) * 110540

  The constants are the metres per degree the model already uses (spherical
  Earth; ``vertical_grid`` and the advection before this frame existed), so a
  y metre is a model metre everywhere and an x metre is one at ``lat0``.
- ``degrees``: x = lon, y = lat, the positions as they were before this frame.
  Its arithmetic is main's to the bit; tests, scripts and anything that builds a
  ``GPUEngine`` without a frame get it.

Both are affine in (lon, lat), so the met grid, regular in lon and lat, is
regular in either frame, and interpolation needs only the grid's end points in
the frame (``frame_grid_bounds`` in ``gpu_engine``). The equation of motion is
the same in both: dx/dt = u / (metres per x unit at the particle's latitude),
dy/dt = v / (metres per y unit). In the metric frame these are
u cos(lat0)/cos(lat) and v. No map-projection error enters, at any latitude or
domain size; what changes is only how the position is rounded in float32
(``docs/physics.md`` §4 and dev/agent/progress/A4.md give the bounds).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch

# Metres per degree on the model's spherical Earth (unchanged from the advection
# that predates this module; also `vertical_grid._M_PER_DEG_*`).
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0
# |cos(lat)| floor of the longitude metric (about 87.1 degrees): unchanged.
COS_LAT_MIN = 0.05

FrameKind = Literal["degrees", "local_metric"]
_T = TypeVar("_T")


@dataclass(frozen=True)
class HorizontalFrame:
    """Affine horizontal frame: ``x = (lon - lon0) * kx``, ``y = (lat - lat0) * ky``.

    ``kx`` and ``ky`` are frame units per degree. All values are Python floats,
    constant for a run, so the compiled step specialises on them once
    (decision 0004).
    """

    kind: FrameKind = "degrees"
    lon0: float = 0.0
    lat0: float = 0.0
    kx: float = 1.0
    ky: float = 1.0

    @classmethod
    def local_metric(
        cls, lon_bounds: Sequence[float], lat_bounds: Sequence[float]
    ) -> HorizontalFrame:
        """Metres from the centre of the box (the met domain)."""

        lon0 = 0.5 * (float(lon_bounds[0]) + float(lon_bounds[1]))
        lat0 = 0.5 * (float(lat_bounds[0]) + float(lat_bounds[1]))
        cos0 = max(abs(math.cos(math.radians(lat0))), COS_LAT_MIN)
        return cls("local_metric", lon0, lat0, M_PER_DEG_LON_EQ * cos0, M_PER_DEG_LAT)

    @property
    def is_degrees(self) -> bool:
        return self.kind == "degrees"

    # -- lon/lat <-> frame. Work on tensors, numpy arrays and floats alike. --------

    def x_from_lon(self, lon: _T) -> _T:
        return lon if self.is_degrees else (lon - self.lon0) * self.kx  # type: ignore[operator, return-value]

    def y_from_lat(self, lat: _T) -> _T:
        return lat if self.is_degrees else (lat - self.lat0) * self.ky  # type: ignore[operator, return-value]

    def lon_from_x(self, x: _T) -> _T:
        return x if self.is_degrees else self.lon0 + x / self.kx  # type: ignore[operator, return-value]

    def lat_from_y(self, y: _T) -> _T:
        return y if self.is_degrees else self.lat0 + y / self.ky  # type: ignore[operator, return-value]

    def to_frame(self, particles: torch.Tensor) -> torch.Tensor:
        """Columns 0 and 1 from (lon, lat) to (x, y); the rest unchanged. Returns
        the input itself in the degrees frame, a new tensor otherwise."""

        if self.is_degrees:
            return particles
        out = particles.clone()
        out[:, 0] = self.x_from_lon(particles[:, 0])
        out[:, 1] = self.y_from_lat(particles[:, 1])
        return out

    def to_lonlat(self, particles: torch.Tensor) -> torch.Tensor:
        """Columns 0 and 1 from (x, y) to (lon, lat); the inverse of ``to_frame``."""

        if self.is_degrees:
            return particles
        out = particles.clone()
        out[:, 0] = self.lon_from_x(particles[:, 0])
        out[:, 1] = self.lat_from_y(particles[:, 1])
        return out

    # -- metric -------------------------------------------------------------------

    def metres_per_x(self, lat_deg: torch.Tensor) -> torch.Tensor:
        """Metres per x unit at latitude ``lat_deg``: an eastward displacement of d
        metres moves x by ``d / metres_per_x``. In the degrees frame this is the
        111320 |cos(lat)| expression the model has always used, to the bit."""

        cos_lat = torch.cos(torch.deg2rad(lat_deg)).abs().clamp(min=COS_LAT_MIN)
        if self.is_degrees:
            return M_PER_DEG_LON_EQ * cos_lat
        return cos_lat * (M_PER_DEG_LON_EQ / self.kx)

    @property
    def metres_per_y(self) -> float:
        """Metres per y unit (constant): 110540 in degrees, 1 in the metric frame."""

        return M_PER_DEG_LAT / self.ky if not self.is_degrees else M_PER_DEG_LAT


DEGREES = HorizontalFrame()
