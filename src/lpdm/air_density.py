"""Dry-air density at each particle, for weighting the footprint (decision 0022).

The footprint of `docs/reference-runs.md` §6 divides each particle's time in a
cell by the dry-air density at the particle as it is counted:

    rho_d = (p - e) / (R_d T),    e = q p / (0.622 + 0.378 q)

Here p, T and q come from the step's met window, at the step's time weight
(the same window and weight as the step: `main._step_window_time` and
`main._advection_alpha_wind_mean`), interpolated to the particle's position.

**Pressure at the particle.** The met window carries temperature and specific
humidity on the terrain-following AGL grid and the surface pressure, but not a
3D pressure field. The pressure at the particle's height z above the ground is
the surface pressure carried up by the hypsometric relation,

    p(z) = p_s exp(-g z / (R_d Tv_bar)),   Tv = T (1 + 0.6077 q),

with the layer-mean virtual temperature Tv_bar taken at z/2 (midpoint rule).
This is the same hydrostatic relation, with the same constants, that GLIDE uses
to reconstruct model-level pressure from the surface pressure (decision 0010,
`vertical_grid.model_level_pressure_pa`), applied from the ground to the
particle instead of level to level. The midpoint rule's error in p is below
1e-7 over the 40 m footprint layer, even in a strong surface inversion, and
about 1e-5 at 1 km for a standard lapse rate. Below the lowest level of the
AGL grid (10 m by default) T and q are held at that level's values.

The three fields are sampled from one [6, Z, Y, X] stack (T, q, p_s at the
window's start and end; p_s is z-broadcast like every surface channel) with
the same trilinear interpolation and vertical lookup as the wind. The stack is
moved to the device once per met window. Per step everything is tensor
operations on the full particle buffer: no host sync, constant shapes, and the
time weight held in a 0-d device buffer rather than a Python scalar
(decision 0004).
"""

from __future__ import annotations

import numpy as np
import torch

from lpdm.gpu_engine import GPUEngine, GridInterpolationBounds
from lpdm.met_reader import HourlyMetTensors

GRAVITY_M_S2 = 9.80665
R_DRY_AIR_J_KG_K = 287.05
# e = q p / (EPS + (1 - EPS) q), EPS = R_d / R_v.
EPS_RD_RV = 0.622
# Tv = T (1 + TV_COEFF q); as `vertical_grid._TV_COEFF` (1/0.622 - 1).
TV_COEFF = 0.6077

#: Footprint attributes for each weighting (store attrs and run_metadata.json).
DENSITY_WEIGHTING_ATTRS: dict[str, dict[str, str]] = {
    "dry_air": {
        "density_weighting": "dry_air",
        "units": "s m3 kg-1",
        "units_long": (
            "particle weight x residence seconds / dry-air density at the particle "
            "(s m3 kg-1 of dry air per unit released mass)"
        ),
        "density_formula": (
            "rho_d = (p - e) / (R_d T), e = q p / (0.622 + 0.378 q), R_d = 287.05 J kg-1 K-1; "
            "p = p_s exp(-g z / (R_d Tv(z/2))), Tv = T (1 + 0.6077 q); T, q, p_s "
            "interpolated to the particle in the step's met window and time weight"
        ),
        "to_surface_sensitivity": (
            "M_air / h x (sum over the 0-h m layer) gives m2 s mol-1 of dry air "
            "(M_air = 0.02897 kg mol-1); apply no further density"
        ),
    },
    "none": {
        "density_weighting": "none",
        "units": "s",
        "units_long": "particle weight x residence seconds (per unit released mass)",
        "density_formula": "none",
        "to_surface_sensitivity": (
            "M_air / (h rho) x (sum over the 0-h m layer); rho applied afterwards, "
            "which is biased where residence and density co-vary (decision 0022)"
        ),
    },
}


def vapour_pressure_pa(p_pa, q):
    """Water-vapour partial pressure from pressure and specific humidity."""

    return q * p_pa / (EPS_RD_RV + (1.0 - EPS_RD_RV) * q)


def dry_air_density_kg_m3(p_pa, t_k, q):
    """Dry-air density (p - e) / (R_d T). Works on tensors, arrays and xarray."""

    return (p_pa - vapour_pressure_pa(p_pa, q)) / (R_DRY_AIR_J_KG_K * t_k)


class ParticleDryAirDensity:
    """Per-particle dry-air density from the step's met window.

    One instance per run. ``inverse_density`` returns ``1 / rho_d`` for every
    particle in the buffer (inactive ones included; the gridder zeroes them).
    """

    CHANNELS: tuple[str, ...] = ("t", "q", "sp")
    # Surface channels: the same value at every level of the window.
    SURFACE_CHANNELS: frozenset[str] = frozenset({"sp"})

    def __init__(
        self,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        *,
        compile_core: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        # On CUDA with GLIDE_COMPILE, fuse the ~60 small per-step kernels of the
        # core into a few (the static path is launch-bound, decision 0004). Every
        # per-step and per-window value enters as a tensor, so it compiles once.
        self._core = self.inverse_density_from_stack
        if compile_core and self.device.type == "cuda":
            self._core = torch.compile(self.inverse_density_from_stack, dynamic=False)
        self._window_key = None
        self._stack: torch.Tensor | None = None
        self._level_arr: torch.Tensor | None = None
        self._ascending: bool = True
        self._bounds: GridInterpolationBounds | None = None
        self._period_mask: torch.Tensor | None = None
        # Time weight of the window end, refilled each step (0-d, device).
        self._alpha = torch.zeros((), device=self.device, dtype=self.dtype)

    def _refresh(self, met_window: HourlyMetTensors) -> None:
        key = met_window.metadata.time_start
        if self._window_key == key and self._stack is not None:
            return
        # Copy each channel slice straight to the device (no host-side gather or
        # concatenation of the window; the met cache usually lives on the host).
        # Surface channels are z-broadcast in the window, so only their lowest
        # level crosses, and is broadcast again on the device.
        n_ch = len(self.CHANNELS)
        first = met_window.channel(self.CHANNELS[0])[0]
        shape = (2 * n_ch, *first.shape)
        if self._stack is None or tuple(self._stack.shape) != shape:
            self._stack = torch.empty(shape, device=self.device, dtype=self.dtype)
        for i, name in enumerate(self.CHANNELS):
            for j, half in enumerate(met_window.channel(name)):
                dst = self._stack[i + j * n_ch]
                if name in self.SURFACE_CHANNELS:
                    dst.copy_(half[0].to(device=self.device, dtype=self.dtype).expand_as(dst))
                else:
                    dst.copy_(half)
        level = np.asarray(met_window.metadata.level, dtype=np.float64)
        self._level_arr = torch.as_tensor(level, device=self.device, dtype=self.dtype)
        self._ascending = bool(level[-1] > level[0])
        lon = met_window.metadata.lon
        lat = met_window.metadata.lat
        self._bounds = GridInterpolationBounds(
            lon_first=float(lon[0]),
            lon_last=float(lon[-1]),
            lat_first=float(lat[0]),
            lat_last=float(lat[-1]),
            alt_first=float(level[0]),
            alt_last=float(level[-1]),
            level_agl_m=tuple(float(v) for v in level),
        )
        # A mean/accumulation channel (decision 0020) is held at its window-end
        # value: weight 1 instead of alpha.
        period = getattr(met_window, "period_channels", frozenset())
        self._period_mask = torch.tensor(
            [1.0 if name in period else 0.0 for name in self.CHANNELS],
            device=self.device,
            dtype=self.dtype,
        ).view(-1, 1)
        self._window_key = key

    def inverse_density(
        self,
        particles_xyz: torch.Tensor,
        met_window: HourlyMetTensors,
        t_alpha: float,
        engine: GPUEngine,
    ) -> torch.Tensor:
        """``1 / rho_d`` (m3 kg-1) at each particle; ``particles_xyz`` is [N, 3]."""

        self._refresh(met_window)
        # fill_ with a host scalar launches a fill; it does not sync the device.
        self._alpha.fill_(float(t_alpha))
        return self._core(
            particles_xyz,
            self._stack,
            self._alpha,
            self._period_mask,
            self._bounds,
            self._level_arr,
            self._ascending,
            engine,
        )

    @staticmethod
    def inverse_density_from_stack(
        particles_xyz: torch.Tensor,
        stack: torch.Tensor,
        alpha: torch.Tensor,
        period_mask: torch.Tensor,
        bounds: GridInterpolationBounds,
        level_arr: torch.Tensor,
        ascending: bool,
        engine: GPUEngine,
    ) -> torch.Tensor:
        """Pure-tensor core: sample, time-interpolate, and form ``1 / rho_d``."""

        n = particles_xyz.shape[0]
        z = particles_xyz[:, 2].clamp(min=0.0)
        # Sample at the particle (T, q, p_s) and at half its height (layer-mean Tv).
        half = torch.stack([particles_xyz[:, 0], particles_xyz[:, 1], 0.5 * z], dim=1)
        pts = torch.cat([particles_xyz, half], dim=0)
        norm = engine.normalize_particle_coordinates(
            pts, bounds, level_arr=level_arr, ascending=ascending
        )
        grid = norm.view(1, 1, 1, -1, 3).to(dtype=stack.dtype)
        sampled = torch.nn.functional.grid_sample(
            stack.unsqueeze(0), grid, align_corners=True
        ).view(stack.shape[0], -1)
        n_ch = stack.shape[0] // 2
        s, e = sampled[:n_ch], sampled[n_ch:]
        w = alpha + period_mask * (1.0 - alpha)  # [3, 1]
        vals = s + (e - s) * w
        t_p, q_p, ps = vals[0, :n], vals[1, :n], vals[2, :n]
        t_h, q_h = vals[0, n:], vals[1, n:]
        tv_half = (t_h * (1.0 + TV_COEFF * q_h)).clamp(min=1.0)
        p = ps * torch.exp(-GRAVITY_M_S2 * z / (R_DRY_AIR_J_KG_K * tv_half))
        rho = dry_air_density_kg_m3(p, t_p.clamp(min=1.0), q_p)
        return 1.0 / rho.clamp(min=1e-6)
