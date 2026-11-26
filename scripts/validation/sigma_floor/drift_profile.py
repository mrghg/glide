"""Forward ½∂σ_w²/∂z near a stable h, per gradient mode / floor form, against
the exact derivative of the floored profile (extends the A6 review's
drift_profile.py). Env GLIDE_WT = the source tree to test; optional argv[1] =
SIGMA_MIN_M_S."""

import os
import sys

import torch

WT = os.environ["GLIDE_WT"]
sys.path.insert(0, WT + "/tests")
import lpdm  # noqa: E402

assert lpdm.__file__.startswith(WT), lpdm.__file__
import test_reciprocity as R  # noqa: E402

import lpdm.turbulence.hanna as H  # noqa: E402
from lpdm.gpu_engine import GPUEngine  # noqa: E402
from lpdm.turbulence.hanna import convective_velocity, obukhov_length  # noqa: E402

torch.set_num_threads(1)
if len(sys.argv) > 1:
    H.SIGMA_MIN_M_S = float(sys.argv[1])
met = R.reciprocity_met(*R.REGIMES["stable"])
h = 400.0
eng = GPUEngine(device="cpu")
d = [6, 4, 3, 2.5, 2.2, 2, 1.5, 1.3, 1.23, 1.2, 1.0, 0.5, 0.1, 0.0, -0.5, -2, -10]
z = torch.tensor([h - x for x in d], dtype=torch.float32)
n = z.numel()
xyz = torch.stack([torch.zeros(n), torch.full((n,), R.LAT0), z], 1)


def coeffs(mode, form):
    s = R.HannaScheme(static_substeps=False)
    s._forward_in_time = True
    s.DRIFT_GRADIENT_MODE = mode
    s.SIGMA_W_FLOOR_FORM = form
    lon, lat = xyz[:, 0], xyz[:, 1]

    def surf(k):
        return s._interp_surface_field(met, k, 0.5, lon, lat, eng.device, eng.dtype)

    blh = s._limit_blh(surf("blh"))
    sp = surf("sp")
    us = surf("ustar").clamp(min=1e-3)
    shf = surf("shf")
    ts = s._interp_t_at_lowest_level(met, 0.5, lon, lat, eng.device, eng.dtype)
    L = obukhov_length(us, shf, ts, sp)
    ws = convective_velocity(blh, shf, ts, sp, blh_min_m=s.blh_min_m)
    ft, gb = s._free_trop_fields(met, eng.device, eng.dtype)
    ck = dict(blh=blh, ustar=us, w_star=ws, h_over_L=blh / L, L=L, lat=lat, lon=lon,
              ft_fields=ft, grid_bounds=gb, engine=eng)  # fmt: skip
    c = s._vertical_coefficients(z, s._density_fields(met, eng.device, eng.dtype), **ck)
    return c[2], c[6], c[7], float(us[0])


sw, tl, g_fd, us = coeffs("fd", "clamp")
m = H.SIGMA_MIN_M_S
sh = 1.3 * us * (1 - z / h).clamp(min=0)
exact = torch.where(sh > m, -1.3 * us / h * sh, torch.zeros_like(sh))  # ½ d/dz max(σ_H, m)²
exact = torch.where(z > h, torch.full_like(z, float("nan")), exact)
rows = {"fd(prod)": g_fd}
for mode in ("kink", "analytic", "hybrid"):
    rows[mode] = coeffs(mode, "clamp")[2]
extra = {}
for form in ("quadrature", "offset"):
    extra["sw_" + form], extra["tl_" + form], rows[form + "_fd"], _ = coeffs("fd", form)
    rows[form + "_an"] = coeffs("analytic", form)[2]
print(f"SIGMA_MIN={m} u*={us} floor binds above h-{m * h / (1.3 * us):.3f} m")
keys = (
    "fd(prod)",
    "kink",
    "analytic",
    "hybrid",
    "quadrature_fd",
    "quadrature_an",
    "offset_fd",
    "offset_an",
)
print(f"{'h-z':>6} {'sig_w':>8} {'T_Lw':>7} {'exact':>10} " + " ".join(f"{k[:13]:>13}" for k in keys)
      + f" {'sw_quad':>8} {'TL_quad':>7} {'sw_off':>8}")  # fmt: skip
for i in range(n):
    print(f"{d[i]:6.2f} {float(sw[i]):8.5f} {float(tl[i]):7.0f} {float(exact[i]):10.3e} "
          + " ".join(f"{float(rows[k][i]):13.3e}" for k in keys)
          + f" {float(extra['sw_quadrature'][i]):8.5f} {float(extra['tl_quadrature'][i]):7.0f}"
          f" {float(extra['sw_offset'][i]):8.5f}")  # fmt: skip
print("drift columns: forward ½∂σ_w²/∂z (the 8th coefficient with _forward_in_time=True); "
      "exact: ½ d/dz max(σ_H, σ_min)², BL side only")  # fmt: skip
