"""Orientation of the horizontal turbulent velocities (decision 0032).

Hanna's σ_u, T_Lu are along the mean wind and σ_v, T_Lv across it. With
``turbulence.horizontal_frame: grid`` GLIDE applies u' eastward and v' northward;
with ``wind`` the OU updates run on along-wind and cross-wind components, which
are rotated to east/north for each displacement by the mean wind direction at the
particle, and the carried state stays east/north.

The plume-width tests use the dynamic (CPU) step path, which does not advect (the
runtime advects before calling the scheme), so the displacements are turbulent
only while the wind still sets the frame.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch
from pydantic import ValidationError

import lpdm.turbulence.hanna as H
from lpdm.gpu_engine import GPUEngine
from lpdm.local_frame import DEGREES, HorizontalFrame
from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata
from lpdm.turbulence import HannaScheme

M_LAT = 110540.0  # metres per degree of latitude (GPUEngine)
M_LON = 111320.0  # metres per degree of longitude at the equator (GPUEngine)
STABLE = (400.0, 0.3, -20.0)  # h (m), u* (m/s), sensible heat flux (W m-2)
NEUTRAL = (300.0, 0.4, 0.0)
CONVECTIVE = (1500.0, 0.3, 200.0)


def _met(
    u: torch.Tensor | float,
    v: torch.Tensor | float,
    *,
    u_end: torch.Tensor | float | None = None,
    v_end: torch.Tensor | float | None = None,
    regime: tuple[float, float, float] = STABLE,
    half_deg: float = 2.0,
    n: int = 5,
) -> HourlyMetTensors:
    """Horizontally uniform turbulence on an ``n``×``n`` grid spanning ±``half_deg``;
    the wind (u, v) at the window start and (u_end, v_end) at its end, each a
    scalar or a [Y, X] field, the same at every level."""

    blh, ustar, shf = regime
    n_lev = 11
    shape = (n_lev, n, n)
    level = np.linspace(0.0, 4000.0, n_lev)

    def field(x: torch.Tensor | float) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32).expand(shape).contiguous()

    def stack(uu, vv) -> torch.Tensor:
        ch = {
            "u": field(uu),
            "v": field(vv),
            "w": field(0.0),
            "blh": field(blh),
            "sp": field(101325.0),
            "t": field(280.0),
            "ustar": field(ustar),
            "shf": field(shf),
        }
        return torch.stack(list(ch.values())), tuple(ch)

    start, names = stack(u, v)
    end, _ = stack(u if u_end is None else u_end, v if v_end is None else v_end)
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    md = MetFieldMetadata(
        lon=np.linspace(-half_deg, half_deg, n),
        lat=np.linspace(-half_deg, half_deg, n),
        level=level,
        pressure_level_hpa=np.linspace(1000.0, 600.0, n_lev),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units=dict.fromkeys(names, "1"),
    )
    height = torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape)
    return HourlyMetTensors(
        hour_start=start,
        hour_end=end,
        metadata=md,
        channel_names=names,
        height_agl_m=height.contiguous(),
    )


def _particles(n: int, z: float | torch.Tensor) -> torch.Tensor:
    p = torch.zeros(n, 4, dtype=torch.float32)
    p[:, 2] = z
    p[:, 3] = 1.0
    return p


def _run(scheme, met, particles, *, steps, dt=60.0, t_alpha=0.5, seed=0, record=None, state=None):
    """``steps`` scheme steps, all particles active; ``t_alpha`` a float or a
    function of the step index. ``record(step, particles, state)`` after each step.
    ``state`` defaults to the scheme's initial (zero-velocity) state."""

    engine = GPUEngine(device="cpu")
    torch.manual_seed(seed)
    n = particles.shape[0]
    if state is None:
        state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)
    for k in range(steps):
        a = t_alpha(k) if callable(t_alpha) else t_alpha
        particles, state = scheme.step(
            particles, state, met, t_alpha=a, dt_seconds=dt, active_mask=active, engine=engine
        )
        if record is not None:
            record(k, particles, state)
    return particles, state


def _metres(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """East and north positions (m) relative to (0, 0), in float64."""

    lat = p[:, 1].double()
    return p[:, 0].double() * M_LON * torch.cos(torch.deg2rad(lat)), lat * M_LAT


# --- switch, helpers ---------------------------------------------------------------


def test_horizontal_frame_default_is_grid_and_is_validated() -> None:
    from lpdm.config import TurbulenceConfig

    assert HannaScheme().horizontal_frame == "grid"
    assert HannaScheme(horizontal_frame="wind").horizontal_frame == "wind"
    with pytest.raises(ValueError, match="horizontal_frame"):
        HannaScheme(horizontal_frame="along")
    assert TurbulenceConfig(scheme="hanna_1982").horizontal_frame == "grid"
    assert TurbulenceConfig(scheme="hanna_1982", horizontal_frame="wind").horizontal_frame == "wind"
    with pytest.raises(ValidationError):
        TurbulenceConfig(scheme="hanna_1982", horizontal_frame="along")


def test_rotation_helpers_invert_and_are_the_identity_without_axes() -> None:
    torch.manual_seed(1)
    u, v = torch.randn(1000), torch.randn(1000)
    phi = torch.rand(1000) * 2 * math.pi
    axes = torch.stack([torch.cos(phi), torch.sin(phi)])
    a, c = HannaScheme._grid_to_wind(u, v, axes)
    # along = projection on the wind; cross = 90° to its left (FLEXPART windalign).
    torch.testing.assert_close(a, u * axes[0] + v * axes[1])
    torch.testing.assert_close(c, -u * axes[1] + v * axes[0])
    uu, vv = HannaScheme._wind_to_grid(a, c, axes)
    torch.testing.assert_close(uu, u, rtol=0, atol=1e-6)
    torch.testing.assert_close(vv, v, rtol=0, atol=1e-6)
    # No axes (the grid frame): the very same tensors, so the grid path is untouched.
    assert HannaScheme._grid_to_wind(u, v, None)[0] is u
    assert HannaScheme._wind_to_grid(u, v, None)[1] is v


def test_wind_axes_follow_the_wind_and_fall_back_to_the_grid_when_calm() -> None:
    """Unit vector of the mean wind where the speed exceeds σ_u; (1, 0) at or below."""

    scheme = HannaScheme(horizontal_frame="wind")
    engine = GPUEngine(device="cpu")
    speeds = torch.tensor([5.0, 5.0, 0.5, 0.3, 0.0])
    angles = torch.tensor([30.0, -120.0, 200.0, 45.0, 0.0])
    sigma_u = torch.full((5,), 0.4)
    axes = []
    for s, a in zip(speeds.tolist(), angles.tolist(), strict=True):
        met = _met(s * math.cos(math.radians(a)), s * math.sin(math.radians(a)))
        uv_s, uv_e = met.channels("u", "v")
        _, bounds = scheme._free_trop_fields(met, torch.device("cpu"), torch.float32, DEGREES)
        axes.append(
            scheme._wind_axes(
                uv_s,
                uv_e,
                0.5,
                sigma_u[:1],
                x=torch.zeros(1),
                y=torch.zeros(1),
                z=torch.full((1,), 50.0),
                grid_bounds=bounds,
                engine=engine,
            )[:, 0]
        )
    axes = torch.stack(axes, dim=1)
    rad = torch.deg2rad(angles[:3])
    torch.testing.assert_close(axes[:, :3], torch.stack([torch.cos(rad), torch.sin(rad)]))
    assert torch.equal(axes[:, 3:], torch.tensor([[1.0, 1.0], [0.0, 0.0]]))


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("wind", [(5.0, 0.0), (0.01, 0.005), (0.0, 0.0)])
def test_wind_frame_is_bit_identical_to_grid_for_an_eastward_or_calm_wind(
    static: bool, wind: tuple[float, float]
) -> None:
    """With the axes (1, 0), an eastward wind or one at or below σ_u, the wind frame
    does exactly what the grid frame does (both step paths). The particles start well
    below the stable top, where σ_u ≥ 0.2 m/s; near h, σ_u → 0 and even 0.01 m/s
    counts as wind."""

    met = _met(*wind)
    z = torch.linspace(5.0, 250.0, 512)
    out = {}
    for frame in ("grid", "wind"):
        scheme = HannaScheme(horizontal_frame=frame, static_substeps=static)
        p, st = _run(scheme, met, _particles(512, z), steps=4, seed=7)
        out[frame] = (p, st["u_prime"], st["v_prime"], st["w_prime"])
    for g, w in zip(out["grid"], out["wind"], strict=True):
        assert torch.equal(g, w)


@pytest.mark.parametrize(
    "metric, wind_dtype",
    [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)],
    ids=["degrees", "metric", "metric-bf16"],
)
def test_static_and_dynamic_paths_agree_in_the_wind_frame(
    metric: bool, wind_dtype: torch.dtype
) -> None:
    """The static path advects inside the step; the dynamic path is given the same
    advected particles. With every particle active the two are bit-identical, in the
    degrees frame and in the runtime's local metric frame with float32 or bfloat16
    winds (task A4)."""

    met = _met(3.0, -4.0, u_end=-2.0, v_end=1.0)
    z = torch.linspace(5.0, 390.0, 256)
    frame = HorizontalFrame.local_metric((-2.0, 2.0), (-2.0, 2.0)) if metric else DEGREES
    engine = GPUEngine(device="cpu", frame=frame, wind_gather_dtype=wind_dtype)
    p0 = _particles(256, z)
    p0[:, 0] = torch.linspace(-0.5, 0.5, 256)
    p0 = frame.to_frame(p0)
    t_alpha, dt = 0.3, 60.0

    static = HannaScheme(horizontal_frame="wind", static_substeps=True)
    dynamic = HannaScheme(horizontal_frame="wind", static_substeps=False)
    st_s = static.initialize_state(256, device=torch.device("cpu"), dtype=torch.float32)
    st_d = dynamic.initialize_state(256, device=torch.device("cpu"), dtype=torch.float32)
    p_s, p_d = p0.clone(), p0.clone()
    active = torch.ones(256, dtype=torch.bool)
    for k in range(3):
        inputs = static._gather_static_inputs(
            met,
            t_alpha,
            torch.device("cpu"),
            torch.float32,
            frame=frame,
            wind_dtype=engine.wind_dtype,
        )
        advected = static._advect_rk2(
            p_d,
            inputs["m_start_uvw"],
            inputs["m_end_uvw"],
            inputs["alpha"],
            inputs["grid_bounds"],
            dt,
            engine,
            level_arr=inputs["level_arr"],
            ascending=inputs["ascending"],
        )
        torch.manual_seed(100 + k)
        p_s, st_s = static.step(p_s, st_s, met, t_alpha, dt, active, engine)
        torch.manual_seed(100 + k)
        p_d, st_d = dynamic.step(advected, st_d, met, t_alpha, dt, active, engine)
    assert torch.equal(p_s, p_d)
    for key in ("u_prime", "v_prime", "w_prime", "sigma_last"):
        assert torch.equal(st_s[key], st_d[key]), key


# --- the review's test: plume widths in wind-relative coordinates ----------------------

N_PLUME = 20_000
# 20,000 particles: the relative standard error of a plume's standard deviation is
# 1/√(2n) = 0.5%, so of the ratio of two independent ones 0.7%. 4% is ~5.7 SE.
WIDTH_RTOL = 0.04


def _plume_widths(frame: str, theta_deg: float, regime, *, seed: int, z0: float):
    """Along-wind and cross-wind standard deviations (m) of a plume from a point,
    after 10 min, the mean wind (5 m/s) blowing towards ``theta_deg`` (0 = east)."""

    th = math.radians(theta_deg)
    met = _met(5.0 * math.cos(th), 5.0 * math.sin(th), regime=regime)
    scheme = HannaScheme(horizontal_frame=frame, static_substeps=False)
    p, _ = _run(scheme, met, _particles(N_PLUME, z0), steps=10, seed=seed)
    x, y = _metres(p)
    along = x * math.cos(th) + y * math.sin(th)
    cross = -x * math.sin(th) + y * math.cos(th)
    return float(along.std()), float(cross.std())


def test_stable_plume_widths_rotate_with_the_wind_only_in_the_wind_frame() -> None:
    """The review's paired experiment: flat, uniform meteorology, meander off, the
    mean wind rotated by 45° and 90°. In the stable boundary layer Hanna's along-wind
    spread exceeds the cross-wind one (K_u/K_v ≈ 3.3, widths ≈ 1.75:1). The wind
    frame gives the same along-wind and cross-wind widths whatever the direction; the
    grid frame swaps them at 90° and makes the plume round at 45°."""

    widths = {
        (frame, th): _plume_widths(frame, th, STABLE, seed=11 + i, z0=100.0)
        for frame in ("grid", "wind")
        for i, th in enumerate((0.0, 45.0, 90.0))
    }
    ref_a, ref_c = widths[("wind", 0.0)]
    assert ref_a / ref_c > 1.5  # the anisotropy the test relies on
    for th in (45.0, 90.0):
        a, c = widths[("wind", th)]
        assert a == pytest.approx(ref_a, rel=WIDTH_RTOL), (th, a, ref_a)
        assert c == pytest.approx(ref_c, rel=WIDTH_RTOL), (th, c, ref_c)
    # Grid frame: east-west is always the "along-wind" component.
    assert widths[("grid", 0.0)] == pytest.approx(widths[("wind", 0.0)], rel=WIDTH_RTOL)
    a45, c45 = widths[("grid", 45.0)]
    assert a45 / c45 == pytest.approx(1.0, rel=WIDTH_RTOL)
    a90, c90 = widths[("grid", 90.0)]
    assert a90 / c90 == pytest.approx(ref_c / ref_a, rel=WIDTH_RTOL)


def test_isotropic_regimes_do_not_depend_on_the_frame() -> None:
    """Where σ_u = σ_v and T_Lu = T_Lv (the convective boundary layer; the free
    troposphere above h) the frame changes the random draws, not the statistics:
    east and north widths agree between the frames within sampling noise."""

    th = math.radians(30.0)
    for regime, z0 in ((CONVECTIVE, 500.0), (STABLE, 2500.0)):
        met = _met(5.0 * math.cos(th), 5.0 * math.sin(th), regime=regime)
        widths = {}
        for i, frame in enumerate(("grid", "wind")):
            scheme = HannaScheme(horizontal_frame=frame, static_substeps=False)
            p, _ = _run(scheme, met, _particles(N_PLUME, z0), steps=10, seed=21 + i)
            x, y = _metres(p)
            widths[frame] = (float(x.std()), float(y.std()))
        assert widths["wind"][0] == pytest.approx(widths["grid"][0], rel=WIDTH_RTOL)
        assert widths["wind"][1] == pytest.approx(widths["grid"][1], rel=WIDTH_RTOL)
        assert widths["grid"][0] == pytest.approx(widths["grid"][1], rel=WIDTH_RTOL)


# --- a turning wind, through calm -------------------------------------------------------


@pytest.mark.parametrize("carry_wind_relative", [False, True])
def test_wind_turning_through_calm_keeps_velocity_statistics_and_adds_no_drift(
    carry_wind_relative: bool,
) -> None:
    """Uniform stable turbulence; over one hour the mean wind turns from east
    (5 m/s) through calm (at 30 min) to west, half-turning within ~3 min. Particles
    start in equilibrium (velocities drawn from the Eulerian distribution). Checks
    in the wind frame at each step's end, on the particles below 300 m (0.75 h):

    - production (carried state east/north, decision 0032): the normalised
      horizontal velocity variance ½(⟨a²/σ_u²⟩ + ⟨c²/σ_v²⟩) stays within
      [1, ½(k + 1/k)], k = σ_u²/σ_v² (derived: the range a rotation of the carried
      east/north velocity can produce), ± 5 standard errors; a turbulent velocity
      lost or doubled at the calm switch would breach it. Each component is back
      to 1 within 0.05 once the wind has stopped turning quickly (from 45 min);
    - carried wind-relative (FLEXPART, the private A/B switch): each component
      stays at 1 within 0.05 throughout, as it must (derived: the wind-frame
      distribution does not change when the frame turns);
    - both: the plume centre does not move (no spurious mean drift; 5 SE).

    Above ~0.75 h the normalised variance drifts upwards in a steady wind too, in
    either frame (σ_u, σ_v fall towards h and T_Lu reaches ~30 min there; u′ and v′
    have no well-mixed correction for the vertical gradient of σ_u and σ_v, as in
    FLEXPART), so those particles are left out of the velocity checks."""

    n, steps = 20_000, 60
    met = _met(5.0, 0.0, u_end=-5.0, v_end=0.6)
    scheme = HannaScheme(horizontal_frame="wind", static_substeps=False)
    scheme._carry_wind_relative = carry_wind_relative
    z0 = torch.linspace(20.0, 380.0, n)
    blh, ustar, shf = STABLE
    one = torch.ones(n)
    obukhov = H.obukhov_length(ustar * one, shf * one, 280.0 * one, 101325.0 * one)
    su, sv, sw, *_ = H.in_bl_sigma_TL(z0, blh * one, ustar * one, 0 * one, blh / obukhov, 0 * one)
    sig0 = torch.stack([su, sv, sw], dim=1).clamp(min=H.SIGMA_MIN_M_S)
    gen = torch.Generator().manual_seed(3)
    state0 = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    state0["sigma_last"] = sig0
    for i, key in enumerate(("u_prime", "v_prime", "w_prime")):
        state0[key] = sig0[:, i] * torch.randn(n, generator=gen)
    k = (2.0 / 1.3) ** 2
    rows = []

    def record(step, p, state):
        alpha = (step + 0.5) / steps
        u = 5.0 * (1 - alpha) - 5.0 * alpha
        v = 0.6 * alpha
        sig = state["sigma_last"]
        speed = math.hypot(u, v)
        windy = sig[:, 0] < speed
        c = torch.where(windy, u / speed, 1.0)
        s = torch.where(windy, v / speed, 0.0)
        if carry_wind_relative:
            a, x = state["u_prime"], state["v_prime"]
        else:
            a, x = HannaScheme._grid_to_wind(
                state["u_prime"], state["v_prime"], torch.stack([c, s])
            )
        low = p[:, 2] <= 300.0
        va = float((a / sig[:, 0]).pow(2)[low].mean())
        vc = float((x / sig[:, 1]).pow(2)[low].mean())
        rows.append((step, va, vc, math.sqrt(2.0 / int(low.sum()))))
        assert torch.isfinite(p).all() and torch.isfinite(state["u_prime"]).all()

    p, _ = _run(
        scheme,
        met,
        _particles(n, z0),
        steps=steps,
        t_alpha=lambda i: (i + 0.5) / steps,
        record=record,
        state=state0,
    )
    for step, va, vc, se in rows:
        if carry_wind_relative:
            assert va == pytest.approx(1.0, abs=0.05), (step, va)
            assert vc == pytest.approx(1.0, abs=0.05), (step, vc)
        else:
            total = 0.5 * (va + vc)
            assert 1.0 - 5 * se < total < 0.5 * (k + 1 / k) + 5 * se, (step, va, vc)
            if step >= 45:
                assert va == pytest.approx(1.0, abs=0.05), (step, va)
                assert vc == pytest.approx(1.0, abs=0.05), (step, vc)
    x, y = _metres(p)
    for d in (x, y):
        assert abs(float(d.mean())) < 5.0 * float(d.std()) / math.sqrt(n)


# --- a spatially turning wind with a calm centre ---------------------------------------


VORTEX_OMEGA = 9e-4  # s-1: calm (speed <= sigma_u) within ~0.9 km of the centre
VORTEX_HALF_M = 5000.0  # particles start uniform over +-5 km
VORTEX_DISK_M = 2400.0  # the calm region plus ~0.8 km of diffusion from it
VORTEX_RING_M = (2400.0, 3200.0)  # outside it, 1.8 km from the box edge (diffusion ~0.8 km)


def _vortex_run(
    frame: str,
    seed: int,
    *,
    regime: tuple[float, float, float] = NEUTRAL,
    n: int = 156_250,
    steps: int = 60,
    dt: float = 60.0,
):
    """One run of the vortex case; returns (disk change, its sd, expected disk change,
    ring change, its sd). Changes are particle counts after - before; sd is the
    square root of the number of particles that crossed the region's edge."""

    blh = regime[0]
    half_deg, ng = 0.1, 21
    lon = torch.linspace(-half_deg, half_deg, ng)
    lat = torch.linspace(-half_deg, half_deg, ng)
    u = (-VORTEX_OMEGA * lat * M_LAT).view(ng, 1).expand(ng, ng)
    v = (VORTEX_OMEGA * lon * M_LON).view(1, ng).expand(ng, ng)
    met = _met(u, v, half_deg=half_deg, n=ng, regime=regime)

    gen = torch.Generator().manual_seed(seed)
    p = _particles(n, torch.rand(n, generator=gen) * blh)
    p[:, 0] = (torch.rand(n, generator=gen) * 2 - 1) * VORTEX_HALF_M / M_LON
    p[:, 1] = (torch.rand(n, generator=gen) * 2 - 1) * VORTEX_HALF_M / M_LAT
    r0 = torch.hypot(*_metres(p))
    scheme = HannaScheme(horizontal_frame=frame, static_substeps=False)
    p, _ = _run(scheme, met, p, steps=steps, dt=dt, seed=seed + 1)
    r1 = torch.hypot(*_metres(p))

    # The derived sink is 2π·½(Var_along − Var_cross)·c: the along-wind minus
    # cross-wind displacement variance after the same time, from rest, of particles
    # started the same way in the same column, measured in a uniform eastward wind
    # (a companion run; the model's own effective K_u − K_v, including how u′ and v′
    # follow the particles' vertical motion, which a column average of σ²T_L misses).
    expected = 0.0
    if frame == "wind":
        nc = 40_000
        gen = torch.Generator().manual_seed(seed + 2)
        pc = _particles(nc, torch.rand(nc, generator=gen) * blh)
        # A fresh scheme: its per-window field cache is keyed by the window's time.
        companion = HannaScheme(horizontal_frame=frame, static_substeps=False)
        pc, _ = _run(
            companion, _met(5.0, 0.0, regime=regime), pc, steps=steps, dt=dt, seed=seed + 3
        )
        xc, yc = _metres(pc)
        dvar = float(xc.var() - yc.var())
        expected = -2 * math.pi * 0.5 * dvar * n / (2 * VORTEX_HALF_M) ** 2

    def change(lo: float, hi: float) -> tuple[float, float]:
        before = (r0 >= lo) & (r0 < hi)
        after = (r1 >= lo) & (r1 < hi)
        return float(after.sum() - before.sum()), math.sqrt(int((before ^ after).sum()))

    d_disk, sd_disk = change(0.0, VORTEX_DISK_M)
    d_ring, sd_ring = change(*VORTEX_RING_M)
    return d_disk, sd_disk, expected, d_ring, sd_ring


def test_vortex_with_a_calm_centre_loses_only_the_derived_neglected_drift_term() -> None:
    """Horizontal homogeneity in a vortex (u = −Ωy, v = Ωx, Ω = 9e-4 s⁻¹, so calm,
    |U| ≤ σ_u, within ~0.9 km of the centre), uniform neutral turbulence (h 300 m,
    u* 0.4 m/s: σ_u/σ_v ≈ 1.5, T_L ≤ ~5 min), particles uniform in the column and
    over ±5 km, no advection, 1 h.

    GLIDE has no horizontal well-mixed drift (∇·K), so its horizontal transport is
    ∂c/∂t = ∂i∂j(K_ij c) in the diffusion limit. With K aligned to the wind,
    K = K_v I + (K_u − K_v) t tᵀ (t the unit tangent), ∂i∂j K_ij is zero everywhere
    outside the calm region and integrates to −2π(K_u − K_v) over it, whatever the
    calm treatment (decision 0032, derived). So: (i) a ring outside the calm region
    keeps its particles; (ii) the disk round the centre loses 2π·½ΔVar·c
    particles, ΔVar the along-wind minus cross-wind displacement variance over the
    same hour, measured in a uniform wind (``_vortex_run``). The neutral column keeps
    T_L ≪ 1 h, where the diffusion limit holds; in a stable column (T_Lu up to ~30
    min near h) the deficit was ~0.75 of this (decision 0032). A wrong sense of
    rotation (K_u radial) gives a gain of about the same size instead (planted:
    caught). The deficit does not depend on the calm treatment, so a fault there is
    the turning test's to catch. Tolerance: 4 standard deviations of each count
    change; calibration over seeds in the decision record."""

    d_disk, sd_disk, expected, d_ring, sd_ring = _vortex_run("wind", seed=5)
    assert expected < -4 * sd_disk  # the test resolves the effect it checks
    assert d_disk == pytest.approx(expected, abs=4 * sd_disk), (d_disk, expected, sd_disk)
    assert abs(d_ring) < 4 * sd_ring, (d_ring, sd_ring)
