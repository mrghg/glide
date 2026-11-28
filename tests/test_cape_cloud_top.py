"""CAPE is the buoyant energy of the diagnosed cloud, LCL to cloud top (decision 0031).

The external review's regression case ("CAPE above the diagnosed cloud top"):
two soundings share the surface parcel, the levels and 35 J/kg of CAPE below the
same cloud top, and both pass every other trigger check. Only the second has
another 30 J/kg of positive buoyancy in a layer above that top, separated from
the cloud by a stable layer. At the default 50 J/kg floor neither may fire.
Before decision 0031 both CAPE calculations summed positive buoyancy to the top
of the column, so the second sounding got 65 J/kg and fired.

The soundings are built backwards from a prescribed buoyancy profile: the parcel
is lifted with the module's own functions, and the environment's virtual
temperature is set to ``T_v,parcel / (1 + B)`` at each level, so the CAPE of
each layer is known in advance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from lpdm.convection import EmanuelReducedConvection
from lpdm.convection import emanuel as em
from lpdm.convection.emanuel import compute_lcl_lnb_cape, compute_lcl_lnb_cape_columns
from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

CAPE_BELOW_TOP = 35.0
CAPE_ABOVE_TOP = 30.0
B_STABLE = -0.01
B_ALOFT = -0.02
B_SUBCLOUD = -0.001  # weak CIN below the LCL: left out of CAPE, not subtracted

T_SFC_K = 300.0
Q_SFC = 0.016
P_HPA = np.arange(1000.0, 199.0, -25.0)  # 33 levels, surface first


def _parcel_tv(p_pa: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The surface parcel's virtual temperature at every level and the LCL index."""

    t = torch.full_like(p_pa, T_SFC_K)
    q = torch.full_like(p_pa, Q_SFC)
    return _parcel_tv_for(t, q, p_pa)


def _environment(
    tv_parcel: torch.Tensor, q_env: torch.Tensor, buoyancy: torch.Tensor
) -> torch.Tensor:
    """Environment temperature giving buoyancy ``B`` against the parcel."""

    tv_env = tv_parcel / (1.0 + buoyancy)
    t_env = tv_env / (1.0 + (1.0 / em.EPSILON_RD_RV - 1.0) * q_env)
    t_env[0] = T_SFC_K
    return t_env


def _layer_dz(t_env: torch.Tensor, p_pa: torch.Tensor) -> torch.Tensor:
    """Hypsometric thickness of the layer below each level (the module's dz)."""

    dz = torch.zeros_like(t_env)
    dz[1:] = (
        em.R_DRY_AIR_J_KG_K
        * 0.5
        * (t_env[:-1] + t_env[1:])
        * (torch.log(p_pa[:-1]) - torch.log(p_pa[1:]))
        / em.GRAVITY_M_S2
    )
    return dz


def two_soundings() -> dict:
    """The review's two soundings (float64, [Z], surface first).

    Levels from the LCL level ``i_lcl``: three buoyant cloud levels
    (35 J/kg in total), a stable layer of three levels starting at the cloud
    top ``i_lcl + 3``, then three levels that are buoyant (30 J/kg) in the
    second sounding only, then stable air to the top. The buoyancy of each
    buoyant layer is a constant scaled until its CAPE hits the target.
    """

    p_pa = torch.as_tensor(P_HPA * 100.0, dtype=torch.float64)
    n_lev = p_pa.numel()
    tv_parcel, i_lcl = _parcel_tv(p_pa)
    q_env = Q_SFC * torch.exp(-torch.arange(n_lev, dtype=torch.float64) / 8.0)
    cloud = slice(i_lcl, i_lcl + 3)
    stable = slice(i_lcl + 3, i_lcl + 6)
    upper = slice(i_lcl + 6, i_lcl + 9)

    def build(with_upper_layer: bool) -> tuple[torch.Tensor, torch.Tensor]:
        b = torch.full((n_lev,), B_ALOFT, dtype=torch.float64)
        b[0] = 0.0
        b[1:i_lcl] = B_SUBCLOUD
        b[stable] = B_STABLE
        b_cloud, b_upper = 0.005, 0.005
        for _ in range(50):  # the layer thickness depends weakly on B
            b[cloud] = b_cloud
            if with_upper_layer:
                b[upper] = b_upper
            t_env = _environment(tv_parcel, q_env, b)
            gdz = em.GRAVITY_M_S2 * _layer_dz(t_env, p_pa)
            b_cloud = CAPE_BELOW_TOP / float(gdz[cloud].sum())
            b_upper = CAPE_ABOVE_TOP / float(gdz[upper].sum())
        return t_env, b

    t1, b1 = build(with_upper_layer=False)
    t2, b2 = build(with_upper_layer=True)
    return {
        "p_pa": p_pa,
        "q": q_env,
        "t": (t1, t2),
        "b": (b1, b2),
        "i_lcl": i_lcl,
        "i_top": i_lcl + 3,
        "upper": upper,
    }


def _height(t_env: torch.Tensor, p_pa: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(_layer_dz(t_env, p_pa), dim=0)


def _window(s: dict) -> HourlyMetTensors:
    """A 1 x 2 met window: the first sounding west, the second east."""

    n_lev = s["p_pa"].numel()
    shape = (n_lev, 1, 2)
    names = ("u", "v", "w", "blh", "sp", "t", "q")
    t3 = torch.stack(s["t"], dim=1).view(shape)
    q3 = torch.stack((s["q"], s["q"]), dim=1).view(shape)
    height = torch.stack([_height(t, s["p_pa"]) for t in s["t"]], dim=1).view(shape)
    chans = {
        "u": torch.zeros(shape, dtype=torch.float64),
        "v": torch.zeros(shape, dtype=torch.float64),
        "w": torch.zeros(shape, dtype=torch.float64),
        "blh": torch.full(shape, 500.0, dtype=torch.float64),
        "sp": torch.full(shape, float(s["p_pa"][0]), dtype=torch.float64),
        "t": t3,
        "q": q3,
    }
    fields = torch.stack([chans[n] for n in names], dim=0)
    t0 = datetime(2024, 7, 1, 12, 0, 0, tzinfo=UTC)
    metadata = MetFieldMetadata(
        lon=np.array([0.0, 1.0]),
        lat=np.array([45.0]),
        level=height[:, 0, 0].numpy(),
        pressure_level_hpa=P_HPA.copy(),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units=dict.fromkeys(names, ""),
    )
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=height.contiguous(),
    )


@pytest.fixture(scope="module")
def soundings() -> dict:
    return two_soundings()


def test_the_soundings_are_what_the_review_describes(soundings) -> None:
    """Construction check, independent of how CAPE is bounded: the positive
    buoyant energy is 35 J/kg in the cloud and 30 J/kg above it (second
    sounding only), and the stable layer between them is negatively buoyant."""

    s = soundings
    p = s["p_pa"]
    for t, b in zip(s["t"], s["b"], strict=True):
        gdz = em.GRAVITY_M_S2 * _layer_dz(t, p)
        cloud = (b.clamp(min=0.0) * gdz)[s["i_lcl"] : s["i_top"] + 1].sum()
        assert float(cloud) == pytest.approx(CAPE_BELOW_TOP, abs=1e-6)
        assert bool((b[s["i_top"] : s["upper"].start] < 0.0).all())
    above = [
        float((b.clamp(min=0.0) * em.GRAVITY_M_S2 * _layer_dz(t, p))[s["upper"]].sum())
        for t, b in zip(s["t"], s["b"], strict=True)
    ]
    assert above[0] == 0.0
    assert above[1] == pytest.approx(CAPE_ABOVE_TOP, abs=1e-6)


def test_cape_stops_at_the_cloud_top_single_column(soundings) -> None:
    """Scalar path: same LCL and cloud top, 35 J/kg in both soundings."""

    s = soundings
    out = [compute_lcl_lnb_cape(t, s["q"], s["p_pa"]) for t in s["t"]]
    for i_lcl, i_lnb, cape, _ in out:
        assert int(i_lcl) == s["i_lcl"]
        assert int(i_lnb) == s["i_top"]
        assert float(cape) == pytest.approx(CAPE_BELOW_TOP, abs=1e-6)
    assert float(out[1][2]) == float(out[0][2])


def test_cape_stops_at_the_cloud_top_columns(soundings) -> None:
    """Batched path, the same two soundings as two columns."""

    s = soundings
    t = torch.stack(s["t"])
    q = torch.stack((s["q"], s["q"]))
    p = torch.stack((s["p_pa"], s["p_pa"]))
    i_lcl, i_lnb, cape, _ = compute_lcl_lnb_cape_columns(t, q, p)
    assert i_lcl.tolist() == [s["i_lcl"]] * 2
    assert i_lnb.tolist() == [s["i_top"]] * 2
    assert cape.tolist() == pytest.approx([CAPE_BELOW_TOP] * 2, abs=1e-6)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_buoyancy_above_the_cloud_top_does_not_trigger(soundings, dtype) -> None:
    """The review's regression test, through the scheme's own trigger at its
    defaults: every other check passes in both columns, and neither fires at
    the 50 J/kg floor. (Before decision 0031 the second column had 65 J/kg and
    fired.) With the floor lowered to 30 J/kg both fire, so CAPE alone decides."""

    met = _window(soundings)
    cpu = torch.device("cpu")
    scheme = EmanuelReducedConvection()
    cols = scheme.column_state(met, t_alpha=0.0, device=cpu, dtype=dtype)

    assert cols.i_lcl.tolist() == [soundings["i_lcl"]] * 2
    assert cols.i_lnb.tolist() == [soundings["i_top"]] * 2
    assert bool((cols.dtv_lcl_plus_1.double() >= scheme.trigger_dtv_k).all())
    assert bool((cols.cloud_depth_m.double() >= scheme.min_cloud_depth_m).all())
    assert cols.fires.tolist() == [False, False]
    assert cols.cape.tolist() == pytest.approx([CAPE_BELOW_TOP] * 2, abs=1e-3)
    assert bool((cols.m_b == 0.0).all())

    lower_floor = EmanuelReducedConvection(min_cape_j_kg=30.0)
    cols30 = lower_floor.column_state(met, t_alpha=0.0, device=cpu, dtype=dtype)
    assert cols30.fires.tolist() == [True, True]
    assert float(cols30.m_b[0]) == float(cols30.m_b[1])


def _varied_columns(dtype: torch.dtype, n: int = 64, seed: int = 31) -> tuple:
    """The two soundings plus ``n`` perturbed copies: surface warmed or cooled,
    moistened or dried, and noise on the temperature profile, so the LCL, the
    cloud top and buoyant layers above the top vary between columns."""

    s = two_soundings()
    g = torch.Generator().manual_seed(seed)
    base_t = torch.stack(s["t"]).repeat(n // 2 + 1, 1)[:n]
    n_lev = base_t.shape[1]
    t = base_t + 1.5 * torch.randn(n, n_lev, generator=g, dtype=torch.float64)
    t[:, 0] = base_t[:, 0] + 3.0 * (torch.rand(n, generator=g, dtype=torch.float64) - 0.5)
    q = s["q"].repeat(n, 1) * (
        1.0 + 0.3 * (torch.rand(n, 1, generator=g, dtype=torch.float64) - 0.5)
    )
    t = torch.cat((torch.stack(s["t"]), t))
    q = torch.cat((torch.stack((s["q"], s["q"])), q))
    p = s["p_pa"].repeat(t.shape[0], 1)
    return t.to(dtype), q.to(dtype), p.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_scalar_and_batched_cape_agree(dtype) -> None:
    """The single-column reference and the batched lift give the same LCL,
    cloud top and CAPE in every column, including columns with buoyant layers
    above the cloud top; and that CAPE is the positive buoyancy of the levels
    [LCL, LNB] and no other (independent sum in float64)."""

    t, q, p = _varied_columns(dtype)
    i_lcl, i_lnb, cape, dtv = compute_lcl_lnb_cape_columns(t, q, p)
    n_above = 0
    for c in range(t.shape[0]):
        r_lcl, r_lnb, r_cape, r_dtv = compute_lcl_lnb_cape(t[c], q[c], p[c])
        assert int(r_lcl) == int(i_lcl[c])
        assert int(r_lnb) == int(i_lnb[c])
        assert float(r_cape) == pytest.approx(
            float(cape[c]), rel=1e-6 if dtype == torch.float32 else 1e-12, abs=1e-9
        )
        assert float(r_dtv) == pytest.approx(float(dtv[c]), rel=1e-12)
    # Many columns must have positive buoyancy above their cloud top, or the
    # comparison would not exercise the bound.
    t64, q64, p64 = (x.double() for x in (t, q, p))
    _, _, cape_top, _ = compute_lcl_lnb_cape_columns(t64, q64, p64)
    tv_env = em.virtual_temperature(t64, q64)
    for c in range(t.shape[0]):
        if int(i_lcl[c]) < 0:
            continue
        lo, hi = int(i_lcl[c]), int(i_lnb[c])
        # Buoyancy is not returned; recover it from the reference CAPE pieces.
        dz = _layer_dz(t64[c], p64[c])
        tv_parcel, _ = _parcel_tv_for(t64[c], q64[c], p64[c])
        b = (tv_parcel - tv_env[c]) / tv_env[c]
        pos = em.GRAVITY_M_S2 * b.clamp(min=0.0) * dz
        assert float(cape_top[c]) == pytest.approx(
            float(pos[lo : hi + 1].sum()), rel=1e-12, abs=1e-9
        )
        n_above += int(float(pos[hi + 1 :].sum()) > 1.0)
    assert n_above >= 10, f"only {n_above} columns have buoyancy above the top"


def _parcel_tv_for(t: torch.Tensor, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, int]:
    """`_parcel_tv` for an arbitrary surface state (float64 column)."""

    t_s, q_s, p_s = t[:1], q[:1], p[:1]
    t_lcl = em.lcl_temperature_bolton(t_s, q_s, p_s)
    p_lcl = em.lcl_pressure_poisson(t_s, t_lcl, p_s).squeeze(0)
    theta_e = em.equivalent_potential_temperature(t_s, q_s, p_s).squeeze(0)
    theta_s = em.potential_temperature(t_s, p_s).squeeze(0)
    dry_t = theta_s * (p / em.P0_PA).pow(em.KAPPA_POISSON)
    moist_t = em.lift_parcel_moist_pseudo_adiabatic(theta_e, p)
    below = p >= p_lcl
    t_parcel = torch.where(below, dry_t, moist_t)
    q_parcel = torch.where(
        below, q_s.expand_as(t_parcel), em.saturation_specific_humidity(t_parcel, p)
    )
    has = bool((~below).any())
    return em.virtual_temperature(t_parcel, q_parcel), int((~below).nonzero()[0]) if has else -1
