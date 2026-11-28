"""Decision 0034: the `trigger` option of the reduced Emanuel scheme.

`flexpart_cloud_base` (the default) applies FLEXPART 11's tests without memory
(`flexpart_cloud_base_trigger_columns`): the parcel not DTMAX = 0.9 K colder than
the environment at the first level above the LCL, DTMIN > 0, and FLEXPART's
other skip conditions; no CAPE floor and no depth check. `glide` (the default
before 0034) must reproduce the trigger before the option existed bit for bit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from lpdm.convection import emanuel as em
from lpdm.convection.emanuel import EmanuelReducedConvection
from tests.test_convection_closure import _convective_column, _window

CPU = torch.device("cpu")


def _state(scheme: EmanuelReducedConvection, window, dtype=torch.float32):
    return scheme.column_state(window, t_alpha=0.0, device=CPU, dtype=dtype)


def _pre_option_fires(scheme: EmanuelReducedConvection, cols) -> torch.Tensor:
    """The trigger expression before the option existed (branch agent/CONV-closure, 0ee9296)."""

    return (
        (cols.i_lcl >= 0)
        & (cols.i_lnb > cols.i_lcl)
        & (cols.cape.double() >= scheme.min_cape_j_kg)
        & (cols.dtv_lcl_plus_1.double() >= scheme.trigger_dtv_k)
        & (cols.cloud_depth_m.double() >= scheme.min_cloud_depth_m)
    )


@pytest.mark.parametrize("n_lev", [12, 23, 40, 80])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_glide_trigger_is_bit_identical_to_before_0034(n_lev: int, dtype: torch.dtype) -> None:
    """`trigger="glide"` gives the pre-0034 trigger expression and M_b bit for bit;
    the default is now `flexpart_cloud_base`, with the same lift and closure."""

    window = _window(n_lev)
    glide = EmanuelReducedConvection(trigger="glide")
    default = EmanuelReducedConvection()
    assert default.trigger == "flexpart_cloud_base"
    a = _state(glide, window, dtype)
    assert torch.equal(a.fires, _pre_option_fires(glide, a))
    m_b = glide.buoyancy_mass_flux_per_k * a.dtmin_k.double().clamp(min=0.0)
    assert torch.equal(a.m_b, torch.where(a.fires, m_b, torch.zeros_like(m_b)))
    b = _state(default, window, dtype)
    for name in ("i_lcl", "i_lnb", "cape", "dtmin_k", "cloud_depth_m"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_config_default_and_validation() -> None:
    from lpdm.config import EmanuelConvectionConfig  # noqa: PLC0415

    assert EmanuelConvectionConfig().trigger == "flexpart_cloud_base"
    assert EmanuelConvectionConfig(trigger="glide").trigger == "glide"
    for bad in (
        {"trigger": "x"},
        {"closure": "cape_velocity"},  # the default trigger needs closure buoyancy
        {"trigger": "flexpart_cloud_base", "closure": "cape_velocity"},
    ):
        with pytest.raises(ValueError):
            EmanuelConvectionConfig(**bad)
        with pytest.raises(ValueError):
            EmanuelReducedConvection(**bad)
    # The pre-0033 scheme is still reachable with the pre-0034 trigger.
    legacy = {"trigger": "glide", "closure": "cape_velocity", "application": "cfl_scaled"}
    EmanuelConvectionConfig(**legacy)
    EmanuelReducedConvection(**legacy)
    # Glide-only options are accepted (and warned by main) under the default trigger.
    EmanuelConvectionConfig(trigger_height_above_lcl_m=200.0, min_cape_j_kg=10.0)
    EmanuelReducedConvection(trigger_height_above_lcl_m=200.0, min_cape_j_kg=10.0)


def _profiles(n_col: int, n_lev: int = 12):
    """Simple columns (surface first): T, q, heights, pressures."""

    z = torch.linspace(0.0, 11000.0, n_lev, dtype=torch.float64)
    t = (300.0 - 6.5e-3 * z).expand(n_col, n_lev).clone()
    q = (0.012 * torch.exp(-z / 2500.0)).expand(n_col, n_lev).clone()
    h = z.expand(n_col, n_lev).clone()
    return t, q, h


def test_flexpart_trigger_tests_one_by_one() -> None:
    """Each FLEXPART skip condition switches off exactly its column."""

    n_col, n_lev = 7, 12
    t, q, h = _profiles(n_col, n_lev)
    p_lcl = torch.full((n_col,), 90000.0, dtype=torch.float64)
    i_lcl = torch.full((n_col,), 2, dtype=torch.long)
    dtv = torch.full((n_col, n_lev), 0.2, dtype=torch.float64)
    dtmin = torch.full((n_col,), 1.5, dtype=torch.float64)
    # 0: passes. 1: parcel 0.9 K colder at ICB. 2: just warmer than -0.9 K, passes.
    dtv[1, 2] = -0.9
    dtv[2, 2] = -0.89
    # 3: DTMIN <= 0. 4: parcel level colder than 250 K. 5: PLCL < 200 hPa.
    dtmin[3] = 0.0
    t[4, 0] = 249.0
    p_lcl[5] = 19000.0
    # 6: moist static energy increasing monotonically (no minimum below the top).
    q[6] = 0.0
    t[6] = 300.0
    ok, dtv_icb = em.flexpart_cloud_base_trigger_columns(t, q, h, p_lcl, i_lcl, dtv, dtmin)
    assert ok.tolist() == [True, False, True, False, False, False, False]
    assert torch.equal(dtv_icb, dtv[:, 2])
    # ICB >= NL - 1 (GLIDE's i_lcl >= Z - 3) and no LCL.
    i_lcl[0], i_lcl[2] = n_lev - 3, -1
    ok, _ = em.flexpart_cloud_base_trigger_columns(t, q, h, p_lcl, i_lcl, dtv, dtmin)
    assert not bool(ok[0]) and not bool(ok[2])


def test_flexpart_trigger_on_synthetic_soundings() -> None:
    """On the convective sounding both triggers fire on 23 levels with the same
    M_b; on 40 levels ΔT_v at LCL+1 is 0.887 K and only FLEXPART's test fires
    (the grid dependence of GLIDE's). A parcel 0.3 K warmer than the environment
    at ICB in a stable layer fires FLEXPART's test, with no CAPE floor and no
    depth check (its first-LNB cloud is one level deep); GLIDE's does not."""

    glide = EmanuelReducedConvection(trigger="glide")
    fp = EmanuelReducedConvection()
    a, b = _state(glide, _window(23)), _state(fp, _window(23))
    assert bool(a.fires.all()) and bool(b.fires.all())
    assert torch.equal(a.m_b, b.m_b)
    a, b = _state(glide, _window(40)), _state(fp, _window(40))
    assert not bool(a.fires.any()) and bool(b.fires.all())
    assert float(a.dtv_lcl_plus_1[0]) == pytest.approx(0.887, abs=0.01)

    t, q_, p = _convective_column(40)
    i_lcl = int(b.i_lcl[0])
    dtv = em.compute_lcl_lnb_cape_columns(
        t.unsqueeze(0), q_.unsqueeze(0), p.unsqueeze(0), return_extra=True
    )[4]["dtv"][0]
    t2 = t.clone()
    t2[i_lcl + 1 :] = t[i_lcl + 1 :] + dtv[i_lcl + 1 :].clamp(min=0.0) + 0.5
    t2[i_lcl] = t[i_lcl] + float(dtv[i_lcl]) - 0.3  # parcel 0.3 K warmer at ICB
    from tests.test_convection import _build_convective_met_window  # noqa: PLC0415

    w2 = _build_convective_met_window(column=(t2, q_, p))
    a2, b2 = _state(glide, w2, torch.float64), _state(fp, w2, torch.float64)
    # min_cloud_depth_m applies to the glide trigger only.
    fp9 = EmanuelReducedConvection(trigger="flexpart_cloud_base", min_cloud_depth_m=9000.0)
    c2 = _state(fp9, w2, torch.float64)
    assert torch.equal(c2.fires, b2.fires)
    assert not bool(a2.fires.any())
    assert bool((c2.dtmin_k > 0).all()) and bool(c2.fires.all())
    assert bool((c2.i_lnb == c2.i_lcl + 1).all()) and bool((c2.cloud_depth_m < 500.0).all())
    assert float(c2.dtv_lcl_plus_1[0]) == pytest.approx(0.3, abs=0.05)
    assert bool((c2.cape < 50.0).all())


# ---------------------------------------------------------------------------
# Decision 0034: the default trigger
# ---------------------------------------------------------------------------

FIXTURE = Path(__file__).parent / "data" / "trigger_columns_l40.npz"


def _fixture_window():
    """The fixture's real columns as a met window of one row: [Z, 1, C]."""

    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata  # noqa: PLC0415

    d = np.load(FIXTURE)
    t, q = torch.as_tensor(d["t"]), torch.as_tensor(d["q"])
    p, h = torch.as_tensor(d["pressure_hpa"]), torch.as_tensor(d["height_agl_m"])
    n_lev, n_col = t.shape
    names = ("t", "q")
    fields = torch.stack([t, q], dim=0).unsqueeze(2)  # [ch, Z, 1, C]
    t0 = datetime(2019, 7, 8, 12, tzinfo=UTC)
    metadata = MetFieldMetadata(
        lon=np.arange(n_col, dtype=np.float64),
        lat=np.zeros(1),
        level=h.double().mean(dim=1).numpy(),
        pressure_level_hpa=p.double().mean(dim=1).numpy(),
        time_start=t0,
        time_end=t0 + timedelta(seconds=float(d["interval_s"])),
        variable_units=dict.fromkeys(names, ""),
    )
    p3 = p.unsqueeze(1)
    window = HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=h.unsqueeze(1),
        pressure_hpa=(p3, p3),
    )
    return window, d


def test_default_trigger_reproduces_the_study_mask_on_real_columns() -> None:
    """On real ERA5 columns (8 Jul 2019 12 UTC, EUR, 40 levels; stratified to
    include columns within 0.3 K of the ΔT_v(ICB) = −0.9 K and DTMIN = 0
    thresholds), the default trigger fires exactly where the Decision 6 study's
    `flexpart_cloud_base` did (`trigger_columns.py` at d9f09f0), and `glide`
    exactly where the pre-0034 trigger did. Fixture:
    `scripts/validation/conv/trigger/make_trigger_fixture.py`."""

    window, d = _fixture_window()
    study_fp = torch.as_tensor(d["fires_flexpart_cloud_base"])
    study_gl = torch.as_tensor(d["fires_glide"])
    # The fixture exercises both outcomes of both triggers.
    assert 0 < int(study_fp.sum()) < study_fp.numel()
    assert 0 < int(study_gl.sum()) < int(study_fp.sum())
    default = _state(EmanuelReducedConvection(), window)
    glide = _state(EmanuelReducedConvection(trigger="glide"), window)
    assert torch.equal(default.fires, study_fp)
    assert torch.equal(glide.fires, study_gl)
    # Columns where the parcel is colder than the environment at ICB, but by
    # less than 0.9 K, fire under the default and never under glide.
    dtv_icb = default.dtv_lcl_plus_1
    colder = (dtv_icb < 0.0) & (dtv_icb > -0.9) & default.fires
    assert bool(colder.any()) and not bool((colder & glide.fires).any())


@pytest.mark.parametrize(("colder_k", "fires"), [(0.5, True), (1.0, False)])
def test_flexpart_trigger_fires_with_the_parcel_slightly_colder_at_icb(
    colder_k: float, fires: bool
) -> None:
    """FLEXPART's sign: the column convects unless the parcel at ICB (the first
    level above the LCL) is 0.9 K or more colder than the environment. With the
    environment warmed at ICB so that the parcel is 0.5 K colder there, the
    convective sounding fires; 1.0 K colder, it does not. The LCL, the cloud and
    DTMIN > 0 are unchanged. Glide's trigger (≥ 0.9 K warmer at LCL+1; 0.887 K
    on this 40-level grid) fires in neither."""

    from tests.test_convection import _build_convective_met_window  # noqa: PLC0415

    t, q, p = _convective_column(40)
    dtv = em.compute_lcl_lnb_cape_columns(
        t.unsqueeze(0), q.unsqueeze(0), p.unsqueeze(0), return_extra=True
    )[4]["dtv"][0]
    icb = int(_state(EmanuelReducedConvection(), _window(40), torch.float64).i_lcl[0])
    t2 = t.clone()
    # Warm the environment at ICB so T_v(parcel) − T_v(env) = −colder_k there
    # (T_v ∝ T at fixed q; the parcel, lifted from the surface, is unchanged).
    t2[icb] = t[icb] + (float(dtv[icb]) + colder_k) / (1.0 + 0.608 * float(q[icb]))
    w = _build_convective_met_window(column=(t2, q, p))
    fp = _state(EmanuelReducedConvection(), w, torch.float64)
    gl = _state(EmanuelReducedConvection(trigger="glide"), w, torch.float64)
    assert bool((fp.i_lcl == icb).all())
    assert float(fp.dtv_lcl_plus_1[0]) == pytest.approx(-colder_k, abs=0.02)
    assert bool((fp.dtmin_k > 0.0).all()) and bool((fp.i_lnb > fp.i_lcl).all())
    assert bool(fp.fires.all()) is fires and bool(fp.fires.any()) is fires
    assert bool((fp.m_b > 0.0).all()) is fires
    assert not bool(gl.fires.any())


def test_options_the_selected_trigger_ignores_are_warned() -> None:
    """trigger_dtv_k, min_cape_j_kg, min_cloud_depth_m and
    trigger_height_above_lcl_m under the FLEXPART trigger are warned (not refused)
    for runtime.physics_warnings; under trigger 'glide' they are not."""

    from lpdm.main import _convection_option_warnings  # noqa: PLC0415
    from tests.test_convection_closure import _run_config  # noqa: PLC0415

    glide_only = {
        "trigger_dtv_k": 0.9,
        "min_cape_j_kg": 50.0,
        "min_cloud_depth_m": 500.0,
        "trigger_height_above_lcl_m": 200.0,
    }
    for name, value in glide_only.items():
        msgs = _convection_option_warnings(_run_config({name: value}))
        assert len(msgs) == 1 and f"convection.emanuel.{name}" in msgs[0]
        assert "trigger 'glide'" in msgs[0] and "decision 0034" in msgs[0]
    assert len(_convection_option_warnings(_run_config(glide_only))) == 4
    assert _convection_option_warnings(_run_config({**glide_only, "trigger": "glide"})) == []
    assert _convection_option_warnings(_run_config({"trigger": "flexpart_cloud_base"})) == []


def test_ignored_trigger_option_is_recorded_in_run_metadata(tmp_path: Path, caplog) -> None:
    """A run whose config sets min_cloud_depth_m under the default trigger logs
    the warning once and records it in run_metadata.json (runtime.physics_warnings)."""

    import json  # noqa: PLC0415
    import logging  # noqa: PLC0415

    from lpdm.config import ConvectionConfig, EmanuelConvectionConfig  # noqa: PLC0415
    from lpdm.convection.no_convection import NoConvection  # noqa: PLC0415
    from lpdm.main import _run  # noqa: PLC0415
    from tests.test_main_runtime import _make_reader, _make_run_config  # noqa: PLC0415

    cfg = _make_run_config(output_uri=str(tmp_path / "out"))
    conv = ConvectionConfig(
        scheme="emanuel_reduced", emanuel=EmanuelConvectionConfig(min_cloud_depth_m=500.0)
    )
    cfg = cfg.model_copy(update={"convection": conv})
    # The warning comes from the config; the scheme passed in is a pass-through.
    with caplog.at_level(logging.WARNING):
        metadata = _run(
            cfg,
            reader=_make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0)),
            convection_scheme=NoConvection(),
        )
    warned = [r.getMessage() for r in caplog.records if "min_cloud_depth_m" in r.getMessage()]
    assert len(warned) == 1
    assert metadata["runtime"]["physics_warnings"] == warned
    on_disk = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert on_disk["runtime"]["physics_warnings"] == warned
