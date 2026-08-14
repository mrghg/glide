# Validation

What GLIDE has been checked against, what it has **not**, and how to run the
comparison against another model yourself.

> **The headline: the transport physics has not been validated against external
> references.** The suite below verifies GLIDE against closed-form solutions and
> against its own invariants. That is a strong internal check — it catches
> well-mixed violations, discretisation bias, sign errors and unit errors — but
> it does not establish that the parameterisations are right for the real
> atmosphere. A systematic comparison against NAME and FLEXPART is in progress
> and not signed off. Treat current results as indicative.

**Contents**

1. [Running the suite](#1-running-the-suite)
2. [What the tests cover](#2-what-the-tests-cover)
3. [The analytic verification tests](#3-the-analytic-verification-tests)
4. [Well-mixed and conservation gates](#4-well-mixed-and-conservation-gates)
5. [GPU-capture guards that run on CPU](#5-gpu-capture-guards-that-run-on-cpu)
6. [What is not validated](#6-what-is-not-validated)
7. [Comparing against FLEXPART, NAME and STILT](#7-comparing-against-flexpart-name-and-stilt)
8. [Adding a physics test](#8-adding-a-physics-test)

---

## 1. Running the suite

```bash
.venv/bin/python -m pytest -q                      # everything
.venv/bin/python -m pytest -q tests/test_physics.py # engine primitives only
```

**298 tests across 16 files, ~135 s, 90% statement coverage, no network access.**
End-to-end tests use synthetic meteorology through `AnalyticMetReader`, so
nothing depends on remote ERA5 or on the restricted validation datasets. Tests
share no state and run order is irrelevant. Stochastic tests are seeded.

---

## 2. What the tests cover

| File | Scope | Tests |
| --- | --- | --- |
| `test_met_reader.py` | Reader mechanics: units, ω→w, heat-flux sign, longitude conventions, multi-store stitching, model-level detection, terrain-following wiring | 40 |
| `test_main_config.py` | Config schema, release expansion, batching | 39 |
| `test_hanna.py` | Regime formulae, stability classification, meander, sub-step machinery, per-window caches, compile gating, static/dynamic equivalence | 35 |
| `test_main_runtime.py` | End-to-end on synthetic met: trajectories, well-mixed through the production scheme, static/dynamic parity, graph guards, outputs, memory-guard aborts | 34 |
| `test_vertical_grid.py` | Terrain-following resample kernels: AGL regrid, sub-surface exclusion, slope correction, hydrostatic model-level pressure | 24 |
| `test_download_sample_cube.py` | Meteorology download helper | 22 |
| `test_convection.py` | Emanuel: thermodynamics, LCL/LNB/CAPE, mass-flux matrix non-divergence and well-mixedness | 19 |
| `test_footprint.py` | Gridder: binning, conservation, drop contracts, per-release scatter | 18 |
| `test_comparison.py` | STILT conversion + conservative regridding | 18 |
| `test_release_generator.py` | Point / column / batch particle generation, seeding | 15 |
| `test_physics.py` | Engine primitives: RK2 (incl. convergence order), OU/Langevin, reflection, well-mixed drift, coordinate normalisation | 13 |
| `test_output_writer.py` | Zarr / Parquet output contracts | 9 |
| `test_plume_footprint.py` | Analytic Gaussian-plume footprint (flagship) | 4 |
| `test_dispersion_analytic.py` | OU autocorrelation, Taylor dispersion, solid-body rotation | 4 |
| `test_terrain_transport.py` | Terrain-following transport end-to-end | 2 |
| `test_diffusion_pde.py` | Langevin diffusion limit vs a PDE reference | 2 |

---

## 3. The analytic verification tests

These are the ones that matter for physics confidence: each compares the model
against a closed-form or independently-computed reference, and each has "teeth" —
a companion test showing the check *fails* when the physics is wrong.

### OU statistics and Taylor dispersion — `test_dispersion_analytic.py`

| Test | Reference | Tolerance (observed) | Seed |
| --- | --- | --- | --- |
| `test_ou_autocorrelation_and_stationarity` | $R(\tau) = e^{-\tau/T_L}$ at $\tau/T_L \in \{0.5, 1, 2\}$; stationary $\mathrm{Var} = \sigma_w^2$ | $\lvert R - e^{-\tau/T_L}\rvert < 0.02$ (obs ~0.003); variance ±3% | 4111 |
| `test_solid_body_rotation_advection_returns_to_start` | circular trajectory closes after one period; RK2 second order | error ratio > 3.5× per $\Delta t$ halving (obs 4.00); finest return < $10^{-3} r$ | deterministic |
| `test_taylor_dispersion_curve_ballistic_to_diffusive` | $\sigma_z^2(t) = 2\sigma_w^2 T_L[t - T_L(1-e^{-t/T_L})]$ at 6 checkpoints, plus both asymptotes | curve < 5% (obs ~0.1%); ballistic < 5%; diffusive < 8% | 2201 |
| `test_taylor_dispersion_position_integration_bias_with_dt` | forward-Euler position bias | tight at $\Delta t/T_L = 0.01$ (< 2%), bounded at 0.2 (< 15%) | 71 |

### The flagship: an analytic plume footprint — `test_plume_footprint.py`

One backward-plume simulation ($z_r = 50$ m, $U = 5\ \mathrm{m\,s^{-1}}$,
$\sigma_v = \sigma_w = 0.5\ \mathrm{m\,s^{-1}}$, $T_L = 100$ s, 200k particles,
$\Delta t = 5$ s, ground reflection) compared against the **exact cell-integrated
reflected-Gaussian surface residence**, with $\sigma(t)$ from Taylor. Raw
residence time is asserted first — no unit ambiguity — and the STILT conversion is
then checked as an exact scale factor.

This exercises the whole chain at once: advection, OU turbulence, ground
reflection, the gridder, and the unit conversion.

| Test | Asserts | Tolerance (observed) |
| --- | --- | --- |
| `test_footprint_matches_analytic_gaussian_plume` | crosswind-integrated columns per travel-time band; 2-D correlation | 2–5 $T_L$: max < 5% (obs 0.4%); ≥ 5 $T_L$: max < 15% (obs ≤ 10%, statistical); corr > 0.995 (obs 0.9985) |
| `test_footprint_absolute_magnitude_matches_plume` | total surface residence vs analytic | ratio within ±3% (obs 0.992) |
| `test_footprint_crosswind_width_matches_taylor` | moment-based $\sigma_y$ (Sheppard-corrected) at ~6 distances | < 5% (obs ≤ 1.1%) |
| `test_stilt_conversion_scales_raw_footprint_exactly` | STILT field = raw × $m_{\mathrm{air}}/(h\rho)$ | exact (rtol $10^{-12}$) |

Seed 9021 throughout.

### The diffusion limit vs a PDE — `test_diffusion_pde.py`

The production OU + Thomson drift + reflection, run with an inhomogeneous
$K(z) = 0.02 + 0.12\min(z, 200\ \mathrm{m})$, compared against a conservative
flux-form Crank–Nicolson solution of $\partial c/\partial t = \partial_z(K
\partial_z c)$. The release slab (5–15 m) sits deliberately *inside* the low-$K$
layer — a mid-column release was measured to be insensitive to near-surface $K$
and would not have discriminated.

| Test | Asserts | Tolerance (observed) |
| --- | --- | --- |
| `test_langevin_diffusion_limit_matches_pde` | binned density at 300/900/1800 s; near-surface (0–60 m) occupancy | $L_1 < 0.08$ (obs ≤ 0.044); near-surface rel. < 12% (obs ≤ 7.9%) |
| `test_diffusion_pde_discriminates_near_surface_k_errors` | **teeth**: distance to PDEs with near-surface $K$ halved and quartered | half $L_1 > 0.12$–0.25 (obs 0.18–0.33); quarter > 0.30–0.45 (obs 0.46–0.66) |

The teeth test targets exactly the bias class that the $T_L$ floors fixed (see
[turbulence.md §5](turbulence.md#5-the-lagrangian-timescale-floors)) — a
near-surface $K$ that is too small.

### Terrain-following transport — `test_terrain_transport.py`

End-to-end through the **real** `ArcoEra5ZarrReader` resample, on a synthetic
pressure-level store containing a Gaussian hill and the terrain-following
vertical velocity.

| Test | Asserts | Observed |
| --- | --- | --- |
| `test_terrain_following_preserves_agl_crossing_hill` | a near-surface particle holds its AGL crossing an 800 m hill (slope correction cancels the imposed $w$) | max excursion ~2.7 m (limit 25 m) |
| `test_no_slope_correction_lets_particle_ride_the_terrain` | **teeth**: without the resample the particle rides the terrain up | excursion ~784 m ≈ the hill height (limit > 200 m) |

### A note on scope

The OU and Taylor tests are verified at the **engine** level with prescribed
constant $(\sigma_w, T_L)$, not driven through `HannaScheme.step`. That is
deliberate: Hanna has no homogeneous regime — $T_L$ is intrinsically
height-dependent, which is the very inhomogeneity the Thomson drift corrects — so
the OU/Taylor statistics have no closed form through the assembled scheme. The
scheme's own integration is covered instead by the well-mixed tests (which *are*
inhomogeneous and use the full step) and by the static/dynamic equivalence tests.

---

## 4. Well-mixed and conservation gates

The well-mixed condition is the unifying correctness criterion for this class of
model: almost every bug worth finding — a missing drift term, a bad reflection, a
sign error, too large a timestep — shows up as a violation of it. These run in
CI:

| Test | Asserts |
| --- | --- |
| `test_well_mixed_condition_drift_keeps_uniform_distribution` | the drift term alone preserves a uniform distribution |
| `test_well_mixed_uniformity_in_periodic_turbulence` | uniformity in a periodic domain (rel-RMS < 0.12) |
| `test_v1_well_mixed_hanna_backward_path` | a flat distribution stays flat through the **production backward scheme** at constant $\rho$ — parametrised over both the static and dynamic paths |
| `test_v1_density_weighted_well_mixed_with_F2` | a $\rho$-weighted distribution stays $\rho$-weighted under varying $\rho$ |
| `test_hanna_well_mixed_no_runaway_lofting` | no systematic upward drift (this caught the frozen $(1+w'^2/\sigma_w^2)$ factor) |
| `test_reflect_surface_w_flip_resolves_one_way_downward_drift` | the joint $(z, w')$ reflection removes the post-reflection downward bias |
| `test_convection_transition_preserves_mass_distribution_both_directions` | $\mathbf{m}^{\mathsf{T}}P = \mathbf{m}^{\mathsf{T}}$ for the convection matrix, forward **and** backward — proved deterministically, not sampled |
| `test_total_mass_conservation_in_bounds`, `test_static_path_footprint_conservation` | total particle weight and footprint mass conserved |

---

## 5. GPU-capture guards that run on CPU

Two classes of regression silently destroy the CUDA-graph capture and cost a 4–6×
slowdown with no error message. Both are caught without a GPU (see
[architecture.md §5](architecture.md#5-the-per-step-path-and-cuda-graphs)):

- `test_step_core_traces_as_one_graph_no_breaks` — compiles the per-step core with
  `fullgraph=True, backend="eager"` and raises on any **graph break**.
- `test_step_core_does_not_recompile_per_step` — sets
  `torch._dynamo.config.error_on_recompile` and steps through changing $\alpha$,
  meteorology values and level arrays, raising on any **recompile**.

Plus `test_compiled_hot_paths_match_eager_and_never_hard_fail`, which enforces
that the compiled path agrees with the eager reference and that a compile failure
degrades to eager rather than killing the run.

---

## 6. What is not validated

**Externally, against other models or observations — everything.** Specifically:

- Quantitative endpoint spread, time–height structure, and column-integrated
  footprint magnitude under the Hanna scheme. The unit tests pin local
  $\sigma$/$T_L$ values against literature forms and the analytic tests pin
  dispersion against closed-form solutions, but a systematic comparison against
  NAME/FLEXPART on identical release setups has not been completed.
- Free-troposphere transport accuracy. The engine-level OU dispersion that the
  Richardson closure feeds is verified; the closure's own $\sigma$/$T_L$
  magnitudes are not.
- Convective transport magnitude. The matrix is proven mass-conserving and
  well-mixed-preserving; whether it moves the *right amount* of mass is untested
  against a reference.

**One deferred test.** A forward/backward reciprocity check (run a
source–receptor pair both ways and require agreement) would test the backward
formulation itself, which nothing else does directly. It is specified but not
implemented.

**Sequencing caveat.** Both the terrain-following coordinate and the move to
native model levels change footprint magnitudes *everywhere*, not only over
mountains — so any comparison run before those landed needs re-running before its
magnitudes are trusted. See [../STATUS.md](../STATUS.md).

---

## 7. Comparing against FLEXPART, NAME and STILT

GLIDE accumulates the footprint **directly onto a configurable target grid**
(`output_grid`), so the usual path is to set the output grid equal to the
reference's grid and skip regridding altogether.

### Step 1 — author a config matching the reference

Align these fields:

| Field | Match to |
| --- | --- |
| `output_grid.lon_bounds` / `lat_bounds` | the reference grid's **outer cell edges**. References usually label cell *centres*, so add half a cell at each edge. |
| `output_grid.n_x` / `n_y` | equal cells filling that interval |
| `output_grid.z_edges_m` | make the bottom pair match the reference's surface layer (0–40 m for FLEXPART / NAME). Direct accumulation into that bin makes the unit conversion exact rather than depth-weighted. |
| `release.point.*`, `release.duration_seconds` | the reference release |
| `simulation.length_seconds` | the reference backward window |

`configs/example_mhd_january.yaml` (single release) and
`configs/example_mhd_january_periodic.yaml` (hourly releases) are both already
aligned with the bundled FLEXPART fixture.

### Step 2 — convert to STILT units

```python
import xarray as xr
from lpdm.comparison import to_stilt_surface_footprint

fp = xr.open_zarr("outputs/mhd-202401-hourly/footprints.zarr")["footprint"]
one = fp.isel(release=0)        # or .sel() on the release coords

stilt = to_stilt_surface_footprint(
    one,
    surface_layer_depth_m=40.0,   # must equal the bottom z-bin you ran with
    air_density_kg_m3=1.225,      # or a 2-D field from surface_air_density_from_met
    integrate_time=True,
)
```

Raw footprints are in seconds per cell; this applies Lin et al. (2003) Eq. 5 to
get $\mathrm{m^2\,s\,mol^{-1}}$.

### Step 3 — compare cell for cell

```python
import numpy as np

ref = xr.open_dataset("data/FLEXPART/FLEXPART_MHD_test_202401.nc", engine="h5netcdf")
ref_field = ref["srr"].sel(time="2024-01-01T00:00:00").sum("time")

diff = stilt - ref_field
print(f"correlation: {float(xr.corr(stilt, ref_field)):.3f}")
print(f"RMSE:        {float(np.sqrt((diff**2).mean())):.3e}")
print(f"total ratio: {float(stilt.sum() / ref_field.sum()):.3f}")
```

If the grids could not be aligned, `lpdm.comparison.regrid_conservative` does
mass-conservative area-weighted regridding for rectangular lat/lon grids.

### Caveats worth stating in any reported tolerance

- **Different meteorology.** GLIDE streams ERA5; FLEXPART runs typically use
  ECMWF operational analyses on native model levels; NAME uses UM analyses.
  Inter-model meteorology differences contribute footprint differences that have
  nothing to do with the turbulence scheme.
- **Mismatched surface-layer depth.** If you cannot make the bins match exactly,
  the converter depth-weights overlapping bins — approximate, and it assumes
  uniform residence-time density within a bin.
- **Spatially varying density.** For runs spanning large latitude or elevation
  ranges, replace the scalar `air_density_kg_m3` with a 2-D field.
- **Release setup.** Keep the release inside the surface layer
  (`alt_agl_m < surface_layer_depth_m`) so "particle not yet mixed" startup
  transients do not dominate.
- **Time resolution.** `integrate_time=False` keeps the `time_ago` axis, which is
  useful for diagnosing *when* the GLIDE plume diverges from the reference.

The validation datasets themselves (NAME, FLEXPART, EDGAR) are not redistributed
with this repository — see [../data/README.md](../data/README.md).

---

## 8. Adding a physics test

When changing `gpu_engine.py`, the runtime loop, a scheme, or the footprint
accumulator:

1. Add or update the engine-level test in `tests/test_physics.py` (or
   `test_footprint.py`, `test_hanna.py`, `test_convection.py` as appropriate).
2. If the change affects end-to-end behaviour, add a test in
   `tests/test_main_runtime.py` using `AnalyticMetReader`, so it runs without
   remote data.
3. If it touches the vertical structure of turbulence or the drift, **check it
   against a well-mixed test** (§4) — that is the gate that catches this class of
   bug.
4. Run the full suite and update the tolerance/seed entries on this page.

Tests must not depend on the restricted validation data; the synthetic fixtures
have to be sufficient.
