# GLIDE Validation Notes

Scope of the validation suite, expected tolerances and seeds, and which checks are
still to be added (see [Planned synthetic verification tests](#planned-synthetic-verification-tests)
and `dev/TEST_REVIEW_2026-07-16.md` for the full plan).

## Running the suite

```bash
.venv/bin/python -m pytest -q tests/
```

The suite (236 tests at the 2026-07-16 review) runs in ~90 seconds with no network
access. End-to-end tests use synthetic met via `AnalyticMetReader` (see
`tests/test_main_runtime.py`). Run order is irrelevant; tests do not share state.

## Test layers

Twelve files, layered roughly unit → scheme → end-to-end:

| File | Scope |
| --- | --- |
| `test_physics.py` | Engine primitives: RK2 advection (incl. convergence order), OU/Langevin, reflection, well-mixed drift, coordinate normalisation |
| `test_hanna.py` | Hanna scheme: regime formulae, stability classification, meander, substep machinery, per-window caches, compile gating |
| `test_convection.py` | Emanuel reduced scheme: thermodynamics, LCL/LNB/CAPE, mass-flux matrix well-mixedness/non-divergence |
| `test_vertical_grid.py` | Terrain-following resample kernels (Finding 7): AGL regrid, sub-surface exclusion, slope correction |
| `test_met_reader.py` | ARCO reader mechanics: units, ω→w, SHF, lon conventions, multi-store stitching, terrain-following wiring |
| `test_footprint.py` | Gridder: binning, conservation, drop contracts, per-release scatter |
| `test_release_generator.py` | Point/column/batch particle generation, seeding |
| `test_comparison.py` | STILT conversion + conservative regridding |
| `test_main_config.py` | Config schema, release expansion, batching |
| `test_main_runtime.py` | End-to-end runs on synthetic met: advection/trajectory checks, WMC through the production scheme, static/dynamic parity, graph guards, outputs |
| `test_output_writer.py` | Output contracts (Zarr/Parquet) |
| `test_download_sample_cube.py` | Met-download helper |

The most detailed per-test tables below are maintained for the physics-bearing
subset; see the files themselves for the rest.

### Engine unit tests (`tests/test_physics.py`)

| Test | Asserts | Tolerance | Seed |
| --- | --- | --- | --- |
| `test_uniform_wind_advection_rk2_precision` | RK2 exact for constant wind | max abs err `< 1e-9` | 11 |
| `test_rk2_advection_second_order_in_dt` | RK2 second-order convergence under linear wind | err ratio `> 3.5x` per `dt` halving | none (deterministic) |
| `test_zero_wind_diffusion_langevin_gaussian_spread` | OU/Langevin variance and Gaussianity | std within ±15%, skew `< 0.2`, excess kurtosis `< 0.35` | 22 |
| `test_well_mixed_uniformity_in_periodic_turbulence` | Well-mixed uniformity in a periodic domain | rel-RMS `< 0.12` | 33 |
| `test_reflect_surface_handles_boundary_cases` | Surface reflection at `z=0` | exact | none |
| `test_reflect_surface_nonzero_z` | Reflection at non-zero `z_surface` | exact | none |
| `test_apply_horizontal_turbulence_displaces_with_cos_lat_correction` | Horizontal-perturbation primitive: cos-lat correction, sign of backward integration, mass invariance | exact (analytic) | none |

### Hanna scheme unit tests (`tests/test_hanna.py`)

| Test | Asserts | Tolerance | Seed |
| --- | --- | --- | --- |
| `test_hanna_is_registered` | HannaScheme self-registers under `hanna_1982` and declares `("t","ustar","shf")` | exact | none |
| `test_coriolis_parameter_signs_and_magnitude` | f sign per hemisphere; magnitude at lat=45° | abs err `< 1e-12` | none |
| `test_air_density_standard_conditions` | ρ at 288 K, 101325 Pa ≈ 1.225 kg/m³ | abs err `< 0.01` | none |
| `test_obukhov_length_signs` | L > 0 stable, L < 0 unstable, ±∞ neutral | exact | none |
| `test_convective_velocity_zero_when_stable_or_neutral` | w\* = 0 for H ≤ 0; magnitude check for H > 0 | abs err `< 1e-6` | none |
| `test_in_bl_sigma_at_z_zero_matches_ustar_scaling` | σ_w = 1.3 u\*, σ_uv = 2.0 u\* at z=0 | abs err `< 1e-6` | none |
| `test_in_bl_stable_sigma_vanishes_at_top_of_bl` | σ_w → 0 at z = h (stable) | `< 1e-2` | none |
| `test_in_bl_regime_selection_at_boundaries` | Regime branching at h/L = ±1 boundaries | qualitative | none |
| `test_surface_layer_sigma_w_regimes` | SL formulae per regime match analytic forms | abs err `< 1e-6` | none |
| `test_surface_layer_sigma_w_stable_is_constant_in_height` | Stable SL σ_w constant with height (FLEXPART v11 convention; 2026-07-02 physics fixes) | exact | none |
| `test_hanna_above_bl_constants_used` | Above-BL σ = 0.1 m/s, T_L = 100 s | exact | none |

### Footprint gridder tests (`tests/test_footprint.py`)

Ten tests, all deterministic, covering: single-cell binning correctness, total mass conservation for in-bounds particles, inactive-particle exclusion, out-of-bounds rejection, repeat-accumulate summing into the same bin, the silent no-op contract for invalid `t_idx`, empty-active-mask behaviour, non-uniform z-edge binning (each interval treated as its own bin via `torch.bucketize`), and constructor validation rejecting non-ascending or too-short edge lists.

### Release-generator tests (`tests/test_release_generator.py`)

| Test | Asserts | Tolerance | Seed |
| --- | --- | --- | --- |
| `test_column_release_uniform_sampling_covers_all_levels` | All altitudes sampled in expected proportions when AK is unset | per-level fraction within ±0.05 of uniform | 101 |
| `test_column_release_averaging_kernel_biases_sampling` | AK weights bias sampling proportionally | per-level fraction within ±0.03 of expected | 202 |
| `test_column_release_rejects_negative_altitudes` | Validation of negative altitude inputs | exact (raises `ValueError`) | none |
| `test_column_release_rejects_kernel_length_mismatch` | Validation of AK shape | exact (raises `ValueError`) | none |
| `test_column_release_rejects_zero_total_weight` | Validation of all-zero AK | exact (raises `ValueError`) | none |

### Comparison utility tests (`tests/test_comparison.py`)

Cover the post-processing helpers in `src/lpdm/comparison.py` used to put GLIDE footprints onto common ground with FLEXPART / NAME / STILT references.

| Test | Asserts | Tolerance | Seed |
| --- | --- | --- | --- |
| `test_to_stilt_full_overlap_thin_surface_bin` | Exact STILT-style conversion when a single z-bin matches the surface layer | rel `< 1e-12` | none |
| `test_to_stilt_partial_overlap_with_thicker_bin` | Partial-overlap depth-weighting for thick bins | rel `< 1e-12` | none |
| `test_to_stilt_time_integration_flag` | `integrate_time=False` keeps `time_ago` axis and is consistent with the integrated form | exact | none |
| `test_to_stilt_rejects_missing_z_edges_coords` | Missing `z_bottom_m` / `z_top_m` coords fail loud | exact (raises) | none |
| `test_to_stilt_rejects_nonpositive_surface_depth` | `surface_layer_depth_m <= 0` rejected | exact (raises) | none |
| `test_to_stilt_records_conversion_metadata` | Conversion attrs (units, depth, density, reference) attached | exact | none |
| `test_regrid_identity_when_grids_match` | Regridding to the same grid returns values unchanged | abs `< 1e-12` | none |
| `test_regrid_coarsen_preserves_total` | Fine → coarse mass conservation | rel `< 1e-12` | none |
| `test_regrid_refine_preserves_total` | Coarse → fine mass conservation | rel `< 1e-12` | none |
| `test_regrid_redistributes_to_overlapping_targets` | Localised pulse splits proportionally between target cells | abs `< 1.0` | none |
| `test_regrid_preserves_extra_dimensions` | Per-frame conservation across leading time/z dims | rel `< 1e-12` | none |
| `test_regrid_lat_cosine_area_factor` | Conservation holds at high latitudes where `cos(lat)` curvature matters | rel `< 1e-12` | none |
| `test_regrid_zero_outside_source_extent` | Target cells outside source extent return zero | exact | none |
| `test_regrid_rejects_non_ascending_centres` | Non-ascending centres rejected | exact (raises) | none |

### End-to-end runtime tests (`tests/test_main_runtime.py`)

| Test | Asserts | Tolerance | Seed |
| --- | --- | --- | --- |
| `test_run_completes_with_synthetic_met` | Full loop (placeholder scheme) produces all four output artifacts | exact (file existence) | 42 |
| `test_preflight_rejects_insufficient_met_coverage` | Preflight raises when met coverage doesn't span the run | exact (raises `PreflightValidationError`) | 42 |
| `test_constant_wind_advection_trajectory` | Mean lon matches analytic backward transport (placeholder scheme) | `< 5e-3` deg | 42 |
| `test_footprint_total_matches_active_particle_time` | Footprint sum equals trajectory active-time integral | rel err `< 1e-5` | 42 |
| `test_hanna_run_completes_with_synthetic_met` | Hanna scheme runs end-to-end and produces all four artifacts | exact (file existence) | 42 |
| `test_hanna_constant_wind_preserves_mean_trajectory` | Hanna's zero-mean perturbations don't bias ensemble mean lon | `< 1e-2` deg | 42 |
| `test_hanna_produces_nontrivial_vertical_spread` | Convective-regime Hanna produces vertical spread ≥ 50 m | qualitative | 42 |

## Validation status

`hanna_1982` is the production turbulence scheme (all example configs use it;
`turbulence.scheme` is a required YAML field — there is no CLI scheme flag).
`placeholder_constant_ou` is kept only as a regression pin and to isolate runtime
plumbing from Hanna in a few end-to-end tests.

**Pending external validation** (don't use as cross-model baselines yet):

- Quantitative endpoint spread, time-height structure, and column-integrated footprint magnitude under Hanna. The unit tests pin local σ/T_L values and the smoke test pins ensemble-mean preservation; systematic comparison against NAME/FLEXPART is in progress (and the terrain-affected v2/v3 comparisons need re-running after the Finding-7 fix — see `dev/CHECKPOINT.md`).
- Free-troposphere transport accuracy: the Richardson-closure FT scheme is implemented, but its dispersion magnitude is untested against theory until T1/T5a below land.

## Planned synthetic verification tests

From the 2026-07-16 test-suite review. Detailed instructions — analytic targets,
setups, tolerances, pitfalls, order of attack — live in
**`dev/TEST_REVIEW_2026-07-16.md`**; each test gains a tolerance/seed row in the
tables above when it lands.

| ID | Test | Analytic target | Chain covered | Status |
| --- | --- | --- | --- | --- |
| T5a | `test_ou_autocorrelation_and_stationarity` | R(τ)=e^(−τ/T_L); Var(w′)=σ_w² | engine OU update | **LANDED** |
| T5b | `test_solid_body_rotation_advection_returns_to_start` | circular trajectory, period 2π/ω, O(dt²) | RK2 in spatially varying wind, backward sign | **LANDED** |
| T1 | `test_taylor_dispersion_curve_ballistic_to_diffusive` (+ `..._position_integration_bias_with_dt`) | σ_z²(t)=2σ_w²T_L[t−T_L(1−e^(−t/T_L))], ballistic→diffusive | engine OU + vertical displacement | **LANDED** |
| T4 | `test_terrain_hill_footprint_continuous_and_agl_preserved` | constant-AGL transit over a hill; no Finding-7 hole | reader terrain resample → advection → gridder (e2e) | **PLANNED — gate for the GH200 terrain acceptance run** |
| T2 | `test_footprint_matches_analytic_gaussian_plume` | f(x,y)=exp(−y²/2σ_y²)exp(−z_r²/2σ_z²)/(πσ_yσ_zU) | advection+OU+reflection+gridder+STILT units (flagship) | **PLANNED** |
| T3 | `test_langevin_diffusion_limit_matches_pde` | ∂c/∂t=∂_z(K∂_zc), Crank–Nicolson reference | inhomogeneous K + drift + reflection; near-surface K bias class | **PLANNED** |
| T6 | forward/backward reciprocity | forward concentration ≡ backward footprint × source | backward formulation itself | **DEFERRED** |

**Note (revised from the plan):** T1/T5a were specced to drive through
`HannaScheme.step`, but Hanna has *no homogeneous regime* — T_L is intrinsically
height-dependent (the inhomogeneity the Thomson drift corrects), so OU/Taylor
statistics have no closed form through the assembled scheme. They are therefore
verified at the engine-OU level with prescribed constant (σ_w, T_L); the scheme's
substep integration is covered by the well-mixed tests (inhomogeneous, full step)
and the static/dynamic substep-equivalence tests.

### Analytic dispersion tests (`tests/test_dispersion_analytic.py`)

| Test | Asserts | Tolerance | Seed |
| --- | --- | --- | --- |
| `test_ou_autocorrelation_and_stationarity` | OU R(τ)=e^(−τ/T_L) at τ/T_L∈{0.5,1,2}; stationary Var=σ_w² | `|R−e^(−τ/T_L)|<0.02` (obs ~0.003); Var within ±3% | 4111 |
| `test_solid_body_rotation_advection_returns_to_start` | Circle closure after one period; RK2 second order | err ratio `>3.5×` per dt-halving (obs 4.00); finest return `<1e-3·r` | none (deterministic) |
| `test_taylor_dispersion_curve_ballistic_to_diffusive` | σ_z(t) vs Taylor at 6 checkpoints; ballistic σ_w·t; diffusive 2Kt | curve `<5%` (obs ~0.1%); ballistic `<5%`; diffusive `<8%` | 2201 |
| `test_taylor_dispersion_position_integration_bias_with_dt` | forward-Euler position bias: tight at dt/T_L=0.01, bounded at 0.2 | fine `<2%`, coarse `<15%` | 71 |

Housekeeping planned alongside (same doc): legacy-flag smokes
(`surface_layer_override`, `flexpart_tl_floors`), `wind_mean`-cache unit test,
Emanuel stable-winter-column guard, memory-guard trip test, one `pytest --cov`
coverage map.

**Stable across runs (with seed):**

- Mean particle position (advection-only and Hanna-with-zero-mean-perturbations)
- Surface reflection behaviour
- Footprint binning, conservation, time-bin attribution, OOB rejection
- Met-coverage preflight behaviour
- Hanna stability classification, regime selection, and per-regime σ/T_L formulae (unit tests above)
- Output artifact contracts (Parquet schema, Zarr dimensions, persisted coordinates)
- `ColumnRelease` sampling distribution

## End-to-end run with sample met

For a manual end-to-end check using `data/sample_met.zarr`, see the "Local Smoke Test" section of `README.md`. That command exercises the full pipeline against real (cropped) ERA5 data and writes the four output artifacts under `outputs/demo-run-local`. It is complementary to the synthetic-met tests above; the test suite covers correctness, the smoke command covers met-reader integration with a real Zarr store.

## Comparing against FLEXPART / NAME / STILT

GLIDE accumulates the residence-time footprint *directly onto a configurable target grid* (`output_grid` in the run YAML), so the most common path is to set the output grid equal to the reference's grid and skip regridding entirely. The `lpdm.comparison.regrid_conservative` helper is still available in the module for off-grid use cases. End-to-end workflow:

### 1. Author a run config matching the reference setup

Two starting points ship with the repo, both wired to the bundled FLEXPART fixture (`data/FLEXPART/FLEXPART_MHD_test_202401.nc`):

- `configs/example_mhd_january.yaml` — single release. Run one of the FLEXPART reference release times at a time; `--output-uri` and `--start-time` overrides let you iterate over the 96 reference release times manually.
- `configs/example_mhd_january_periodic.yaml` — 744 hourly releases for January 2024 via `kind: "periodic_point"`. One invocation, one 5D `footprints.zarr` indexed by `release_time`. Slice with `isel(release_time=…)` to align with FLEXPART's release times.

The key sections to align with the reference:

- `output_grid.{lon_bounds, lat_bounds, n_x, n_y}` — set to the reference grid's *outer* cell edges (FLEXPART/NAME usually store cell centres, so add half a cell at each edge).
- `output_grid.z_edges_m` — make the bottom edge pair match the reference's surface-layer depth (0–40 m for FLEXPART / NAME). Direct accumulation into that bin makes the downstream STILT-unit conversion exact (no overlap-fraction approximation).
- `release.point.{lon, lat, alt_agl_m}` and `release.duration_seconds` — match the reference's release.
- `simulation.length_seconds` — match the reference's backward window (per release, in the periodic case).
- `turbulence.scheme: hanna_1982`.

Then run:

```bash
.venv/bin/python -m lpdm.main --config configs/example_mhd_january_periodic.yaml
```

CLI overrides: `--device {auto|cpu|cuda|mps}`, `--output-uri PATH`, `--start-time ISO`. Anything else is set in the YAML.

### 2. Convert the raw footprint to STILT units

```python
import xarray as xr
from lpdm.comparison import to_stilt_surface_footprint

# All GLIDE footprints are 5D (release_time, time_ago, z_bin, lat, lon). For
# single-release runs the release_time axis is length 1; for the periodic config
# below it's 744 and you select one release to compare against the FLEXPART
# fixture's matching timestamp.
glide_5d = xr.open_zarr("outputs/mhd-202401-hourly/footprints.zarr")["footprint"]
glide_raw = glide_5d.isel(release_time=0)  # or .sel(release_time="2024-01-01T00")

# Raw footprint is in `s` per cell. Lin 2003 Eq. 5 converts to m^2 s mol^-1
# (equivalent to (mol/mol)/(mol/m^2/s)). surface_layer_depth_m must equal the
# bottom z-bin you ran with. Density: 1.225 kg/m^3 for standard conditions,
# or derive from sp / (R_d * T) per cell using the run's met fields.
glide_stilt = to_stilt_surface_footprint(
    glide_raw,
    surface_layer_depth_m=40.0,
    air_density_kg_m3=1.225,
    integrate_time=True,
)
```

### 3. Compare cell-for-cell

```python
import numpy as np

ref = xr.open_dataset("data/FLEXPART/FLEXPART_MHD_test_202401.nc", engine="h5netcdf")
ref_field = ref["srr"].sel(time="2024-01-01T00:00:00").sum("time")  # match GLIDE's release window

diff = glide_stilt - ref_field
print(f"GLIDE total:     {float(glide_stilt.sum()):.3e}")
print(f"Reference total: {float(ref_field.sum()):.3e}")
print(f"RMSE:            {float(np.sqrt((diff ** 2).mean())):.3e}")
print(f"Correlation:     {float(xr.corr(glide_stilt, ref_field)):.3f}")
```

If the output grid was authored to match the reference, no regridding step is needed. If you can't (or don't want to) align grids — e.g. comparing two GLIDE runs against a single reference — `lpdm.comparison.regrid_conservative` performs mass-conservative area-weighted regridding for rectangular lat/lon grids and is unit-tested in `tests/test_comparison.py`.

### Tips & caveats

- **Output grid bounds are outer cell edges**: GLIDE's `output_grid.lon_bounds` / `lat_bounds` are the outermost edges, with `n_x`, `n_y` equal cells filling the interval. Reference outputs often label coordinates by cell centres — add half a cell on each side when authoring the YAML so your cells align.
- **Mismatched surface-layer depth**: if the reference uses a layer other than 0–40 m, set `surface_layer_depth_m` (and the matching `z_edges_m` bottom bin) accordingly. If you can't make them match exactly, the converter depth-weights overlapping bins (approximate, assumes uniform residence-time density within each bin).
- **Spatial air-density variation**: for runs spanning large lat or elevation ranges, replace the scalar `air_density_kg_m3` with a 2D `xarray.DataArray` of `sp / (R_d * T_surface)` from the met.
- **Different met**: GLIDE streams ARCO ERA5 *pressure-level* fields, resampled onto a fixed terrain-following AGL grid per met window (Finding 7, 2026-07-16); FLEXPART runs typically use ECMWF operational analyses on native model levels; NAME uses UM analyses. Inter-model met differences contribute to footprint differences that aren't due to the turbulence scheme — note this in any documented tolerance.
- **Different release setup**: keep the release within the surface layer (`release.alt_agl_m < surface_layer_depth_m`) to avoid "particle-not-yet-mixed" startup transients dominating the comparison.
- **Time resolution**: pass `integrate_time=False` if you want time-resolved sensitivity; useful for diagnosing when the GLIDE plume diverges from the reference.

## Adding a new physics test

When changing physics in `src/lpdm/gpu_engine.py`, the runtime loop, or the footprint accumulator:

1. Add or update the relevant engine-level test in `tests/test_physics.py` or `tests/test_footprint.py`.
2. If the change affects end-to-end behaviour, add or update a test in `tests/test_main_runtime.py` using `AnalyticMetReader` so it runs without remote ERA5 access.
3. Re-run `.venv/bin/python -m pytest -q tests/` and confirm no regressions.
4. Update the relevant tolerance/seed entries in this file.

When the M1 turbulence rewrite lands, the placeholder list above should be promoted to first-class baselines, with new tolerance entries added here and the corresponding lines moved out of the placeholder section.
