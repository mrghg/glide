# GLIDE Validation Notes

Scope of the validation suite, expected tolerances and seeds, and which metrics carry placeholder status pending Milestone 1.

## Running the suite

```bash
.venv/bin/python -m pytest -q tests/
```

The suite runs in well under 30 seconds with no network access. All tests use synthetic met data via `AnalyticMetReader` (see `tests/test_main_runtime.py`). Run order is irrelevant; tests do not share state.

## Test layers

The validation suite is split into three layers, each owned by a separate file.

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
| `test_surface_layer_sigma_w_caps_in_very_stable` | Stable SL capped at 1.3 u\* · 6 | exact | none |
| `test_hanna_above_bl_constants_used` | Above-BL σ = 0.1 m/s, T_L = 100 s | exact | none |

### Footprint gridder tests (`tests/test_footprint.py`)

Seven tests, all deterministic, covering: single-cell binning correctness, total mass conservation for in-bounds particles, inactive-particle exclusion, out-of-bounds rejection, repeat-accumulate summing into the same bin, the silent no-op contract for invalid `t_idx`, and empty-active-mask behaviour.

### Release-generator tests (`tests/test_release_generator.py`)

| Test | Asserts | Tolerance | Seed |
| --- | --- | --- | --- |
| `test_column_release_uniform_sampling_covers_all_levels` | All altitudes sampled in expected proportions when AK is unset | per-level fraction within ±0.05 of uniform | 101 |
| `test_column_release_averaging_kernel_biases_sampling` | AK weights bias sampling proportionally | per-level fraction within ±0.03 of expected | 202 |
| `test_column_release_rejects_negative_altitudes` | Validation of negative altitude inputs | exact (raises `ValueError`) | none |
| `test_column_release_rejects_kernel_length_mismatch` | Validation of AK shape | exact (raises `ValueError`) | none |
| `test_column_release_rejects_zero_total_weight` | Validation of all-zero AK | exact (raises `ValueError`) | none |

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

The Hanna 1982 / FLEXPART scheme (`hanna_1982`) landed in M1 with the unit and end-to-end tests above. The default `--turbulence-scheme` remains `placeholder_constant_ou` until external comparison signs off — see `docs/turbulence.md` §5.3.

**Pending external validation** (don't use as cross-model baselines yet):

- Quantitative endpoint spread, time-height structure, and column-integrated footprint magnitude under Hanna. The unit tests pin local σ/T_L values and the smoke test pins ensemble-mean preservation, but the dispersion magnitude has not been compared to a FLEXPART/STILT reference run on identical met.
- Above-BL transport accuracy: the constant-K placeholder (σ = 0.1 m/s, T_L = 100 s) is intentionally crude and will need a refined N²-based scheme to be defensible (logged as M1.x in `docs/turbulence.md`).

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

## Adding a new physics test

When changing physics in `src/lpdm/gpu_engine.py`, the runtime loop, or the footprint accumulator:

1. Add or update the relevant engine-level test in `tests/test_physics.py` or `tests/test_footprint.py`.
2. If the change affects end-to-end behaviour, add or update a test in `tests/test_main_runtime.py` using `AnalyticMetReader` so it runs without remote ERA5 access.
3. Re-run `.venv/bin/python -m pytest -q tests/` and confirm no regressions.
4. Update the relevant tolerance/seed entries in this file.

When the M1 turbulence rewrite lands, the placeholder list above should be promoted to first-class baselines, with new tolerance entries added here and the corresponding lines moved out of the placeholder section.
