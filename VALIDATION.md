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
| `test_run_completes_with_synthetic_met` | Full loop produces all four output artifacts | exact (file existence) | 42 |
| `test_preflight_rejects_insufficient_met_coverage` | Preflight raises when met coverage doesn't span the run | exact (raises `PreflightValidationError`) | 42 |
| `test_constant_wind_advection_trajectory` | Mean lon matches analytic backward transport | `< 5e-3` deg | 42 |
| `test_footprint_total_matches_active_particle_time` | Footprint sum equals trajectory active-time integral | rel err `< 1e-5` | 42 |

## Placeholder metrics pending Milestone 1

The current turbulence step uses hard-coded constants (`t_lagrangian = 300 s`, `sigma_w2 = 1.0 m^2/s^2`) and has no horizontal stochastic diffusion. Until M1 replaces this with a met-driven scheme, the following metrics are placeholder and **must not** be used as scientific baselines or for cross-model comparison:

- Endpoint particle spread (lon, lat, alt)
- Time-height integrated footprint structure
- Column-integrated footprint magnitude and spatial extent
- Any external comparison against FLEXPART/STILT

The following are valid now and stable across runs (with seed):

- Mean particle position (advection-only)
- Surface reflection behaviour
- Footprint binning, conservation, time-bin attribution, OOB rejection
- Met-coverage preflight behaviour
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
