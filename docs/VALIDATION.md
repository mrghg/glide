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
8. [Golden fixtures](#8-golden-fixtures)
9. [Adding a physics test](#9-adding-a-physics-test)

---

## 1. Running the suite

```bash
.venv/bin/python -m pytest -q                      # everything
.venv/bin/python -m pytest -q tests/test_physics.py # engine primitives only
```

**319 tests across 16 files, about 2.5 minutes, 90% statement coverage, no network access.**
End-to-end tests use synthetic meteorology through `AnalyticMetReader`, so
nothing depends on remote ERA5 or on the restricted validation datasets. Tests
share no state and run order is irrelevant. Stochastic tests are seeded.

---

## 2. What the tests cover

| File | Scope | Tests |
| --- | --- | --- |
| `test_met_reader.py` | Reader mechanics: units, ω→w, heat-flux sign, longitude conventions, multi-store stitching, model-level detection, terrain-following wiring | 54 |
| `test_main_config.py` | Config schema, release expansion, batching | 39 |
| `test_main_runtime.py` | End-to-end on synthetic met: trajectories, well-mixed through the production scheme, static/dynamic parity, graph guards, outputs, memory-guard aborts | 37 |
| `test_hanna.py` | Regime formulae, stability classification, meander, sub-step machinery, per-window caches, compile gating, static/dynamic equivalence | 36 |
| `test_vertical_grid.py` | Terrain-following resample kernels: AGL regrid, sub-surface exclusion, slope correction, hydrostatic model-level pressure | 24 |
| `test_download_sample_cube.py` | Meteorology download helper | 23 |
| `test_convection.py` | Emanuel: thermodynamics, LCL/LNB/CAPE, mass-flux matrix non-divergence and well-mixedness | 21 |
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
| `test_ou_autocorrelation_and_stationarity` | $R(\tau) = e^{-\tau/T_L}$ at $\tau/T_L \in \lbrace 0.5, 1, 2\rbrace $; stationary $\mathrm{Var} = \sigma_w^2$ | $\lvert R - e^{-\tau/T_L}\rvert < 0.02$ (obs ~0.003); variance ±3% | 4111 |
| `test_solid_body_rotation_advection_returns_to_start` | circular trajectory closes after one period; RK2 second order | error ratio > 3.5× per $\Delta t$ halving (obs 4.00); finest return < $10^{-3} r$ | deterministic |
| `test_taylor_dispersion_curve_ballistic_to_diffusive` | $\sigma_z^2(t) = 2\sigma_w^2 T_L[t - T_L(1-e^{-t/T_L})]$ at 6 checkpoints, plus both asymptotes | curve < 5% (obs ~0.1%); ballistic < 5%; diffusive < 8% | 2201 |
| `test_taylor_dispersion_position_integration_bias_with_dt` | forward-Euler position bias | tight at $\Delta t/T_L = 0.01$ (< 2%), bounded at 0.2 (< 15%) | 71 |

### The flagship: an analytic plume footprint — `test_plume_footprint.py`

One backward-plume simulation ($z_r = 50$ m, $U = 5\ \mathrm{m\ s^{-1}}$,
$\sigma_v = \sigma_w = 0.5\ \mathrm{m\ s^{-1}}$, $T_L = 100$ s, 200k particles,
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
flux-form Crank–Nicolson solution of
$\partial c/\partial t = \partial_z(K \partial_z c)$. The release slab
(5–15 m) sits deliberately *inside* the low-diffusivity layer — a mid-column
release was measured to be insensitive to near-surface $K$ and would not have
discriminated.

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
[architecture.md §5](architecture.md#5-making-the-gpu-work-one-step-one-launch)):

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
  $\sigma$ and $T_L$ values against literature forms and the analytic tests pin
  dispersion against closed-form solutions, but a systematic comparison against
  NAME/FLEXPART on identical release setups has not been completed.
- Free-troposphere transport accuracy. The engine-level OU dispersion that the
  Richardson closure feeds is verified; the closure's own $\sigma$ and $T_L$
  magnitudes are not.
- Convective transport magnitude. The matrix is proven mass-conserving and
  well-mixed-preserving; whether it moves the *right amount* of mass is untested
  against a reference.

**One deferred test.** A forward/backward reciprocity check (run a
source–receptor pair both ways and require agreement) would test the backward
formulation itself, which nothing else does directly. It is planned as queue
task A6 ([docs/validation-plan.md](validation-plan.md), Tier 0).

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
| `release.lon` / `lat` / `alt_agl_m`, `release.duration_seconds` | the reference release |
| `simulation.length_seconds` | the reference backward window |

`configs/example_mhd_january.yaml` (single release) and
`configs/example_mhd_january_periodic.yaml` (hourly releases) are both already
aligned with the FLEXPART reference fixture
`data/FLEXPART/FLEXPART_MHD_test_202401.nc`. The fixture is not redistributed
with the repository; see [data/README.md](../data/README.md) for access. It was
driven by different meteorology from ERA5, so differences against it mix
meteorology and model; the ERA5-driven FLEXPART runs of the validation plan
(Tier 2) are the reference for model comparisons.

### Step 2 — convert to STILT units

```python
import xarray as xr
from lpdm.comparison import to_stilt_surface_footprint

fp = xr.open_zarr("outputs/mhd-202401-hourly/footprints.zarr")["footprint"]
one = fp.isel(release=0)  # or .sel() on the release coords

stilt = to_stilt_surface_footprint(
    one,
    surface_layer_depth_m=40.0,  # must equal the bottom z-bin you ran with
    integrate_time=True,
)
```

By default (`output_grid.density_weighting: dry_air`, decision 0022) raw
footprints are already divided by the dry-air density at each particle, in
s m³ kg⁻¹, and this multiplies by $M_\mathrm{air}/h$ to get
$\mathrm{m^2\ s\ mol^{-1}}$ of dry air. A footprint run with
`density_weighting: none` is in plain seconds and needs
`air_density_kg_m3=` (a scalar, or a 2-D field from
`surface_air_density_from_met`), Lin et al. (2003) Eq. 5; the helper reads the
store's `density_weighting` attribute and refuses a density for a weighted
footprint. Keep the attributes through any reduction (`keep_attrs=True`).
FLEXPART's and NAME's footprints are put into the same dry-air units by
`flexpart_to_dry_air_footprint` and `name_to_dry_air_footprint`.

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
- **Density.** A `dry_air` footprint carries the density of each particle as
  it was counted; dividing an unweighted footprint afterwards by one density
  per cell is biased where residence and density co-vary (decision 0022).
- **Release setup.** Keep the release inside the surface layer
  (`alt_agl_m < surface_layer_depth_m`) so "particle not yet mixed" startup
  transients do not dominate.
- **Time resolution.** `integrate_time=False` keeps the `time_ago` axis, which is
  useful for diagnosing *when* the GLIDE plume diverges from the reference.

The validation datasets themselves (NAME, FLEXPART, EDGAR) are not redistributed
with this repository — see [../data/README.md](../data/README.md).

---

## 8. Golden fixtures

A golden fixture freezes the output of a named configuration at one git rev, on
one device, with the configuration's fixed seed. Any later change is compared
with it: a change that should not alter the output must reproduce it, and a
change that should alter it must say by how much (task A0;
[dev/agent/INSTRUCTIONS.md](../dev/agent/INSTRUCTIONS.md) §3). The fixtures
need the ERA5 stores the configurations read, so they are run by hand or by an
agent on the GPU node, not by `pytest`.

### The scripts

All three live in `scripts/agent/` and run the model as a subprocess
(`python -m lpdm.main`) from the checkout they sit in, under `/usr/bin/time -v`.
Source `scripts/gpu_env.sh` first for a CUDA run.

| script | what it does |
| --- | --- |
| `make_golden.py --config <yaml> --device cpu\|cuda [--large]` | runs the config and stores `footprint.npz` (the dense footprint [release, time_ago, z_bin, lat, lon], float32, compressed, with the release times), `endpoint_particles.parquet`, `run_metadata.json` and `manifest.json` in `tests/golden/<config>/<device>/`. With `--large` only the manifest goes there and the data go to `<outputs>/golden/<config>/<device>/` (`$GLIDE_AGENT_OUTPUTS`, or the main checkout's `outputs/`). `--from-output <run>` builds the fixture from a finished run instead. |
| `compare_golden.py <fixture dir> [--output <run>]` | runs the fixture's config on its device at the current checkout (or takes a finished run), checks the fixture's checksums, and prints for every release the maximum absolute and relative footprint differences and the difference of the integrated sensitivity (the release's total), the largest value of each criterion, and PASS or FAIL (exit status 0 or 1). |
| `benchmark.py --config <yaml> --device cuda` | runs with `GLIDE_PHASE_TIMERS=1` and prints, in Markdown, footprints per second (releases ÷ process wall time, start-up, compilation and writing included), the phase table, peak host RSS and peak GPU memory. `--from-log <run>.log` re-reports a finished run. |

The manifest records the git rev (and whether `src/` was modified), the SHA-256
of the YAML file and of the resolved configuration (paths and device left out),
device, seed, torch version, CUDA version and GPU name, the met stores'
provenance from `run_metadata.json` (producer, version, commit, recipe, format,
processing), `GLIDE_COMPILE`, the run's wall time and peak memory, and the
checksum and size of each data file. `compare_golden.py` notes any difference
in configuration hash, torch version, met provenance or device before it
compares.

### Tolerances

| fixture | criterion (defaults) |
| --- | --- |
| CPU (eager) | footprint and endpoint particles bit-identical |
| CUDA (static path) | per footprint cell \|a − b\| ≤ 1e-5 \|b\| + 1e-7 max\|b\| (max over the release), and each release's total within 1e-6 relative; endpoint particles reported, not judged |

CUDA's `scatter_add_` adds the dry-air-weighted contributions (decision 0022)
in an order that varies from run to run, so two same-seed GPU runs differ in
the last bits of the footprint. This was accepted rather than made
deterministic (24 Nov 2026; QUEUE.md A0). Particle positions are not affected:
the endpoint particles of repeated CUDA runs were identical. `--exact`, or
`--rtol`, `--atol-frac` and `--total-rtol`, override the defaults; a task that
changes the numerics states its own tolerance in its spec.

### The fixtures

Regenerated on 24 Nov 2026 for decision 0025 (Hanna density and potential
temperature from each column's own pressure, not the met domain's area mean
per level), on a GH200 (Isambard AI), torch 2.14.1+cu126, CUDA 12.6, ERA5-ARCO
EUROPE stores built by met-bazarr (surface geopotential repaired by v0.4.0), at
`src/` of commit 663ac5e (branch `agent/hanna-column-pressure`). The first
fixtures (A0, `src/` of `main` after PR #59) are superseded; what the change
did to them:

| fixture | new code vs old fixture |
| --- | --- |
| smoke CPU | release total +2.2e-5 relative; 0–40 m total +0.03%; 850 cells differ |
| smoke CUDA | release total +6.2e-5 relative; 836 cells differ |
| reference CUDA | release totals: median +3e-4 (MHD), −8e-5 (RGL), +8e-5 (TAC), largest 3.2e-3; 0–40 m totals per release up to 2.5% (median −0.04%, −0.20%, −0.18%), summed over each site's 48 releases −0.03%, −0.29%, −0.18% |

The three reference sites are low-lying, so the change is small there and the
per-cell differences are mostly the particles' trajectories diverging. Over
high ground it is not small: one release from the Jungfraujoch site (1,000 m
above ground, 5 days back) lost 18% of its 0–40 m sensitivity, against 0.5%
seed-to-seed (decision 0025).

| fixture | config | where | size | repeat run vs fixture |
| --- | --- | --- | --- | --- |
| `smoke_mhd_single_release/cpu` | smoke: MHD, 1 release, 20,000 particles, 1 day back | committed | 0.37 MB | bit-identical (footprint and endpoints) |
| `smoke_mhd_single_release/cuda` | the same, CUDA static path, compiled | committed | 0.37 MB | max cell \|a−b\|/bound 0.28, max rel 2.9e-6, total 3.5e-8; endpoints identical |
| `ab_multisite_perf/cuda` | MHD, RGL, TAC × 48 hourly releases, 20,000 particles, 5 days back, 2 batches | data under `outputs/golden/ab_multisite_perf/cuda` (57 MB; backup in `/projects/u6za/glide-data/golden-fixtures/ab_multisite_perf/cuda`; the superseded A0 data are kept beside both as `cuda-A0-superseded`); manifest committed | — | max cell \|a−b\|/bound 0.44, max rel 4.5e-6, total 4.2e-8 (all 144 releases); endpoints identical |

The CPU and CUDA smoke fixtures are not expected to agree with each other bit
for bit (their totals differ by 2.4e-4 relative); static/dynamic parity is
checked by the runtime tests, not by the fixtures.

### Benchmark baseline

`benchmark.py` on `configs/ab_multisite_perf.yaml`, GH200, `GLIDE_COMPILE=1`,
192 h met cache on the host, the GPU otherwise idle, 24 Nov 2026, at `src/` of
663ac5e (decision 0025; regenerated with the fixtures):

| quantity | value |
| --- | --- |
| device | cuda |
| releases (footprints) | 144 in 2 batch(es), 20000 particles each |
| model steps | 17280 |
| process wall time | 1149.9 s |
| **footprints per second** | **0.1252** |
| peak host RSS | 53.7 GB |
| peak GPU allocated / reserved | 0.69 / 1.65 GiB |

Phase timers: total wall 1134.9 s, 168 met fetches (no CUDA sync: GPU phases under-counted).

| phase | total (s) | % wall | calls | ms/call |
| --- | ---: | ---: | ---: | ---: |
| met_fetch | 893.94 | 78.8 | 17280 | 51.732 |
| gridder | 71.78 | 6.3 | 17280 | 4.154 |
| step | 56.25 | 5.0 | 17280 | 3.255 |
| convection | 14.56 | 1.3 | 288 | 50.552 |
| (residual/untimed) | 98.40 | 8.7 |  |  |

The previous baseline (A0, before decision 0025) was 1,217.7 s, 0.1183
footprints/s, step 3.797 ms/call. The change itself does not make the model
faster per step beyond removing a host-to-device copy of the level pressures
that the density and free-troposphere fields made on every step (they are now
built inside the once-per-window cache); most of the 6% difference is met
fetch time (51.7 vs 54.6 ms/call), which depends on the shared file system and
on what else ran on the node (two CPU-only jobs during the A0 run).

This run is also the run the reference fixture was made from (the phase timers do not change the output). Met fetch dominates. The phase timers do not synchronise the GPU, so the step
and gridder times are under-counted and the residual holds the rest. Compare
like with like: same node type, same cache settings, nothing else on the GPU.

### Phase breakdown with sub-phases (task A1)

Since task A1 the phases nest and each row is the phase's own time
(`lpdm/phase_timers.py`; [architecture.md §8](architecture.md#8-diagnostics-and-knobs)),
so the former 8.7% residual is now named. The same config and settings as the
baseline, 25 Nov 2026, at `src/` of 4388a6e, with `GLIDE_PHASE_TIMERS_SYNC=1`
(the GPU synchronised at every phase boundary, so each phase carries its own
GPU work; another agent's small job shared the GPU):

| quantity | value |
| --- | --- |
| process wall time | 1154.1 s |
| **footprints per second** | **0.1248** (baseline 0.1252) |
| peak host RSS | 53.0 GB |

| phase | total (s) | % wall | calls | ms/call |
| --- | ---: | ---: | ---: | ---: |
| met_fetch | 832.76 | 72.7 | 17280 | 48.192 |
| gridder | 90.57 | 7.9 | 17280 | 5.242 |
| step | 73.93 | 6.5 | 17280 | 4.278 |
| gc | 47.67 | 4.2 | 345 | 138.188 |
| mask_liveness | 44.06 | 3.8 | 51840 | 0.850 |
| convection.lift | 11.08 | 1.0 | 286 | 38.727 |
| loop_overhead | 10.44 | 0.9 | 17280 | 0.604 |
| setup | 7.59 | 0.7 | 1 | 7589.972 |
| wind_mean | 7.49 | 0.7 | 17280 | 0.433 |
| diagnostics | 5.01 | 0.4 | 17280 | 0.290 |
| output_write | 2.79 | 0.2 | 4 | 696.400 |
| saturation_poll | 2.26 | 0.2 | 17282 | 0.131 |
| convection.matrices | 2.11 | 0.2 | 286 | 7.362 |
| convection.columns | 1.82 | 0.2 | 286 | 6.372 |
| batch_setup | 1.15 | 0.1 | 2 | 572.830 |
| convection.redistribute | 1.03 | 0.1 | 286 | 3.616 |
| log_guard | 0.99 | 0.1 | 17280 | 0.058 |
| convection.log | 0.54 | 0.0 | 286 | 1.877 |
| batch_end, convection (own), particle_generation, batch_diagnostics | 0.89 | 0.1 | | |
| (residual/untimed) | 0.56 | 0.0 |  |  |

Convection with its parts is 16.9 s (1.5% of wall, 58.6 ms per met window);
the parcel lift is two thirds of it (38.7 ms per window). The same run without
GPU synchronisation (1371.0 s, 0.1050 footprints/s, slower met fetch at 59.3
ms/call on a cold file cache) charged 93.4 s to `gc` (271 ms per
`gc.collect`) and 37.4 s to `mask_liveness`; with or without
synchronisation the untimed residual is below 0.1% of wall. The per-call cost
of `gc` is Python's full collection over the run's object graph, every 50 steps
(`memory.gc_every_steps`); `mask_liveness` is the active mask, the time-ago
bins and the met-domain escape check, three small elementwise passes over the
1.44 million-particle buffer every step.

**Changing the compiled step changes the CUDA random realisation.** Task A1
added two scalar outputs to the captured step (the substep-saturation count).
On the smoke config the footprint still met the CUDA tolerance and the endpoint
particles were identical, but on `ab_multisite_perf` every particle's
trajectory changed, so the comparison with the reference fixture fails (largest
cell |a−b|/bound 9.0e4). The eager static path is unchanged bit for bit (a
12-hour, 12-release cut of the config, main against the branch, on CUDA with
`GLIDE_COMPILE=0`), and even returning `active_mask.sum()` from the core alters
the compiled result: Inductor generates the step's random numbers per kernel,
so a different kernel layout (here, a reduction over 1.44 million particles
that splits differently from one over 20,000) gives a different, equally valid
stream. Two runs of the new code were identical. Against the fixture the
change is the size of a different random seed: release totals differ by a
median 2.9e-5 (largest 4.5e-4) relative, the 0–40 m layer by a median 0.04%
(largest 0.45%) per release, and summed over each site's 48 releases by
−0.005%, +0.023% and +0.006% (MHD, RGL, TAC). Any later edit to the captured
step will do the same; the reference CUDA fixture then has to be regenerated,
and the CUDA tolerance only covers unchanged compiled code.

---

## 9. Adding a physics test

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
5. On the GPU node, compare with the golden fixtures (§8). A change that should
   not alter the output must pass at the default tolerances; one that should
   must report the differences `compare_golden.py` prints.

Tests must not depend on the restricted validation data; the synthetic fixtures
have to be sufficient.
