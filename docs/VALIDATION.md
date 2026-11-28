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
10. [Validation figures and the gallery](#10-validation-figures-and-the-gallery)

---

## 1. Running the suite

```bash
.venv/bin/python -m pytest -q                      # everything
.venv/bin/python -m pytest -q tests/test_physics.py # engine primitives only
GLIDE_SLOW_TESTS=1 .venv/bin/python -m pytest -q   # also the slow full-strength tests
```

Tests marked slow (minutes each on CPU) are skipped unless `GLIDE_SLOW_TESTS=1`
is set; they are full-strength versions of default-suite gates (§4).

**597 tests on `main` at d19755d (26 Nov 2026): 583 passed and 14 skipped (local meteorology, DATEM and ETEX data absent), 14 minutes on 8 CPU threads, no network access.** This count predates PR #72, which split the boundary-layer-top gate into a default tier and a slow tier. The statement-coverage figure (90%) dates from an earlier, smaller suite and has not been remeasured.
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
| `test_near_surface_well_mixed.py` | Near-surface well-mixed gate through the production Hanna scheme at $\Delta t = 60$ s, three regimes | 3 |
| `test_bl_top_well_mixed.py` | Boundary-layer-top well-mixed gate: open column uniform in $[0, 2h]$ through the production Hanna scheme at $\Delta t = 60$ s, three regimes; a full-strength copy is slow (`GLIDE_SLOW_TESTS=1`) | 3 + 3 slow |
| `test_reciprocity.py` | Forward–backward reciprocity through the production Hanna scheme (forward-in-time test mode; exact pathwise check of the backward sign construction; the plan's criteria and companions are a slow release check, `GLIDE_SLOW_TESTS=1`) | 8 (+8 slow) |
| `test_cape_cloud_top.py` | CAPE bounded at the diagnosed cloud top (decision 0031): the review's two soundings (35 J/kg in the cloud, one with 30 J/kg more above its top) neither fires at 50 J/kg; scalar and batched CAPE agree and equal the cloud's positive buoyancy; edge cases (no LNB, LCL at the top level, a weak one-level dip ends the cloud) | 10 |
| `test_convection_closure.py` | Decision 0033: the matrix exponential of the rate generator equals the true exponential, is non-negative, row-stochastic and keeps layer masses stationary; backward = time reversal of forward and the small-step limit of the Euler backward matrix (T1 cycle); 3 h = three 1 h; the scheme's hourly transport within 15% on 23/40/79 levels (old closure ×1.5+ per step); production defaults; the pre-0033 options reproduce the old matrix; Courant numbers beyond the squaring cap counted | 20 |
| `test_sigma_floor_gradient.py` | The drift gradient at $h$ and the σ_w floor (decision 0030): analytic gradient in three regimes, bit identity with the stencil elsewhere, static/dynamic parity, transfer-symmetry gate across a stable $h$ | 33 |

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
| `test_footprint_matches_analytic_gaussian_plume` | crosswind-integrated columns per travel-time band; 2-D correlation | 2–5 $T_L$: max < 5% (obs 0.4%); ≥ 5 $T_L$: max < 15% (obs ≤ 10%: single columns that receive 9 or 11 time samples instead of 10, because along-wind motion is deterministic and particle positions fall on cell edges; not particle noise); corr > 0.995 (obs 0.9985) |
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

### Forward–backward reciprocity — `test_reciprocity.py`

The backward footprint of a receptor should equal the concentration a forward
run puts at that receptor from a unit source in each cell (Flesch, Wilson &
Yee 1995; [physics.md §4](physics.md#4-running-backward-in-time)). The forward
run uses the production `HannaScheme` with a private flag,
`_forward_in_time`, that no configuration can set: the drift unnegated and the
displacement and advection $+\Delta t$, with reflection, the boundary-layer-top
interface and the velocity carry shared. The column is steady and horizontally
uniform (Hanna profile below $h$, the free-troposphere closure above, a
hydrostatic density, a westerly wind from 3 m s⁻¹ at the ground to 6 m s⁻¹ at
$h$); cells are rows $h/10$ deep up to $1.3h$ and 3 km columns. Both
directions count $\Delta t/\rho_d$ for 2 h at $\Delta t$ = 60 s and 16
sub-steps and follow validation plan §3: two selection runs, one backward and
one forward, each with its own seeds, keep the cells that at least 50
different particles pass through in both (the plan states the rule for the
backward run; it is applied to the forward map too); both maps are
bootstrapped (400 resamples); the test passes if at least 90% of the tested
cells and the sum over them agree within their 95% noise range. 40,000
backward particles; 10,000 forward particles per source row (13 rows).

**What it tests.** In a steady, horizontally uniform column the backward step
is, pathwise, the forward step with $w' \to -w'$: a backward run with noise
$\eta$ equals the forward run with $-\eta$ and negated start velocities, bit
for bit (derived; measured over 120 steps, `outputs/agents/A6/review/pathwise.log`;
asserted by the fourth test below). With a start distribution symmetric in
$w'$ the backward run is therefore the forward process in law, so the
comparison checks (i) the sign construction and (ii) the transfer symmetry of
the one discrete scheme both directions share,
$\rho_s K(s \to r) = \rho_r K(r \to s)$ between cells. It cannot say whether
an error is "forward" or "backward". Two limits:

- The harness starts both directions from the equilibrium (Eulerian)
  turbulent velocities, as the theorem assumes. Production releases start at
  $w' = 0$; what that does near the receptor is not measured.
- The column is steady with a fixed $h$. A moving-$h$, time-varying case is
  the outstanding test of the backward formulation itself.

| Test | Asserts |
| --- | --- |
| `test_forward_mode_is_private_and_off_by_default` | the flag is not a constructor argument and defaults to backward |
| `test_forward_step_static_and_dynamic_paths_agree` | forward, the CUDA-graph path and the CPU path agree (bit-identical but for the last bits of longitude, as backward) |
| `test_forward_mode_flips_the_drift_and_the_displacement_only` | forward coefficients are the backward ones with the two drift pieces negated |
| `test_backward_step_is_the_forward_step_with_the_velocity_negated` (8) | the sign construction, exactly: backward with $\eta$ equals forward with $-\eta$ and negated start velocities, bit for bit (positions after every step, $u', v', w'$ of opposite sign), stable and convective, both step paths, positions in degrees and in the local metric frame (task A4), 30 steps with ground reflection and crossings of $h$; with the backward sign flip removed it breaks (about 5 s each) |
| `test_removing_the_backward_sign_flip_breaks_reciprocity_quick` | quick companion (about 55 s on 8 threads): without the backward sign flip ≤ 30% of the cells agree (measured 5–8%, 3 seeds), with it ≥ 80% (89–95%) |
| `test_backward_footprint_equals_forward_concentration` (slow, 5 cases) | **release check**, the plan's criteria; does not pass at the defaults (below) |
| `test_reciprocity_fails_when_the_physics_is_broken` (slow, 3 companions) | the plan's criteria fail without the sign flip, without the interface rule, and with the drift stencil straddling $h$ |

**The slow cases are an explicit release check** (`GLIDE_SLOW_TESTS=1`,
about 15 minutes for all eight on 8 threads), not part of the default suite.
A6 stays open against the plan's criterion: the results below are recorded as
they are, and nothing was tuned to make them pass.

Results at main's physics (this branch's backward steps are bit-identical to
main d19755d; seed 0; `scripts/validation/reciprocity/run_cases.py`, outputs
in `outputs/agents/A6/revise/main/`, measured). "Selected" is the cells each
selection run keeps; "agree" the fraction of tested cells inside their range;
"sum" backward over forward summed over the tested cells, with its 95% noise
range.

| Case (receptor row) | Selected bwd / fwd | Tested | Agree | Sum | Plan criteria |
| --- | --- | --- | --- | --- | --- |
| convective, 0–150 m | 153 / 144 | 144 | 0.882 | 1.0162 (−0.007, +0.006) | fail |
| convective, $h$ to $h$ + 150 m | 179 / 160 | 160 | 0.938 | 0.9977 (−0.007, +0.008) | pass |
| stable, 0–40 m | 116 / 108 | 108 | 0.769 | 0.9778 (±0.009) | fail |
| stable, $h$ − 40 m to $h$ | 158 / 142 | 142 | 0.852 | 1.0113 (±0.010) | fail |
| stable, $h$ to $h$ + 40 m | 113 / 64 | 64 | 0.922 | 1.0038 (±0.009) | pass, but blind to the boundary layer (below) |
| companion: no backward sign flip (convective, 0–150 m) | 155 / 144 | 144 | 0.042 | 0.9947 | fail, as it must |
| companion: no interface rule (convective, above $h$) | 171 / 185 | 171 | 0.175 | 0.2423 | fail, as it must |
| companion: stencil straddling $h$ (convective, above $h$) | 182 / 32 | 32 | 0.844 | 0.9968 | fail, but only just (below) |

With PR #75 merged in (decision 0030, the analytic drift gradient at the
$\sigma_w$ floor; not merged into main or into this branch: a scratch merge of
`agent/SF-sigma-floor-gradient` 505177c with this branch, same seeds, outputs
in `outputs/agents/A6/revise/pr75/`, measured). The last column repeats each
case's agreement over the cells the backward selection alone keeps (the
procedure before the forward rule), main → PR #75.

| Case (receptor row) | Selected bwd / fwd | Tested | Agree | Sum | Plan criteria | Agree, backward-selected cells |
| --- | --- | --- | --- | --- | --- | --- |
| convective, 0–150 m | 153 / 144 | 144 | 0.882 | 1.0162 (−0.007, +0.006) | fail | 0.876 → 0.876 |
| convective, $h$ to $h$ + 150 m | 179 / 160 | 160 | 0.938 | 0.9977 (−0.007, +0.008) | pass | 0.927 → 0.927 |
| stable, 0–40 m | 116 / 108 | 108 | 0.769 | 0.9778 (±0.009) | fail | 0.776 → 0.776 |
| stable, $h$ − 40 m to $h$ | 159 / 141 | 141 | 0.894 | 1.0131 (±0.010) | fail | 0.861 → 0.899 |
| stable, $h$ to $h$ + 40 m | 107 / 66 | 66 | 0.924 | 1.0013 (±0.009) | pass, blind to the boundary layer | 0.673 → 0.897 |
| companions (no sign flip; no interface rule; straddling stencil) | as at main | | 0.042; 0.175; 0.844 | 0.9947; 0.2423; 0.9968 | fail, as they must | |

The other three cases change only in the fifth digit of the sum. Near the stable
top the fix moves the cells the defect touched towards agreement: for the
receptor above $h$ the boundary-layer rows 7–8 go from backward/forward 1.31
and 1.69 to 1.03 and 1.43, and for the receptor below $h$ the row above $h$
from 0.86 to 0.95 (measured, the two `stable_*_h.npz`). With PR #75, three of
the five cases still fail the plan's criteria; the stable just-below-$h$ case
by a small margin (0.894 against 0.90, and its sum 1.3% high against a ±1.0%
range). Its remaining disagreements are away from $h$: 9 of the 15 failing
cells are in row 5 (200–240 m; backward/forward 1.10 there at both main and
PR #75), whose cause is not identified (measured, `pr75/stable_below_h.npz`).

What the failures are:

- **Near the ground: the scheme's near-surface well-mixed error.** The
  backward footprint summed over the tested cells is the total residence
  time, which is conserved; the forward sum is the scheme's occupancy of the
  receptor relative to its well-mixed share. A column started well mixed
  holds 0.981 of its share in the lowest 150 m (convective, 16 sub-steps,
  `outputs/agents/A6/explore/wm_dir_convective.json`, measured), and the
  convective surface sum is 1.016–1.018 over three seeds (1.008 at
  64 sub-steps, one seed; measured, `outputs/agents/A6/explore/`, before
  the forward selection rule). We read this (argued) as the first-order
  sub-step error of
  [decision 0027](../dev/decisions/0027-fixed-substep-count.md); the
  occupancy at 32 and 64 sub-steps (0.989, 0.988, one seed each) does not
  resolve its convergence beyond 32.
- **Near a stable boundary-layer top: the drift defect at the $\sigma_w$
  floor.** In this stable column $\sigma_w = 1.3u_*(1 - z/h)$ reaches its
  floor (10⁻³ m s⁻¹) 1.23 m below $h$ (derived), and the production ±1 m
  drift stencil puts a drift inside that floor layer, where the floored
  profile has none, and over-states it 5.5 times at its edge (measured,
  `outputs/agents/A6/review/drift_profile.log`). This is the defect that
  decision 0030 (PR #75, not yet merged) addresses: a defect of the
  production drift, not a tall-tower question. At row level (cells
  summed along the wind, 40,000 particles per row) the scheme's transfer
  across $h$ is asymmetric: between each of the boundary-layer rows 6–8 and
  the row above $h$, the downward residence exceeds the upward one by a factor
  1.38–1.75 (±0.11–0.18) forward at 16 sub-steps, and 1.47–1.65 backward,
  which is the same process; at 256 sub-steps the factor is 1.15–1.25
  (±0.12–0.17). With PR #75 merged it is 0.99–1.21 forward and 1.00–1.25
  backward at 16 sub-steps (measured; main from
  `outputs/agents/A6/review/rowmat/`, PR #75 from
  `outputs/agents/A6/revise/pr75/rowmat/`; summary in
  `outputs/agents/A6/revise/logs/pr75_rowmat_summary.log`). The density
  changes by under 4% across these rows (derived), and PR #75 expects about
  1.15 at 16 sub-steps from decision 0027's sub-step error in its own column.
  - With both selection rules the stable above-$h$ case passes at main, but
    only because the forward rule removes every boundary-layer cell: rows
    0–6 get too few of the 10,000 forward particles per source row, so only
    rows 7–12 are tested. Of the 5 cells left in rows 7–8, 1 agrees, and
    their backward/forward ratios summed by row are 1.31 and 1.69 (measured,
    `outputs/agents/A6/revise/main/stable_above_h.npz`). At these counts the
    cell test cannot see the defect for this receptor; the row matrices above,
    and PR #75's transfer-symmetry gate, can.
  - The earlier statement that the backward footprint in the boundary-layer
    cells is "1.5–3 times" the forward concentration is not resolved by
    particle noise in rows 0–4: those cells receive a few to a few tens of
    forward particles each, and two independent forward runs differ by up to
    a factor 2.8 there (measured, `outputs/agents/A6/review/null_stable_above_h.json`).
  - At main the stable just-below-$h$ case fails mainly in rows 5 and 10
    (4 of 13 and 9 of 15 tested cells agree; measured,
    `main/stable_below_h.npz`); PR #75 fixes row 10 but not row 5 (above).
- **The forward rule blinds the test where forward transfer is suppressed.**
  A cell is dropped when too few forward particles reach the receptor from
  it, which is exactly what a suppressed upward transfer does. The stable
  above-$h$ case above is one example. The straddling-stencil companion is
  another: the forward rule keeps only rows 10–11 (32 cells), so it fails
  with 0.844 of the cells agreeing; on the 182 backward-selected cells 0.148
  agree (measured, `teeth_straddling_stencil_convective_above_h.npz`;
  0.148 and sum 1.126 before the forward rule, `outputs/agents/A6/cases/`).
  Restoring coverage needs more forward particles from the boundary-layer
  rows; that is a follow-up, not done here.
- **The 90% criterion is tight for this procedure.** Comparing two
  independent runs of the *same* direction (an exact null) through the same
  backward selection and bootstrap gave 92.2% and 95.0% (forward) and 92.7% and
  96.6% (backward) of the cells inside their ranges (convective, receptor above
  $h$, two repeats, `outputs/agents/A6/explore/nullcal_conv10.json`): the
  cells of a row share their forward particles, so the fraction varies by
  several points between repeats, and the bootstrap may under-cover a little.
  A correct model therefore fails the 90% criterion now and then at these
  counts.

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
| `test_near_surface_well_mixed_at_60_s` (`test_near_surface_well_mixed.py`) | at the shipped $\Delta t = 60$ s and the default sub-step count, a closed boundary layer (lid inside every sub-step) keeps its lowest 5% and its 0–40 m layer within ±0.09 of their well-mixed share, stable, neutral and convective (decision 0027; the code before it gave 1.66–1.79 in the lowest 5% and fails it in all three regimes) |
| `test_uniform_column_stays_uniform_across_the_boundary_layer_top` (`test_bl_top_well_mixed.py`) | at the shipped $\Delta t = 60$ s and the default sub-step count, a column started uniform in $[0, 2h]$, open at $h$ to the free-troposphere closure and closed by a lid at $2h$, keeps each 50 m band from $h - 100$ to $h + 100$ m near uniform and the occupancy continuous across $h$, stable, neutral and convective. Default suite (~2.7 min at 8 threads): stable 60,000 particles, 6 h, records every 5 min from 1 h, band and jump tolerance ±0.08; neutral and convective 40,000 particles, 2 h, records from 15 min, ±0.20. Every tolerance is at least 5.2 seed-to-seed standard deviations from the mean (12 seeds stable, 24 neutral and convective), and PR #69's head fails it in all three regimes (stable by the jump, +0.12; neutral −0.59; convective +3.2) (decision 0028) |
| `test_uniform_column_stays_uniform_across_the_boundary_layer_top_full` (`test_bl_top_well_mixed.py`, slow: `GLIDE_SLOW_TESTS=1`) | the same column at full strength (~8.5 min at 8 threads): 150,000 particles, 6 h, records from 1 h, ±0.08 on every band and the jump in all three regimes; each tolerance at least 5.4 standard deviations from the mean over 12 seeds; PR #69's head and main fail it in all three regimes |
| `test_exchange_across_a_stable_top_is_symmetric` (`test_sigma_floor_gradient.py`) | at the shipped $\Delta t = 60$ s and the default sub-step count, in a steady stable column whose σ_w floor layer (0.68 m) is thinner than the drift stencil (h 400 m, u* 0.45 m s⁻¹, H −60 W m⁻²), the ratio of downward to upward exchange between the 40 m above $h$ and the 160 m below it over 2 h is in the band 0.98–1.32 (decision 0030; about 1.15 is expected at 16 sub-steps, first-order sub-step error; this code 1.07–1.21 over 10 seeds; main 3.5–4.4 and a halved stable gradient 0.66–0.69 fail) |
| `test_drift_stencil_stays_on_the_particles_side_of_h`, `test_interface_crossing_rule_backward`, `test_carry_is_one_in_a_steady_column_and_tracks_a_moving_h` (`test_hanna.py`) | the drift never differences the jump at $h$; the transmission probability, velocity and overshoot scaling and reflection at $h$ in backward time; $u'/\sigma$ carried across steps (decision 0028) |
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

**Reciprocity at the defaults.** The forward–backward reciprocity test (§3,
task A6) confirms the backward *sign construction* exactly (pathwise, bit for
bit) and through its companions. Beyond the signs it tests the transfer
symmetry of the scheme both directions share, and at the shipped 16 sub-steps
that does not pass the validation plan's criteria in every case: near the
ground by the scheme's near-surface well-mixed error (about 2%), and near a
stable boundary-layer top by the drift defect at the $\sigma_w$ floor that
decision 0030 (PR #75) addresses. The slow cases are a release check; A6 is
open (§3). A moving-$h$, time-varying case is still needed to test the
backward formulation itself.

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

> **Stale since PR #69 (2026-11-25).** The sub-stepping fix and the
> boundary-layer-top interface (decisions 0027, 0028) change the output by far
> more than any tolerance (reference config, 0–40 m: +48%). The fixtures below
> describe the physics before that change. They will be regenerated once, after
> the residence-counting decision (A2); until then a comparison against them
> reports the change, not a pass.

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

---

## 10. Validation figures and the gallery

The figures of [validation-plan.md §5](validation-plan.md#5-the-figure-set) are
made by scripts in `scripts/validation/figures/`, which implement the design
rules of §6 once:

- `style.py`: one palette and set of colour scales for every tier (the
  project's validated reference palette). GLIDE is blue, a reference model
  orange, an exact or numerical reference solution black and dashed.
- `core.py`: the title stamp (physics tag and a 10-digit hash of the exact
  configuration), log₁₀ ratios on one diverging scale, footprints on a
  logarithmic scale in STILT units, and particle-noise bands. A band is the
  statistic recomputed on 10 random disjoint particle subsets: the standard
  error of the full-sample value is the subset spread over √10, drawn at ±2
  standard errors. Each figure is saved with a JSON file holding the tag, the
  hash, the configuration itself, the git revision and headline metrics.
- `tier0.py`: the Tier 0 set, recomputed from the analytic tests of §3–4 with
  their parameters, seeds and reference solutions (imported from the tests
  where the tests expose them): Taylor dispersion, the well-mixed test in three
  stability regimes with the drift switched off as a companion, the analytic
  plume, the diffusion limit, a convergence preview (time step, sub-step
  target, particle count; the full study is task A5) and terrain. The
  reciprocity figure draws the slow reciprocity cases of §3 and the
  sign-flip companion (task A6; about 15 minutes on 8 threads, so it is
  best made with `scripts/validation/reciprocity/run_cases.py --figure`).
- `gallery.py`: writes `docs/validation/gallery.md`, keyed by physics tag, with
  an index that has one column per tag so a figure can be compared across
  versions, and lists the figures still pending.
- `explorer.py`: a static footprint explorer, one self-contained HTML page per
  run (no server, network or libraries), with GLIDE beside a reference and
  their log ratio for each release. The exportable form of
  `notebooks/footprint_explorer.ipynb` for supplementary material.

Regenerate every figure and the gallery (CPU, about eight minutes on 8 threads, well under
2 GB):

```bash
.venv/bin/python scripts/validation/figures/make_all.py [--root DIR] [--only ID ...]
.venv/bin/python scripts/validation/figures/explorer.py RUN_DIR --reference REF_DIR --out page.html
```

Figures go to `<root>/figures/<physics tag>/tier<N>/`; the default root is
`docs/validation`. Before the freeze tag exists the tag reads
`pre-freeze-<revision>`. A validation task that adds figures adds a builder
to its tier's module and an entry to its `FIGURES` list, so the gallery shows
the gap until the figure exists.
