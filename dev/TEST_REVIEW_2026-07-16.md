# Test-suite review & synthetic-physics test plan (2026-07-16)

Work order from the 2026-07-16 test-suite review. Goal: the missing layer of the
suite — **quantitative statistical verification of the dispersion against known
analytic results** ("a physics test suite robust enough to convince me it's
right"). Everything here runs on a CPU box with fixed seeds, no network, no GPU,
no restricted data — by design, since that is the whole value of synthetic tests.

State at review time: 236 tests, 12 files, strong on mechanics/regression
(well-mixed pair with-and-without drift, RK2 order-of-convergence, static/dynamic
path parity, graph-break + recompile CPU guards, conservation asserted almost
everywhere) but with **no test of dispersion magnitude or footprint shape against
theory**, and **no end-to-end particle test of the terrain-following coordinate**
(`AnalyticMetReader` builds `HourlyMetTensors` directly, bypassing the resample).

Definition of done for every test below: deterministic seed; runs in < 30 s
alone; asserts a quantitative tolerance against a stated analytic target; gets a
row in `docs/VALIDATION.md` (tolerance + seed); placed per the layering rules at
the bottom of VALIDATION.md.

---

## T1 — Taylor (1921) dispersion curve through `HannaScheme.step`  [HIGH, ~half day]

**Target.** For an OU velocity process with constant σ_w, T_L:

    σ_z²(t) = 2 σ_w² T_L [ t − T_L (1 − e^(−t/T_L)) ]

exact at all t: ballistic limit σ_z ≈ σ_w·t for t ≪ T_L, diffusive limit
σ_z² ≈ 2Kt (K = σ_w²T_L) for t ≫ T_L. The existing
`test_zero_wind_diffusion_langevin_gaussian_spread` asserts this at ONE time
point, engine-primitive level, ±15%. This test asserts the **full curve through
the production scheme**, so the substep machinery, T_L floors, and the exact-OU
update are all inside the test.

**Setup.** Drive `HannaScheme.step` directly (pattern:
`test_v1_well_mixed_hanna_backward_path`), releasing all particles **well above
the BL** where the scheme's free-troposphere constants (σ = 0.1 m/s, T_L = 100 s)
are homogeneous by construction. Meander OFF. Domain tall enough that no particle
reaches the surface or `alt_max` (release at ~7000 m, run ≤ 30·T_L). n = 50k.

**Assert.** σ_z(t) within ±10% of Taylor at t/T_L ∈ {0.1, 0.3, 1, 3, 10, 30};
same for σ_x, σ_y (horizontal FT constants). Run at dt = 10 s AND dt = 60 s:
position integration is Euler in z (velocity update is exact-OU), so expect an
O(dt/T_L) variance bias at dt = 60 — assert dt = 10 within 5% and pin the dt = 60
ratio, documenting the production-dt bias explicitly rather than hiding it.

**Pitfalls.** Keep `flexpart_tl_floors` irrelevant (T_L = 100 > 30 floor). Do not
let the substep cap bind (max_substeps default is now 6; T_L = 100, dt = 60,
substep_c = 0.5 → k = 2, fine). Parametrize static/dynamic substep paths.

## T2 — Analytic Gaussian-plume footprint  [FLAGSHIP, 1–2 days]

**Target.** Backward release from height z_r in uniform wind U with homogeneous
(σ_v, σ_w, T_L) and perfect ground reflection: the time-integrated STILT surface
footprint equals the reflected Gaussian plume sensitivity

    f(x, y) = 1 / (π σ_y(x) σ_z(x) U) · exp(−y²/2σ_y²) · exp(−z_r²/2σ_z²)

with σ_y(x), σ_z(x) from Taylor's formula at travel time t = x/U. This is the
only proposed test that verifies the **whole chain** — advection + OU +
reflection + gridder + `to_stilt_surface_footprint` — in shape AND absolute
magnitude against a closed form.

**Setup (v1, engine-level).** Compose the loop in-test like the WMC tests do:
`rk2_advect_backward` (uniform U) + OU updates with prescribed constant σ_v, σ_w,
T_L + `reflect_surface` + `FootprintGridder.accumulate` each step, then the STILT
conversion. Prescribing constants keeps the analytic correspondence exact; a
scheme-level variant (Hanna FT constants + a reflecting floor) can follow later.
z_r = 50 m, U = 5 m/s, σ_v = σ_w = 0.5 m/s, T_L = 100 s, dt = 10 s, n ≥ 200k,
run long enough that the plume spans ≥ 20 downwind cells. Work on a local
metres-based grid (or pin lat = 0 so cos-lat ≈ 1).

**Assert.** Beyond the near field (x > 2·U·T_L, where the diffusive form holds):
(a) crosswind-integrated footprint vs analytic within 15% cell-wise RMS;
(b) fitted Gaussian σ_y(x) within 10% of Taylor at 3 downwind distances;
(c) spatial correlation with the analytic field > 0.98;
(d) absolute magnitude of the peak within 15% (this is the units check nothing
else provides).

**Pitfalls.** Exclude x < 2·U·T_L (plume not yet Gaussian-diffusive). Keep
σ_z(x_max) ≪ any imposed lid so the unbounded-plume formula applies — or add the
image terms. Surface bin depth (e.g. 0–40 m) must be ≪ σ_z at the distances
tested, else integrate the analytic form over the bin.

## T3 — Diffusion-limit vs PDE (inhomogeneous K)  [HIGH, ~1 day]

**Target.** For t ≫ T_L the Langevin model must converge to the diffusion
equation ∂c/∂t = ∂/∂z (K(z) ∂c/∂z). **This is the test class that would have
caught the v2-validation near-surface bias** (K = σ_w²T_L was 3–8× low in the
lowest 15 m): well-mixed tests cannot see a wrong K (uniform stays uniform under
ANY K); only a quantitative profile-evolution test can.

**Setup.** 1-D column, K(z) = κ·u*·z capped at K_max (log-layer-like; e.g.
u* = 0.3, cap at z = 200 m), T_L = 5 s, σ_w²(z) = K(z)/T_L, dt = 1 s (must
resolve T_L), WMC drift term ON (σ_w² varies), reflection at z = 0 and z = H.
Initial condition: narrow slab at z₀ = 100 m. Evolve 1–2 h, n = 100k. Reference:
Crank–Nicolson solve of the PDE on a fine grid — scipy is a core dependency, use
`scipy.linalg.solve_banded` (~40 lines in-test, no new deps).

**Assert.** Binned particle density vs PDE solution: L1 error < 8% at 2–3
checkpoint times; near-surface (lowest 3 bins) occupancy within 10% — the
near-surface number is the point of the test.

## T4 — Terrain hill (Finding 7 synthetic acceptance)  [HIGH, ~1 day — do before the GH200 terrain run]

**Target.** End-to-end over orography: particles hold AGL crossing a hill, and
the surface footprint is continuous over it (no Finding-7 hole). Field-level
tests (`test_vertical_grid`, the two reader resample tests) cannot see the
particle behaviour; nothing e2e exercises the resample today.

**Setup.** Synthetic pressure-level store via the `test_met_reader` mock-builder
pattern, with a Gaussian hill h(x) (peak ~800 m, half-width a few cells). Uniform
wind U; set the stored vertical velocity to terrain-following flow,
w(x, z) = U·∂h/∂x·decay(z) (provide w in m/s units — avoids the omega
conversion; the mock already supports m/s). Geopotential per level = terrain +
fixed AGL offsets, as the existing mock does. Run through the REAL
`ArcoEra5ZarrReader` path with `terrain_following=True` (in-memory subclass) into
either `_run` or an engine-level advect loop using the reader's window.

**Assert.** (a) A particle released at 50 m AGL upwind crosses the hill staying
within ±25 m AGL (the reader's slope correction cancels the imposed w near the
surface); (b) the surface (0–40 m) footprint over the hill cells is nonzero and
within a factor ~2 of the flat-terrain cells at equal downwind distance; (c) a
release ON the hill initialises at terrain + alt_agl (probe the first-step
sampled fields or endpoint distribution). Also run the same scenario with
`terrain_following=False` and assert (b) FAILS — proving the test has teeth.

## T5a — OU autocorrelation & stationarity through the substep loop  [QUICK WIN, ~2 h]

**Target.** R(τ) = e^(−τ/T_L) and Var(w') = σ_w², stationary, through the
production substep machinery. Catches integration bias from `substep_c` /
`max_substeps` / T_L-floor interactions — knobs retuned twice recently
(20→6 substeps; floors flipped on) with no statistical gate.

**Setup/assert.** Homogeneous σ_w, T_L (FT constants again); evolve ≥ 200·T_L;
autocorrelation at τ/T_L ∈ {0.5, 1, 2} within ±5% of exp(−τ/T_L); stationary
variance within ±3%. Parametrize static/dynamic paths × floors on/off.

## T5b — Solid-body-rotation advection  [QUICK WIN, ~2 h]

**Target.** v = ω × (x − c): the trajectory is a circle, returning to the start
after T = 2π/ω, with RK2 global error O(dt²). Upgrades the constant/linear wind
tests to a spatially varying field with curvature; also exercises the backward
sign convention (backward integration rotates the opposite way).

**Setup/assert.** Engine-level `wind_fn` like the existing physics tests, on a
local metres grid; ω = 2π/3600 s⁻¹, one full period; assert return-to-start
within tolerance and error-ratio > 3.5 per dt-halving (reuse the
`test_rk2_advection_second_order_in_dt` pattern).

## T6 — Forward/backward reciprocity  [DEFERRED — outline only]

Thomson reciprocity: forward concentration at the receptor from a surface source
equals (backward footprint × source). The engine primitives all take `backward=`
flags, so this is implementable in the homogeneous T2 setup, and it is the most
fundamental check of the backward formulation. Most work of the six; do after
T1–T5 are green.

---

## Housekeeping (from the same review)

1. **`docs/VALIDATION.md` refreshed 2026-07-16** (stale claims fixed, placeholder
   rows added for T1–T6). Keep it current: every landed test updates its row.
2. **Legacy-flag branches untested**: one smoke each for
   `surface_layer_override=True` and `flexpart_tl_floors=False` (kept for A/B),
   or delete the flags.
3. **`wind_mean` cache** (perf #2, 2026-07-16): add a small unit test — cached vs
   uncached identical, cache invalidates on window change. (Verified ad hoc in
   session; not yet in the suite.)
4. **Emanuel January guard**: a runtime test asserting Emanuel does NOT fire on a
   stable winter bbox-mean column (the "check the v2 log for 'Emanuel fires'"
   concern, as a test instead of a log grep).
5. **Memory-guard trip path**: no test exercises `guard_max_*` →
   `MemoryError` + diagnostics in `run_metadata.json`.
6. **Run `pytest --cov`** once and record the module-level coverage map here —
   this review was by reading, not instrumentation.

## Order of attack

1. T5a + T5b (same day, quick wins, immediately strengthen the suite)
2. T1 (the curve, production scheme)
3. T4 (before the GH200 terrain acceptance run — it is the CPU-side gate for PR #7's physics)
4. T2 (flagship; budget uninterrupted time)
5. T3 (targets the known near-surface risk class)
6. Housekeeping items 2–5 as filler between the above; T6 last, if at all.
