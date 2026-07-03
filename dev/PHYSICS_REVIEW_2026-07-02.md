# Physics review findings — 2026-07-02 (handoff document)

> **Implementation status (branch `physics-fixes-jul02`):**
> - **Finding 1 (SHF sign)** — DONE (`fix(met): flip ERA5 sensible-heat-flux sign`).
> - **Findings 2+3 (convection non-divergent matrix + backward transpose + M_b
>   share)** — DONE (`fix(convection): non-divergent mass-flux matrix …`).
> - **Findings 4+5+6 (Hanna σ/T_L: FLEXPART v11 coefficients, |f| Coriolis,
>   stable-SL σ_w) + Finding 7 minor items (t_idx clamp, Z_MIN_M)** — DONE.
> - **Follow-ups IMPLEMENTED 2026-07-02 (after v2 validation):** FLEXPART's
>   Lagrangian-timescale floors and the removal of the MO surface-layer override
>   are now the DEFAULTS (`turbulence.flexpart_tl_floors: true`,
>   `surface_layer_override: false`), motivated by the v2 48-h run: GLIDE
>   over-estimated mean enhancements at polluted low-inlet sites (~120 vs
>   <60 ppb) — traced to near-surface K = σ_w²·T_Lw being 3–8× smaller than
>   FLEXPART's below ~15 m on stable nights. Legacy behaviour retained behind the
>   two config flags for A/B.
> - Finding 7's hardcoded convection interval (3600 s) left as-is (documented
>   in-code; all shipped configs are hourly).

Full physics audit of docs (`docs/turbulence.md`, `docs/convection.md`,
`docs/LPDM_physics_spec.md`) against the implementation (`hanna.py`,
`gpu_engine.py`, `emanuel.py`, `footprint_gridder.py`, `comparison.py`,
`met_reader.py`) and the literature. This file is the work order for the fixes;
each finding has location, evidence, consequence, and fix guidance.

**Read "What is verified CORRECT" (bottom) before changing anything** — the
core Langevin/WMC machinery passed the audit and must not be "fixed".

Workflow expectations: CLAUDE.md at repo root. Run tests with
`.venv/bin/python -m pytest -q`. This box is CPU-only; all fixes below are
CPU-testable. After runtime-impacting edits, run at least the targeted tests
for changed modules. The V1 well-mixed tests
(`tests/test_main_runtime.py::test_v1_*`) are the acceptance gate for any
drift/turbulence change.

---

## Finding 1 — ERA5 SHF sign inversion (HIGH, confirmed, fix FIRST)

**Location:** `src/lpdm/met_reader.py` `_convert_shf_to_w_per_m2` (~line 941)
and its call site in `_dataset_to_channel_tensor` (`elif key == "shf"`).
Consumers: `obukhov_length`, `convective_velocity` in
`src/lpdm/turbulence/hanna.py` (~lines 125–160).

**Behaviour:** the reader de-accumulates `surface_sensible_heat_flux` (J/m² →
W/m²) but never flips its sign. ERA5/ARCO uses the ECMWF convention
(**positive = downward**). `obukhov_length` documents and assumes
**positive = upward**.

**Evidence (verified 2026-07-02 against the local store
`~/data/arco-era5/EUROPE_202401.zarr`):** midday Spain (2024-01-15 12 UTC,
39–41N, −6..−2E) mean = **−37,116 J/m²** (true flux upward ⇒ data records it
negative ⇒ downward-positive convention); night (03 UTC) = **+43,217 J/m²**.
NOTE the store's `standard_name` attr says `surface_upward_sensible_heat_flux`
— that CF label is WRONG for the actual data (known ERA5 metadata trap);
trust the empirical check above.

**Consequence:** stability classification is inverted on all real-met runs:
daytime convective BLs classified stable (and `w* = 0` since the `cube > 0`
guard sees H<0), stable nights classified convective. Winter example: H_true =
+10 W/m², u* = 0.3 → code computes L ≈ +230 m → h/L ≈ +2 → "stable" when the
truth is weakly unstable. First-order error in summer.

**Why tests never caught it:** all synthetic fixtures (`tests/test_hanna.py`,
`tests/test_main_runtime.py` `AnalyticMetReader`, `shf_w_m2=200.0` meaning
convective) use the physics convention, so the pipeline is self-consistent on
synthetic met and only wrong on real ERA5.

**Fix:** negate in the reader at the `shf` conversion point (`arr = -arr`
inside `_convert_shf_to_w_per_m2`, or at its call site), with a comment naming
the ECMWF downward-positive convention AND warning about the misleading
`standard_name`. Decide the canonical internal convention = positive-upward
(matches the docs and every consumer). Add a regression test that feeds a
J/m²-unit field through the reader and asserts the sign flip (the fixture
convention note belongs in the test docstring). Check whether
`instantaneous_surface_sensible_heat_flux` (the preferred alternative named in
`DEFAULT_VARIABLE_MAP` comments) has the same convention — it does (also
ECMWF downward-positive), so the flip must apply to both unit branches.

**After fixing:** re-run the 56-site validation (notebooks/multisite_validation.ipynb
against a fresh GH200 run) — expect day/night residual structure vs NAME/FLEXPART
to improve.

---

## Finding 2 — Convection backward kernel not time-reversed + no subsidence (HIGH)

**Location:** `src/lpdm/convection/emanuel.py` `_redistribute_particles`
(~line 732: `move_probs = ma[host] / layer_mass[host]`); doc claim in
`docs/convection.md` §3.7.

**Behaviour:** backward runs apply the *forward* kernel: a particle hosted in
layer j moves with row probabilities `MA[j, :] / m_j`.

**Correct physics:** the time-reversed transition for a backward particle in
layer j is "which layer did this air come from": `p_b(j→i) = MA[i, j] / m_j`
— the **column** of MA, not the row. Row and column agree only under detailed
balance, which this matrix does not satisfy (sub-LCL rows feed the cloud;
nothing feeds back). Additionally, `docs/convection.md` §4 departure 6 claims
compensating subsidence is "implicit in mass conservation" — no mechanism
implements it: non-displaced particles never move down, so a well-mixed column
does NOT stay well-mixed through a convection call in either time direction
(the BL drains upward with no return path). FLEXPART models subsidence as an
explicit downward displacement of non-displaced particles (Forster et al.
2007).

**Consequence:** systematic vertical redistribution bias whenever convection
fires. Currently masked: January Europe rarely triggers (CAPE ≥ 50 J/kg +
Δθv ≥ 0.9 K + depth ≥ 500 m). Will matter for summer validation.

**Fix order:**
1. FIRST write the missing test: a V1-style well-mixed test through
   `maybe_convect` — initialise particles ∝ layer mass in a column with a
   convectively unstable sounding (there is one in `tests/test_convection.py`
   already), call `maybe_convect` repeatedly, assert the mass-weighted profile
   stays flat. Run it for the forward kernel (it should FAIL, demonstrating
   the bug) — then implement:
2. Backward kernel = transposed, mass-weighted matrix (`MA[:, j] / m_j`).
   The scheme needs to know the run direction — it currently doesn't take one;
   GLIDE is backward-only in practice, so the minimal fix is to use the
   transposed kernel unconditionally and document that `maybe_convect`
   implements the backward form.
3. Compensating subsidence for non-displaced particles (FLEXPART approach:
   downward displacement balancing the column's net upward mass flux), OR
   demonstrate via the test that transpose-only is well-mixed-preserving and
   document the residual.
Acceptance: the new well-mixed test passes; `tests/test_convection.py` all green.

---

## Finding 3 — Every sub-LCL row carries the full M_b (MEDIUM-HIGH)

**Location:** `src/lpdm/convection/emanuel.py` `compute_mass_flux_matrix`
sub-LCL block (~lines 430–437): each row i < LCL is normalised so THAT ROW
sums to `m_b`. Doc describes the same math (`docs/convection.md` §3.5).

**Behaviour vs intent:** the closure defines M_b as the total cloud-base mass
flux, but with n_BL sub-LCL layers each contributing M_b, the total BL→cloud
flux is n_BL·M_b. Numbers: M_b·3600 ≈ 540 kg/m² per hour vs ERA5 layer masses
~255 kg/m² ⇒ per-layer move probability hits the 0.99 clamp — essentially the
entire BL vents every convective hour.

**Fix:** share M_b across the BL mass-weighted: row i sums to
`M_b · m_i / Σ_{k<LCL} m_k`. Update the doc §3.5 wording ("the BL acts as a
single source" then becomes true). Update the matrix test in
`tests/test_convection.py` that asserts "BL rows sum to M_b exactly" to assert
the pooled total instead.

---

## Finding 4 — Neutral regime: signed Coriolis breaks SH (MEDIUM, latent)

**Location:** `src/lpdm/turbulence/hanna.py` `_in_bl_neutral` (~lines
182–196); `coriolis_parameter` returns signed `2Ω sin(lat)`.

**Behaviour:** for lat < 0, `exp(-2 f z / u*)` has a positive exponent → σ
GROWS with height (×2.6 at z = 1 km, u* = 0.3); the `1 + 15 f z / u*`
denominator goes < 1 and gets clamped to 1. The Ekman-depth physics scales
with |f|.

**Fix:** use `f = |2Ω sin(lat)|` inside the neutral formulae, with a small
floor so the equator doesn't degenerate. (Verified 2026-07-02: FLEXPART v11
sidesteps the issue entirely by hardcoding f = 1e-4 s⁻¹ — see the Finding 5
transcription; GLIDE's per-latitude |f| with a ~1e-5 floor is strictly better,
just needs the abs().) Add a unit test at lat = −45° asserting σ_w decreases
with z. Harmless for the current EUROPE runs; wrong for any SH/equatorial
domain.

---

## Finding 5 — Hanna coefficients diverge from FLEXPART (MEDIUM — cross-check now DONE, reference below)

**Location:** `src/lpdm/turbulence/hanna.py` `_in_bl_stable` / `_in_bl_neutral`
/ `_in_bl_unstable` (~lines 163–224) AND the same formulas in
`docs/turbulence.md` §3.2.2. Both doc and code carry the warning that the
constants "MUST be cross-checked against the FLEXPART source tree".

**The cross-check was performed 2026-07-02 against FLEXPART v11 master**
(`src/turbulence_mod.f90`, `subroutine hanna`, lines ~296–384 — NOTE: v11 has
no `hanna.f90`; the v10 files were folded into `turbulence_mod.f90`). Re-fetch
with:

```
curl -s "https://gitlab.phaidra.org/api/v4/projects/flexpart%2Fflexpart/repository/files/src%2Fturbulence_mod.f90/raw?ref=master"
```

(Do NOT vendor the file into this repo — FLEXPART is GPL, GLIDE is Apache-2.0.
The transcription below is the reference.)

**FLEXPART v11 `subroutine hanna` — definitive formulas** (ζ = z/h; regime:
NEUTRAL if h/|L| < 1, else UNSTABLE if L < 0, else STABLE — equivalent to
GLIDE's ±1 thresholds on h/L):

- **Neutral** (f hardcoded to 1e-4 s⁻¹; `corr = z/u*`):
  - σ_u = 1e-2 + 2.0·u*·exp(−3e-4·corr)   [Hanna Eq 7.25 — decay 3f]
  - σ_w = 1.3·u*·exp(−2e-4·corr) + 1e-2; **σ_v = σ_w**   [Eq 7.26 — σ_v is 1.3, NOT 2.0]
  - T_Lu = T_Lv = T_Lw = 0.5·z/σ_w/(1 + 1.5e-3·corr)   [Eq 7.27 — ALL EQUAL]
  - analytic dσ_w/dz = −2e-4·σ_w
- **Unstable**:
  - σ_u = σ_v = 1e-2 + u*·(12 − 0.5·h/L)^(1/3)   [Caughey 1982 Eq 4.15] — GLIDE MATCHES
  - σ_w = √(1.2·w*²·(1−0.9ζ)·ζ^(2/3) + (1.8−1.4ζ)·u*²) + 1e-2   [Ryall & Maryon 1998]
  - T_Lu = T_Lv = 0.15·h/σ_u   [Eq 7.17] — GLIDE MATCHES
  - T_Lw piecewise: z<|L| → 0.1·z/(σ_w·(0.55−0.38·|z/L|));
    elif ζ<0.1 → 0.59·z/σ_w;  else → 0.15·h/σ_w·(1−exp(−5ζ))
  - analytic dσ_w/dz = 0.5/σ_w/h·(−1.4u*² + w*²·(0.8·max(ζ,1e-3)^(−1/3) − 1.8·ζ^(2/3)))
- **Stable**:
  - σ_u = 1e-2 + 2.0·u*·(1−ζ)   [Eq 7.20 — LINEAR, not (1−ζ)^¾]
  - **σ_v = σ_w** = 1e-2 + 1.3·u*·(1−ζ)   [Eq 7.19 — σ_v is 1.3, NOT 2.0]
  - T_Lu = 0.15·h/σ_u·√ζ [7.22]; **T_Lv = 0.467·T_Lu** [7.23]; T_Lw = 0.1·h/σ_w·ζ^0.8 [7.24]
- **Floors (all regimes): T_Lu ≥ 10 s, T_Lv ≥ 10 s, T_Lw ≥ 30 s** (GLIDE floors
  T_L at 1 s — a large difference near the surface); σ floors via the +1e-2 m/s
  addend.

**Confirmed GLIDE divergences** (each shifts σ/T_L by 10–40% in its regime —
these set the dispersion rates the FLEXPART/NAME validation measures):

| Quantity | GLIDE (doc+code) | FLEXPART v11 |
| --- | --- | --- |
| Unstable σ_w² | `1.5u*²(1−0.95ζ)^⅔ + 1.6w*²ζ^⅔(1−ζ)²` | `1.2w*²(1−0.9ζ)ζ^⅔ + (1.8−1.4ζ)u*²` |
| Unstable T_Lw | `0.15h/σ_w` (single form) | 3-branch piecewise (above) |
| Neutral σ_u decay | `exp(−2fz/u*)` | `exp(−3fz/u*)` |
| Neutral σ_v | `2.0u*` (= σ_u) | `1.3u*` (= σ_w) |
| Neutral T_L | `T_Lu = T_Lv = T_Lw/1.5` | all three equal |
| Stable σ shape | `(1−ζ)^¾` | linear `(1−ζ)` |
| Stable σ_v | `2.0u*(…)` (= σ_u) | `1.3u*(…)` (= σ_w) |
| Stable T_Lv | `= T_Lu` | `0.467·T_Lu` |
| T_L floors | 1 s (all) | 10 / 10 / 30 s (u/v/w) |

These are NOT WMC violations (the drift is finite-differenced from the same
profile — internally consistent), but they directly bias the FLEXPART/NAME
comparison and plausibly contribute to the 56-site magnitude offsets.

**Fix:** transcribe the FLEXPART v11 formulas above into
`_in_bl_{stable,neutral,unstable}`, raise the T_L floors to 10/10/30 s, and pin
unit tests to hand-computed reference values per regime (doc §5.1 always
planned this). Update `docs/turbulence.md` §3.2.2 to match, citing FLEXPART v11
`turbulence_mod.f90` (not "hanna.f90"). Delete the "MUST cross-check" warnings
once done. Note: GLIDE evaluates σ/T_L per-particle from interpolated met — the
formula swap is contained in those three functions + the unstable T_Lw needing
`L` as an argument.

---

## Finding 6 — Stable surface-layer σ_w grows with height (MEDIUM)

**Location:** `src/lpdm/turbulence/hanna.py` `surface_layer_sigma_w`
(~lines 326–348); doc `docs/turbulence.md` §3.2.4.

**Behaviour:** stable branch is `σ_w = 1.3u*(1+5z/L)`, capped at 6× ⇒ σ_w
*increases* with stability up to 7.8u* — larger than convective values.
`(1+5z/L)` is φ_m (the momentum-gradient MO function), not a σ_w scaling.
Observations and Flesch et al. (1995) App. B have σ_w ≈ 1.3u* constant (or
suppressed) in the stable surface layer. Also creates a σ discontinuity at
the z = 0.1h seam (SL σ_w > in-BL σ_w), which the drift finite-difference
turns into a spurious drift spike.

**Consequence:** nocturnal near-surface mixing overestimated → stable-night
surface footprints biased.

**Strengthened by the 2026-07-02 FLEXPART v11 check:** v11's `subroutine hanna`
has NO surface-layer override at all — the regime formulas run down to the
ground (with the 10/30 s T_L floors). So `docs/turbulence.md` §3.2.4's claim
that the SL treatment "matches FLEXPART mod_par_var_pbl.f90 / hanna.f90 (or
equivalent)" is unfounded (neither file exists in v11; no equivalent override
found). The whole SL override is a GLIDE addition.

**Fix options** (pick one, document it): (a) drop the SL override entirely and
follow FLEXPART v11 (regime formulas + raised T_L floors from Finding 5 —
simplest, most defensible for the FLEXPART comparison); or (b) keep an MO
surface layer but fix the stable branch to `σ_w = 1.3u*` (Flesch et al. 1995
App. B; also check the unstable coefficient — Flesch uses `(1−3z/L)^⅓` vs
GLIDE's `(1−2z/L)^⅓`). Either way update doc §3.2.4 and re-run the V1
well-mixed tests (the drift FD changes at the SL seam).

---

## Finding 7 — Minor items

- **t_idx clamp** (`src/lpdm/main.py` ~line 1349 and ~1417):
  `t_idx.clamp_(max=n_t−1)` lumps residence older than the last time-ago bin
  into that bin. Fine for `n_time_bins=1` (time-integrated); for n_t > 1 it
  silently contradicts the `time_ago_end_hours` metadata. Fix: only clamp when
  `n_t == 1`; otherwise let the gridder's `t_idx < n_t` validity mask drop them.
- **Units abuse:** `T_L_MIN_S` (1.0 s) used as a 1-metre height floor in
  `hanna.py` `_free_trop_fields` (`height.clamp(min=T_L_MIN_S)`, ~line 1299)
  and `_column_turbulence` (`z_for_TL = z_query.clamp(min=T_L_MIN_S)`, ~line
  1186). Numerically harmless; introduce a `Z_MIN_M = 1.0` constant.
- **Hardcoded convection interval:** `convection_call_interval_s = 3600.0` in
  `emanuel.py` `maybe_convect` — should come from the met window / runtime.

---

## Doc corrections (do alongside the code fixes)

- `docs/turbulence.md` §1 bullet "no explicit Thomson 1987 drift term" and §6
  bullet "We use FLEXPART's piecewise-homogeneous treatment, not Thomson 1987
  explicit drift" are STALE — contradicted by §3.2.5 (drift added 2026-05-29).
- `docs/turbulence.md` §2.5 CLI flags (`--turbulence-scheme`,
  `LPDM_TURBULENCE_SCHEME`) no longer exist — scheme selection is YAML config.
- `docs/turbulence.md` §6 "Above-BL constant-K is a placeholder" — superseded
  by the Ri closure (§3.2.3 documents the supersession; §6 wasn't updated).
- `docs/convection.md` §3.7 backward-WMC claim: rewrite per Finding 2 outcome.
- `docs/convection.md` §4 departure 6 ("subsidence implicit"): rewrite per
  Finding 2 outcome.
- `docs/turbulence.md` §3.2.2 constants: update per Finding 5 outcome.

---

## What is verified CORRECT (do not touch)

The 2026-07-02 audit confirmed the following against Thomson (1987), Wilson &
Flesch (1993), Stohl & Thomson (1999), Flesch et al. (1995), Lin et al. (2003):

- **OU kernel** (`gpu_engine._ou_step_kernel`): exact-OU `a·w + drift·dt +
  √(σ²(1−a²))·η`; reduces to `b² = 2σ²/τ_L`; Gaussian forcing; component-wise
  σ/τ pairing. Correct.
- **WMC drift** (hanna `_step_core` / dynamic path): Thomson σ-gradient term
  with per-substep `(1+w'²/σ²)` factor + Stohl-Thomson density term; both
  correctly sign-flipped (together, relaxation unchanged) for backward
  integration per Flesch et al. Verified by
  `test_v1_well_mixed_hanna_backward_path` and
  `test_v1_density_weighted_well_mixed_with_F2`. Correct.
- **Drift-profile consistency** (spec P4): ∂σ_w²/∂z finite-differenced from
  the same `_column_turbulence` profile used for the OU step. Correct.
- **Reflection**: joint `(z, w')` flip (W&F §6) + constant-σ basal layer
  (z_ubl 2 m, W&F §7b). Correct.
- **Position integration**: implicit scheme (position advanced with the
  UPDATED velocity) — the Flesch et al. §3a requirement for backward +
  surface sources. Correct; do not "simplify" to explicit.
- **Substepping/numerics**: per-particle adaptive substeps vs T_Lw, 4σ
  rogue-trajectory clip, T_L floors. Correct.
- **ω→w conversion** (`met_reader._convert_vertical_velocity_to_m_s`):
  `w = −R_d·T/(g·p)·ω`, gated on Pa/s units (verified the ARCO store carries
  `Pa s**-1`). Correct.
- **Convection thermodynamics** (Bolton e_sat/T_LCL, θ_e-conserving bisection
  lift, hypsometric CAPE, virtual temperature): correct.
- **Footprint accumulation** (residence time × weight) and **STILT
  conversion** (`m_air/(h·ρ)`, Lin et al. Eq. 5): correct.
- **Meander scheme**: faithful to Maryon/FLEXPART design as documented.

## Suggested commit order

1. Finding 1 (SHF sign) + regression test — one-line physics fix, unblocks
   trustworthy validation runs.
2. Finding 2 test (well-mixed through convection) — RED; then Findings 2+3
   fixes — GREEN. One PR.
3. Findings 4+6 (+ the small Finding 7 items) — small independent fixes, one PR.
4. Finding 5 (adopt the FLEXPART v11 formulas transcribed above) + doc
   corrections — the cross-check is DONE (2026-07-02); no external source
   needed. Largest validation impact after Finding 1. Consider doing 5 and 6
   together since both touch the σ/T_L profile assembly.

After 1–3 land: fresh GH200 run of the 56-site config, re-run
`notebooks/multisite_validation.ipynb`, compare correlations/means against the
2026-06-30 baseline (FLEXPART 0.67, NAME-UMG 0.74, NAME-UKV 0.85; GLIDE means
13–19% high).
