# 0034 — Convection: FLEXPART's cloud-base trigger, without memory, as the default

Adopted by Matt on 28 Nov 2026 ("Decision 6"), from the controlled study on
branch `agent/CONV-trigger-study` (d9f09f0). Labels:
- **measured**: the file is given, under `outputs/agents/CONV/` in the main checkout;
- **derived**: follows from the code or the equations;
- **argued**.

"Jan" and "Jul" are the column samples of [0033](0033-buoyancy-closure-exact-exponential.md):
nine January hours on `configs/ab_multisite_perf.yaml` (EUROPE, 150,974
columns) and eight hours of 25 Jun–12 Jul 2019 on the EUR store (45,241
columns). Figures are on 40 levels unless stated. FLEXPART is FLEXPART 11's
own `convect`, extracted from the local checkout (d4e0a86) and run on the
stores' native L137 columns, as in 0033.

## Context

After [0033](0033-buoyancy-closure-exact-exponential.md) the cloud-base mass
flux per firing column matched FLEXPART's, but GLIDE fired in 0.53% (Jan) and
1.30% (Jul) of column-hours, against FLEXPART's 55.7% and 61.2% (measured,
`trigger/columns/{jan,jul}.txt`). The trigger, not the closure, set the domain
total. GLIDE's trigger differed from FLEXPART's in three ways (derived, FLEXPART
`conv_mod.f90` `convect`, lines 1415–1462 at d4e0a86):

- **Opposite sign and level of the ΔT_v test.** GLIDE fired only if the
  surface parcel was at least 0.9 K **warmer** than the environment at the
  level LCL+1 (`trigger_dtv_k`). FLEXPART skips a column only if its parcel is
  at least DTMAX = 0.9 K **colder** at ICB, the first level above the LCL, and
  only when the column was not already convecting
  (`IF (CBMF.EQ.0.0 .AND. TVP(ICB).LE.(TV(ICB)-DTMAX))`). So FLEXPART convects
  with a parcel up to 0.9 K colder than its environment at cloud base; GLIDE
  required it to be 0.9 K warmer one level higher. The code attributed GLIDE's
  form to Forster et al. (2007) Eq. 34; that could not be checked
  ([0031](0031-cape-bounded-at-the-cloud-top.md)), and it is not FLEXPART's.
- **A CAPE floor** of 50 J kg⁻¹ (`min_cape_j_kg`). FLEXPART has none; its CAPE
  only sets the cloud top.
- **A cloud-depth check** of 500 m (`min_cloud_depth_m`). FLEXPART has none.

GLIDE's test also depended on the grid: the height of the level LCL+1 above the
LCL halves from 40 to 80 levels, and so did the firing count (measured, 0033).

## Decision

1. **The default trigger is FLEXPART's cloud-base test without memory**
   (`convection.emanuel.trigger: flexpart_cloud_base`,
   `flexpart_cloud_base_trigger_columns`). A column convects unless:
   - the parcel is 0.9 K or more colder than the environment at ICB
     (GLIDE's $i_{\mathrm{LCL}}$): $\Delta T_v\vert_{\mathrm{ICB}} \le -0.9$ K;
   - DTMIN ≤ 0, so the buoyancy closure's $M_b$ is zero (FLEXPART convects while
     CBMF > 0). The trigger therefore needs `closure: buoyancy`, and refuses
     `cape_velocity` with a message pointing to `trigger: glide`;
   - the parcel level has T < 250 K or q ≤ 0;
   - the moist static energy has no minimum below the top two levels
     (`IHMIN = NL − 1`);
   - PLCL is outside 200–2000 hPa, there is no LCL, or ICB is one of the top two levels.

   The cloud must have at least one level ($i_{\mathrm{LNB}} > i_{\mathrm{LCL}}$;
   FLEXPART's INB is at least ICB + 1). There is no CAPE floor and no depth check.
2. **No depth check.** The study's first version kept GLIDE's 500 m depth check.
   It was dropped: FLEXPART has none, and with it the firing count changed by
   1.41× / 0.83× (Jan) and 1.50× / 0.77× (Jul) on 23 / 80 levels against 40,
   because GLIDE's first-LNB cloud depth depends on the grid (measured,
   `trigger/columns/{jan,jul}.txt`, "depth >= 500 m" rows).
3. **No memory**, as for the closure (0033): FLEXPART skips the ΔT_v test where
   CBMF > 0 from its previous call. A backward run visits the hours in reverse,
   so that memory would come from the future (argued, 0033).
4. **`trigger: glide` stays selectable** and reproduces the trigger before this
   decision bit for bit (§Evidence). `trigger_dtv_k`, `min_cape_j_kg`,
   `min_cloud_depth_m` and `trigger_height_above_lcl_m` apply only to it. Setting
   one under the default trigger is warned, not refused, and recorded in
   `runtime.physics_warnings` of run_metadata.json, as for the options of 0033.
   The shipped configs no longer set them.
5. **The "Forster 2007 Eq. 34" and "matches FLEXPART" text** in the code and docs
   about GLIDE's old trigger is corrected.

## Evidence

**Firing** (measured, `trigger/columns/{jan,jul}.txt`, 40 levels; share of column-hours):

| | Jan | Jul |
|---|---|---|
| this decision | 49.4% | 60.2% |
| FLEXPART (CBMF > 0) | 55.7% | 61.2% |
| GLIDE before (`glide`) | 0.53% | 1.30% |
| share of this trigger's firing where FLEXPART fires | 98.3% | 94.7% |
| share of FLEXPART's firing this trigger catches | 87.1% | 93.1% |

**Domain-total cloud-base flux**, Σ M_b cos φ: 3,805 against FLEXPART's
steady-state 4,166 (Jan, 91%) and 967 against 1,010 (Jul, 96%) (measured, same
files). The pre-0034 trigger gave 92 and 51.

**Grid independence** (measured, same files, "Grid dependence" rows). On 23 and
80 levels relative to 40:

| | firing | Σ M_b | Σ E/h |
|---|---|---|---|
| this decision, Jan | 0.965 / 1.012 | 1.020 / 0.982 | 1.053 / 0.940 |
| this decision, Jul | 0.979 / 1.003 | 1.024 / 0.979 | 1.058 / 0.934 |
| before, Jan | 1.555 / 0.531 | 1.428 / 0.592 | 1.461 / 0.583 |
| before, Jul | 1.917 / 0.552 | 1.481 / 0.660 | 1.497 / 0.652 |

So the domain totals are grid-independent to ±7% (E/h, the sub-cloud air moved
into the cloud per hour), against factors of 0.53–1.9 before.

**The same masks in this PR's code** (measured, `trigpr/make_fixture.log`). On
the hour 8 Jul 2019 12 UTC (45,241 columns, 40 levels), this branch's default
fires in the same 31,764 columns as the study's `flexpart_cloud_base`, and
`trigger: glide` in the same 1,857 columns as the study's `glide`, which the
study had checked against task CONV's runs (`grid/*_lcl1.npz`) in all 51 level-hours. A
stratified sample of 229 of these columns is a test fixture
(`tests/data/trigger_columns_l40.npz`).

**Footprints** (measured, from the study, not rerun here;
`trigger/footprints/{jfj,kre}/compare.log`; GPU, paired seeds, July 2019,
3 days back, 40 levels):
- **Jungfraujoch** (seeds 42–46): 0–40 m **+3.0%** (range +0.6% to +5.3%, all
  five positive; seed noise ±1–2%); total −0.5%. Below 2 km the footprint rises
  4–7%, and 2–5 km falls 5–7%. Convection displaces 2.8% of particle-events,
  against 1.9%.
- **Křešín**, 50 m (seeds 42–46): 0–40 m **+0.6%** (−0.4% to +2.2%; within the
  seed noise of ±1–2%); total +0.1%. Convection displaces 4.9% of
  particle-events, against 1.9%.
- On 80 levels (seeds 42–44; `{jfj,kre}_L80/compare.log`): Jungfraujoch 0–40 m
  +1.5%, Křešín −0.5%, totals within 0.2%.

**Cost** (measured).
- Column diagnosis on the CPU: 600 ms against 597 ms for 45,241 columns
  (`trigger/bench/columns_cpu.log`); the trigger is a few element-wise tests.
- GPU footprint runs (compiled; `trigger/footprints/*/*.log`): wall times
  1:12–1:25 with either trigger.
- Transition matrices are built only for firing columns that hold particles. In
  the study's GPU runs (seed 42) that is 14× as many on Jungfraujoch (5,093
  against 371) and 24× on Křešín (8,271 against 340). On the CPU, where the
  matrices are not hidden behind other work, a run that meets many convecting
  columns does correspondingly more of this work (argued; see §Side effects).

## Remaining differences from FLEXPART

- **Parcel origin.** GLIDE's parcel is the surface level; FLEXPART's is NK, the
  level of maximum moist static energy below its minimum. The T < 250 K and
  q ≤ 0 tests use GLIDE's surface parcel.
- **No memory** (Decision 3).
- **Cloud top.** GLIDE uses the first LNB, FLEXPART the highest
  (`max(INB, INB1)`), so many of the newly firing columns have one-level clouds
  in GLIDE (61% of firing column-hours have a cloud shallower than 500 m, Jan,
  measured). 
- **Detrainment profile.** GLIDE's is linear in level index; FLEXPART's
  follows |ΔT_v| plus entrainment.
- **Downdrafts.** GLIDE has no saturated-downdraft branch.

The last three set how high the air goes. With this trigger the domain's
sub-cloud air newly above 3 km after an hour is 8.5·10⁻⁵ (Jan) and
4.8·10⁻⁴ (Jul), against FLEXPART's 1.17·10⁻³ and 1.18·10⁻³: **7% and 40%**
of FLEXPART's (measured, `trigger/columns/{jan,jul}.txt`, "lift>3km"). Before
this decision it was 0.6% and 15%. That gap is deferred.

## Side effects

- **Model output changes** wherever particles meet convecting columns: most
  footprints below 2 km gain a few per cent of convective mixing (above). The
  golden fixtures are stale and will be regenerated in the planned single
  regeneration PR.
- **The CPU smoke run now has convection.** Before, no particle in
  `configs/smoke_mhd_single_release.yaml` was ever in a firing column (0033);
  see the PR for the measured change.
- **CAPE no longer gates convection.** It remains a diagnostic (and the
  `cape_velocity` closure's input), and the `cape` statistics in the log are now
  medians over many weakly unstable columns.
- **`dtv_lcl_plus_1`** in `ColumnConvection` holds ΔT_v at ICB under the default
  trigger (at LCL+1 under `glide`).
- **More transition matrices** are built per event (above). The work is bounded
  by the number of particles, not of columns.
- Configs that select `closure: cape_velocity` must now also select
  `trigger: glide`; the message says so.

## Rejected alternatives

- **Keep GLIDE's trigger** and test ΔT_v at a fixed height above the LCL
  (`trigger_height_above_lcl_m`, 0033): fixes the grid dependence but keeps the
  sign, the CAPE floor and the depth check, so the firing stays at about 1%.
- **FLEXPART's trigger with GLIDE's 500 m depth check** (the study's first
  version): fires in 19–22% of column-hours, and depends on the grid through the
  first-LNB depth (Decision 2).
- **FLEXPART's memory:** anti-causal in a backward run (0033).

## Status

In force from the PR "TRIG: FLEXPART cloud-base convection trigger as default
(decision 0034)". It supersedes the trigger of
[0007](0007-reduced-emanuel-convection.md) and item 3 of
[0033](0033-buoyancy-closure-exact-exponential.md), and settles the trigger
difference recorded in [0031](0031-cape-bounded-at-the-cloud-top.md). See
`docs/convection.md` §3, §7 and §8.
