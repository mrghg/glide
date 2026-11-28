# CONV — a grid- and cadence-independent closure for GLIDE's reduced Emanuel convection

Task CONV, 28 Nov 2026 (approved by Matt the same day). This is an investigation and prototype only, with no PR. Branch `agent/CONV-closure`, based on PR #76 (`agent/CAPE-cloud-top-bound`). Model: claude-opus-5-5.

Result paths are under `outputs/agents/CONV/` in the main checkout. Code paths (`scripts/`, `src/`, `tests/`, `dev/`) are on the branch. A copy of this file is `outputs/agents/CONV/SUMMARY.md`.

Labels: **measured** (file), **derived** (from code or equations), **argued**.

Terms used below:
- **Firing columns**: columns where GLIDE's trigger fires.
- **Jan**: the nine January hours of the CAPE review, on `configs/ab_multisite_perf.yaml` (EUROPE, 150,974 columns).
- **Jul**: eight hours between 25 Jun and 12 Jul 2019, on the EUR store (45,241 columns).
- **E**: the sub-cloud air moved into the cloud layers over the interval. **E/Δt** is the same per unit time, in kg m⁻² s⁻¹.

## 1. Cause

- **The current closure is not intended, and it is not defensible as a transport rate.** So I did not stop.
- **The matrix is a one-step explicit transfer over Δt** (derived, `src/lpdm/convection/emanuel.py`).
  - `m_b` is documented as "already multiplied by the event interval", and `maybe_convect` passes M_b·Δt with Δt = the met interval (3,600 s).
  - FLEXPART's `calcmatrix` does the same with `fmassfrac = delt·fmass`, but delt = lsynctime = 900 s (`conv_mod.f90`, `readoptions_mod.f90`, local FLEXPART at d4e0a86).
- **The relevant Courant number is M_b·Δt / m[i_LCL−1]** (derived).
  - Sub-cloud layer k sheds `M_bΔt·Σ_{i≤k} m_i / m_BL`: its own entrainment plus the subsidence that refills the layers below it.
  - Layer i_LCL sheds the subsidence `Φ[i_LCL−1] = M_bΔt`.
  - So layers i_LCL−1 and i_LCL each shed exactly M_b·Δt, whatever the detrainment profile.
  - The largest outflux fraction is therefore `M_bΔt / min(m[i_LCL−1], m[i_LCL])`, and the thinner layer is the one below cloud base. This is why the review found E = 0.9·m[i_LCL−1] in every column.
  - Physically, compensating subsidence of M_b/ρ ≈ 0.16 m/s travels about 570 m in an hour. A one-step matrix moves subsiding air only one layer per event, and the layers near cloud base are 50–100 m thick.
- **Why M_b is about 10× over the limit** (derived and measured).
  - M_b = 0.03·ρ_LCL·min(√(2 CAPE), 5) = 0.15·ρ_LCL ≈ 0.17 kg m⁻² s⁻¹ in every firing column. So M_b·Δt ≈ 620 kg m⁻² h⁻¹, of the order of the whole sub-cloud air mass.
  - The layer below cloud base holds about 90 kg m⁻² on 40 levels.
  - Measured Courant number, median (p10–p90), Jan: 3.6 (2.2–3.7) on 23 levels, 7.2 (4.7–9.5) on 40, 15 (10–19) on 80. Jul medians: 2.2 / 3.6 / 8.5 (`grid/jan_lcl1.txt`, `grid/jul_lcl1.txt`, legacy).
  - The CFL scale is 0.9/C. The transport per event is therefore 0.9 × the mass of one layer.
- **History** (derived, from the commit history).
  - The cap came with the compensating subsidence in bb5c266 (2 Jul 2026), as a guard to keep the stay probability ≥ 0.
  - Its comment says FLEXPART "enforces the equivalent limit". That is wrong: `calcmatrix` accepts IFLAG = 4 (CFL violated) and does not scale.
  - The closure's docstring claims M_b ≈ 0.01–0.1; the formula gives 0.15·ρ.
- **Detrainment profile.** It does not change the bottleneck above. It decides how high the air goes.
  - The profile is linear in level index and puts most of the air at cloud base.
  - In January GLIDE lifts about 10× less sub-cloud air above 3 km than FLEXPART on the same columns (measured, §4).
- **What a meaningful transport per unit time is** (argued).
  - The cloud-base mass flux M_b is the rate at which sub-cloud air enters the cloud. It is a column property, so it does not depend on grid or cadence.
  - A scheme should move M_b per unit time, less only what is physically re-entrained.
  - exp(Δt·Q) of the rate generator does this: its initial rate is M_b on any grid, and P(3 h) = P(1 h)³.
  - The size of M_b must then come from a closure, not from a numerical cap.

## 2. Candidates

These are config options on the branch. The defaults are unchanged and bit-identical.
- `convection.emanuel.closure`: `cape_velocity` (current) or `buoyancy`.
- `application`: `cfl_scaled` (current), `exponential` or `substeps` (with `n_substeps`).
- `trigger_height_above_lcl_m`: unset (current, the level LCL+1) or a height.

The variants:
- **buoyancy**: FLEXPART's steady-state CBMF.
  - M_b = (0.1·ALPHA/DAMPS)·max(DTMIN, 0) = 0.00833·DTMIN kg m⁻² s⁻¹, with FLEXPART's ALPHA = 0.025 and DAMPS = 0.3.
  - DTMIN uses GLIDE's own parcel: ΔT_v at the LCL (interpolated) + 0.9 K + the mass-weighted mean ΔT_v below cloud.
  - Calibration: FLEXPART's (Forster's ALPHA is 8× below Emanuel's 0.2). Checked against FLEXPART's own CONVECT in §4.
- **exponential**: `generator_exponential`.
  - Uniformised scaling and squaring in float64. The number of squarings is chosen per column and applied with `torch.where`.
  - Static shapes and no host sync: 0 device-to-host copies when profiled on the GH200 (`bench/gpu.log`).
  - Non-negative and row-stochastic by construction. Layer masses stay stationary to 1e-12. Backward = D⁻¹·P_fwdᵀ·D.
  - Matches `torch.linalg.matrix_exp` to ≤1e-11 (`tests/test_convection_closure.py`, 14 tests pass).
- **substeps**: n Euler transfers of M_bΔt/n, each CFL-capped. n = 4 is FLEXPART's 900 s on hourly met.
- **c003**: the current formula with closure_c = 0.003 (M_b ≈ 0.017), applied exponentially. This is option (b) of decision 0031; CAPE stays an on/off switch.

Transport per firing column over 1 h, median E/Δt (measured, `grid/{jan,jul}_lcl1.txt`). The last column is one 3-h event divided by three 1-h events, at 40 levels on the same met:

| variant | Jan, 23 / 40 / 80 levels | Jul, 23 / 40 / 80 levels | 3-h ÷ 3 × 1-h, Jan / Jul |
|---|---|---|---|
| legacy (cape_velocity + cfl_scaled) | 0.045 / 0.022 / 0.011 | 0.067 / 0.040 / 0.017 | 0.43 / 0.43 |
| legacy + substeps(4) | 0.091 / 0.063 / 0.036 | 0.106 / 0.099 / 0.056 | 0.63 / 0.63 |
| nominal M_b + exponential | 0.084 / 0.083 / 0.081 | 0.100 / 0.099 / 0.099 | 1 (exact) |
| c003 + exponential | 0.016 / 0.016 / 0.016 | 0.015 / 0.015 / 0.015 | 1 (exact) |
| **buoyancy + exponential** | **0.015 / 0.016 / 0.018** | **0.015 / 0.018 / 0.020** | **1 (exact)** |
| buoyancy + substeps(4) | 0.015 / 0.017 / 0.018 | 0.015 / 0.018 / 0.021 | 1.04 / 1.03 |
| buoyancy + cfl_scaled | 0.016 / 0.017 / 0.011 | 0.016 / 0.019 / 0.016 | 0.51 / 0.70 |

- Under the buoyancy closure, M_b in firing columns has a median of 0.016–0.020, from DTMIN 2.0–2.7 K.
- M_b = 0 in 0–14% of firing columns, mostly on 23 levels in July. There GLIDE's trigger fires, but the parcel is not buoyant at the LCL.

**The trigger is also grid-dependent, and it dominates domain totals** (measured).
- The ΔT_v ≥ 0.9 K test at the level LCL+1 rejects 61–89% of the columns that pass every other check (82–89% in Jan, 61–76% in Jul).
- How high LCL+1 sits above the LCL depends on the grid. Medians on 23 / 40 / 80 levels: 371 / 135 / 59 m in Jan, 446 / 250 / 92 m in Jul.
- Firing column-hours on 23 / 40 / 80 levels: 11,263 / 7,243 / 3,849 in Jan; 9,020 / 4,706 / 2,598 in Jul.
- With the test at a fixed 200 m above the LCL (interpolated): 8,000 / 9,318 / 9,983 in Jan (+29% at 40 levels); 3,886 / 3,934 / 3,999 in Jul (−16% at 40 levels).

**Domain total Σ (E/Δt)·cos φ on 23 and 80 levels, relative to 40 levels** (measured, `grid/*_{lcl1,trig200}.txt`):

| | Jan, 23 / 80 levels | Jul, 23 / 80 levels |
|---|---|---|
| legacy, current trigger | 3.66 / 0.26 | 3.27 / 0.20 |
| buoyancy + exponential, current trigger | 1.46 / 0.58 | 1.50 / 0.65 |
| legacy, trigger at 200 m | 2.07 / 0.54 | 1.88 / 0.48 |
| **buoyancy + exponential, trigger at 200 m** | **0.92 / 1.06** | **1.03 / 1.00** |
| c003 + exponential, trigger at 200 m | 0.87 / 1.07 | 0.99 / 1.02 |

**Cadence on real, changing met** (measured, `cadence/{jan,jul}.log`, 40 levels).
- Compared: three hourly events, against one 3-h event on the first hour's met, in the columns that fire at t0.
- Legacy gives 0.63 (Jan) and 0.65 (Jul). Buoyancy + exponential gives 1.48 and 1.57.
- Both are biased by the trigger's intermittency: a column that fires at t0 fires in only 2.1 of the 3 hours on average.
- The ratio of the two, 2.34, equals the constant-met factor 1/0.43 (derived). So the scheme's cadence effect is 2.3×; the rest is met sampling, which any cadence has.

**Memory of M_b.**
- FLEXPART relaxes CBMF by a factor 0.7 *per call* (derived). Its own spin-up therefore depends on its cadence.
  - Four 900 s calls reach 0.77 of steady state (measured: 0.0142 / 0.0184 in Jan).
  - One 3,600 s call reaches 0.395 of what four 900 s calls give (measured, `flexpart/compare_*.log`).
- GLIDE could hold M_b per met column as a [C] tensor in `ConvectionState`, with static shape. But GLIDE runs backward, so the previous event is physically later, and carrying it would relax towards the future (argued).
- A causal version needs the DTMIN of the earlier hour, one extra column diagnosis per window. For linear relaxation it only shifts timing (lag ~40 min, e-folding 2.8 calls × 900 s). Episode-integrated transport is unchanged (derived).
- FLEXPART's own memory is kept per OpenMP thread, and is updated only in columns that hold particles (derived, `cbaseflux(ix,jy,ithread)`).
- Recommendation: no memory (argued).

## 3. Recommendation

Adopt **closure = buoyancy, application = exponential**. Decide the trigger level separately; 200 m is proposed.

- **Evidence.**
  - Per-column transport is independent of the grid (±10%) and of the cadence (exactly).
  - With the 200 m trigger, the domain total varies by −8% to +6% across 23 / 40 / 80 levels.
  - Its M_b matches FLEXPART's own CONVECT on the same columns (§4).
  - Run time does not change measurably (§5).
- **Calibration.** FLEXPART's: 0.00833 kg m⁻² s⁻¹ per K of DTMIN.
- **The 200 m.** It lies between the median height of LCL+1 above the LCL on today's 40-level grid in January (135 m) and in July (250 m). It is a continuity choice, not a physical constant (argued).
- **Cheaper alternative.** `substeps(4)` is within 4% of the exponential at 40 levels. But the CFL cap binds again at 80 levels with 3-h intervals (median ratio 0.82, Jan).

## 4. FLEXPART comparison

**How it was run** (measured, `flexpart/compare_{jan,jul}_L40.log`, `flexpart/flexpart_version.txt`).
- FLEXPART 11's own CONVECT and TLIFT were extracted at build time from the local checkout (d4e0a86); they are not copied into GLIDE.
- They ran on the stores' native L137 columns, carrying CBMF between calls, together with calcmatrix and redist.
- FLEXPART's displacement of particles the matrix does not move (compensating subsidence) is not applied.
- A full FLEXPART run was not possible: no flex_extract input.

**Cloud-base mass flux in GLIDE's firing columns**, median (p10–p90), kg m⁻² s⁻¹:

| | FLEXPART steady (900 s) | GLIDE buoyancy | GLIDE legacy, nominal | GLIDE legacy, effective |
|---|---|---|---|---|
| Jan | 0.0184 (0.0151–0.0229) | 0.0179 (0.0145–0.0223) | 0.177 | 0.022 |
| Jul | 0.0197 (0.0103–0.0274) | 0.0188 (0.0086–0.0262) | 0.159 | 0.040 |

- From a cold start FLEXPART gives 0.0142 (Jan) and 0.0152 (Jul).
- Legacy's effective rate matching FLEXPART in January is a coincidence of layer thickness: it is 0.045 on 23 levels and 0.011 on 80.

**Where convection happens.**
- FLEXPART has CBMF > 0 in 55.7% (Jan) and 61.2% (Jul) of all column-hours. GLIDE fires in 0.53% and 1.30%.
- So FLEXPART's domain-total cloud-base flux is about 45× (Jan) and 20× (Jul) GLIDE's under the buoyancy closure.
- This is a difference in the trigger, not the closure, and is not addressed here.

**Vertical redistribution of a sub-cloud tracer over 1 h.** Median fraction of sub-cloud air above 3 km:

| | FLEXPART | GLIDE buoyancy + exponential | GLIDE legacy |
|---|---|---|---|
| Jan | 0.0084 | 0.0009 | 0.0010 |
| Jul | 0.0114 | 0.0114 | 0.0212 |

- Above 6 km in July: FLEXPART 0.0040, GLIDE buoyancy 0.
- FLEXPART detrains higher. Its cloud top is the highest LNB, and its mass-flux profile is ∝ |ΔT_v| + entrainment rather than linear in level index.

**Caveats.**
- FLEXPART's matrix alone saturates at about 0.0066 kg m⁻² s⁻¹. Its updraft draws from one level, NK (the lowest, ~24 kg m⁻²), which FLEXPART's turbulence refills. So CBMF, not the matrix-only budget, is the rate to compare (argued).
- FLEXPART's own matrix would move more than a level holds in 0.03% (Jan) and 0.5% (Jul) of convecting columns at 900 s, but in 40% and 34% at 3,600 s (measured). It is safe only because it calls every 900 s.

## 5. Production effect and cost

**CPU smoke** (`configs/smoke_mhd_single_release.yaml`, seed 42; measured, `footprints/smoke/compare.log`).
- Legacy and buoyancy + exponential give bit-identical footprints and endpoints, so 0–40 m and the total change by 0.
- No particle was ever in a firing column (25 events, 0 occupied).
- Legacy reproduces the smoke sums of PR #76 exactly (1931.7495 / 39624.061 / 34862.278).

**Summer elevated receptor** (measured, `footprints/jfj/compare.log`).
- Setup: Jungfraujoch, 1,000 m AGL, 8 Jul 2019 12 UTC, 3 days back, 21,600 particles. That is `scripts/validation/zs-repair/case2-jfj.yaml` with length 259,200 s on the EUR store.
- Changes are relative to legacy with seed 42.

| run | 0–40 m | 40–1000 m | 1–5 km | total | particle-events displaced by convection |
|---|---|---|---|---|---|
| legacy, seed 42 (reference) | | | | | 4.16% |
| legacy, seed 43 (noise) | −1.8% | −0.6% | +0.3% | +0.07% | 4.19% |
| buoyancy + exponential | −2.7% | −1.4% | +0.5% | +0.06% | 2.25% |
| buoyancy + exponential, trigger 200 m | −5.5% | −0.5% | +0.2% | +0.04% | 1.97% |

- The new closure displaces half as many particles.
- The footprint changes little: 0–40 m by −3% to −5%, against ±1.8% from the seed alone (one seed pair, so a crude noise estimate). The total changes by less than 0.1%.
- Cell-level 0–40 m log-ratios (p5/p95 ≈ −0.5/+0.2) are the same size as the seed noise.

**Cost** (measured).
- Run wall time is unchanged: smoke 6:50 against 6:50, JFJ 15:52 against 15:55 (CPU, 3 threads).
- Per event, building the transition matrices for U occupied firing columns (exponential against cfl_scaled; `bench/{gpu,cpu}.log`):

| case | GPU (GH200) | CPU (2 threads, loaded node) |
|---|---|---|
| U = 256, Z = 40 | not run | 58 ms against 1.4 ms |
| U = 2048, Z = 40 | 3.5 ms against 1.3 ms | 0.51 s against 4.4 ms |
| U = 2048, Z = 137 | 28 ms against 2.2 ms | 5.1 s against 47 ms |

- JFJ had at most 26 occupied firing columns per event, so the cost is negligible there.
- On a CPU run with thousands of occupied columns, `substeps(4)` costs 3× cfl_scaled instead.

## 6. What Matt needs to decide

1. **Closure.**
   - Replace `cape_velocity` (the cap and CFL scale, option (a) of decision 0031) with `buoyancy` (option (c), without memory).
   - Or with c003 (option (b)), which keeps CAPE as an on/off switch and a near-constant M_b.
   - Recommended: buoyancy.
2. **Time integration.** `exponential` (exact; recommended), or `substeps` with a fixed n (cheaper on the CPU, approximate).
3. **Trigger level.** A fixed height above the LCL (200 m proposed), or keep LCL+1. Without it, domain-total transport still varies ×1.5 / ×0.6 between 23 / 40 / 80 levels, even with a rate closure.
4. **Detrainment profile and cloud top** (first vs highest LNB): whether these are in scope before the freeze. In January GLIDE lifts about 10× less sub-cloud air above 3 km than FLEXPART.

Any of these changes model output, so it needs a decision record and new defaults before `v0-physics`.

## 7. Unresolved

- **Trigger form and frequency vs FLEXPART.** FLEXPART convects in 50–100× more columns. Not touched.
- **Detrainment profile and cloud top.** Measured only, not changed.
- **Footprint evidence is thin.** One JFJ case and one seed pair over 3 days. No low-inlet continental summer site and no multi-site run.
- **DTMIN.** GLIDE computes it from its surface parcel on the AGL grid; FLEXPART uses the level of maximum moist static energy (NK) on model levels. The medians agree within 3–5% on GLIDE's firing columns. That does not prove they agree elsewhere.
- **Tests.**
  - The full CPU suite was not run.
  - Run on the branch: `tests/test_convection*.py`, `test_cape_cloud_top.py` and `test_config*.py` (49 passed); `test_phase_timers.py`, `test_main_runtime.py` and `test_golden_harness.py` (94 passed).
  - The defaults are bit-identical, by test and by the smoke run.
- **GPU.** No end-to-end GPU run. The exponential path was only micro-benchmarked and profiled.
- **Massless ocean layer.** In ocean columns GLIDE's 0 m and 10 m levels have the same pressure, so layer 0 has zero mass (measured in passing). It is clamped and harmless for the matrices, but worth noting.
