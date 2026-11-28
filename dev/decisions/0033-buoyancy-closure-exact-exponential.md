# 0033 — Convection: FLEXPART's steady-state buoyancy closure, applied exactly over the met interval

Adopted by Matt on 28 Nov 2026 ("Decision 5"), from task CONV. Labels:
- **measured**: the file is given, under `outputs/agents/CONV/` in the main checkout;
- **derived**: follows from the code or the equations;
- **argued**.

"Jan" is the nine January hours of the CAPE review on `configs/ab_multisite_perf.yaml` (EUROPE, 150,974 columns). "Jul" is eight hours of 25 Jun–12 Jul 2019 on the EUR store (45,241 columns). The independent review of the prototype is in `review/`.

## Context

Before this decision, the reduced Emanuel scheme had two parts that together set its transport ([0007](0007-reduced-emanuel-convection.md), [0031](0031-cape-bounded-at-the-cloud-top.md)):
- a cloud-base mass flux $M_b = 0.03\,\rho_{\mathrm{LCL}}\min(\sqrt{2\,\mathrm{CAPE}}, 5\ \mathrm{m\,s^{-1}})$;
- one explicit transfer of $M_b\,\Delta t$ per met interval (Δt = 3,600 s on ERA5), with the whole matrix scaled down so that no layer sheds more than 90% of its mass.

**Why the transport was set by the grid and the cadence** (derived and measured):
- **The bottleneck is the two layers at cloud base.** Layers $i_{\mathrm{LCL}}-1$ and $i_{\mathrm{LCL}}$ each shed exactly $M_b\Delta t$, whatever the detrainment profile. This flux is the compensating subsidence, which a one-step matrix can carry only one layer per event.
- **So the Courant number that matters is $M_b\Delta t / m_{i_{\mathrm{LCL}}-1}$.**
- **$M_b$ was about ten times too large for that.** The cap binds in every firing column, so $M_b = 0.15\,\rho \approx 0.17$ kg m⁻² s⁻¹, about the whole sub-cloud air mass per hour.
- **Measured Courant numbers** (medians, `grid/{jan,jul}_lcl1.txt`):
  - Jan: 3.6 / 7.2 / 15 on 23 / 40 / 80 levels.
  - Jul: 2.2 / 3.6 / 8.5.
- **The CFL scale therefore bound in about 100% of firing columns**, and an event moved $0.9\,m_{i_{\mathrm{LCL}}-1}$.
- **The transport per firing column halved at each grid refinement** (median, kg m⁻² s⁻¹, on 23 / 40 / 80 levels):
  - Jan: 0.045 / 0.022 / 0.011.
  - Jul: 0.067 / 0.040 / 0.017.
- **A 3-hourly store would have moved 0.43× what an hourly one does** (same meteorology).

**The scale was a non-negativity guard, not a calibration** (derived, bb5c266, July 2026). FLEXPART has no counterpart: `calcmatrix` accepts the CFL flag (`IFLAG = 4`) and is called every 900 s.

## Decision

1. **Closure: FLEXPART's cloud-base mass flux in its steady state** (`closure: buoyancy`, the default):

   $$M_b = k\,\max(\mathrm{DTMIN}, 0),\qquad k = 0.1\,\alpha/\mathrm{DAMPS} = 0.00833\ \mathrm{kg\,m^{-2}\,s^{-1}\,K^{-1}}$$

   - **Provenance** (derived, FLEXPART 11 `conv_mod.f90` `convect`, local checkout d4e0a86):
     - FLEXPART relaxes `CBMF = (1 − DAMPS)·CBMF + 0.1·ALPHA·DTMA` each call.
     - ALPHA = 0.025 is Forster's value; Emanuel's 4.3c has 0.2.
     - DAMPS = 0.3, because FLEXPART fixes DELT0 = DELT/3.
   - **DTMIN** = the parcel's ΔT_v at the LCL (interpolated in pressure) + 0.9 K + its layer-mass-weighted mean ΔT_v below the LCL. It is computed from GLIDE's surface parcel (`cloud_base_buoyancy_excess_columns`).
   - **No memory between intervals.** A backward LPDM needs $M_b$ as a function of the meteorology alone: it visits the hours in reverse order, so carrying FLEXPART's remembered CBMF from the previous event would relax towards the future (argued).
     - For linear relaxation, the memory only shifts onset and decay by FLEXPART's relaxation time (e-folding 2.8 calls, about 40 min at 900 s). The episode-integrated flux is unchanged (derived).
     - FLEXPART's own memory is held per OpenMP thread and updated only in columns that hold particles (derived, `cbaseflux(ix,jy,ithread)`).

2. **Exact application over the interval** (`application: exponential`, the default).
   - **What is applied.** The matrix is the same non-divergent updraft-plus-subsidence construction, now a rate (kg m⁻² s⁻¹) with no CFL scale. The transition is $P = \exp(\Delta t\,Q)$.
   - **The generator.** For the backward direction, $Q_{ij} = \mathrm{fmass}[j,i]/m_i$ off the diagonal, with rows summing to zero. That is the transposed column, weighted by layer mass, as FLEXPART's `redist` uses it with `ldirect=-1`.
   - **Direction and stationarity** (derived, tested):
     - $P$ equals $D^{-1}P_{\mathrm{fwd}}^{\mathsf T}D$, the time reversal of the forward transition.
     - $\mathbf m^{\mathsf T}P = \mathbf m^{\mathsf T}$, so a well-mixed ensemble stays well mixed.
     - For a short interval $P$ reduces to `backward_transition_matrices`. On task T1's three-layer cycle they agree to 10⁻¹¹ at Δt = 10⁻⁶, and the backward direction (towards the source layer) holds.
   - **The algorithm** (`generator_exponential`):
     - uniformised scaling and squaring in float64, Taylor degree 12;
     - per-column squarings, up to 12, applied with `torch.where`;
     - static shapes and no host synchronisation (0 device-to-host copies profiled on the GH200, `bench/gpu.log`);
     - non-negative and row-stochastic by construction.
   - **Accuracy against `scipy.linalg.expm`** (measured, `review/check_expm.log`):
     - ≤ 10⁻¹¹ up to Courant ≈ 1,700 on 23–137 levels;
     - 10⁻⁸ to 2·10⁻³ beyond the squaring cap of 0.5·2¹² = 2,048;
     - the minimum entry is always 0, and rows sum to one within 3·10⁻¹³.
   - **Guard.** Columns above 2,048 are counted in `runtime.convection` of run_metadata.json (`columns_beyond_courant_cap`, `courant_max`) and logged at the end of the run. On the Jan and Jul hours the buoyancy closure gives at most 5.6 on 23–80 levels; the old nominal $M_b$ gave at most 108 (measured, `grid/*.npz`).

3. **The trigger is unchanged.** It stays the level LCL+1, +0.9 K, CAPE ≥ 50 J kg⁻¹ and depth ≥ 500 m. Testing ΔT_v at a fixed height above the LCL (`trigger_height_above_lcl_m`) is an option, off by default; the trigger is a separate, controlled study (Decision 6).

4. **The previous scheme stays selectable** for A/B: `closure: cape_velocity` with `application: cfl_scaled`. It reproduces the earlier output bit for bit. `substeps` (n CFL-capped transfers per interval) is kept for comparison.

## Evidence

**Grid independence of the closure** (measured, `grid/{jan,jul}_lcl1.txt`). Transport per firing column, median kg m⁻² s⁻¹:

| | Jan, 23 / 40 / 80 levels | Jul, 23 / 40 / 80 levels |
|---|---|---|
| this decision | 0.015 / 0.016 / 0.018 | 0.015 / 0.018 / 0.020 |
| before | 0.045 / 0.022 / 0.011 | 0.067 / 0.040 / 0.017 |

**Domain totals are still grid-dependent, through the trigger.** Domain-total transport on 23 / 80 levels relative to 40:

| | Jan | Jul |
|---|---|---|
| this decision | 1.46 / 0.58 | 1.50 / 0.65 |
| before | 3.66 / 0.26 | 3.27 / 0.20 |
| this decision, trigger at 200 m above the LCL | 0.92 / 1.06 | 1.03 / 1.00 |

(The last row is `grid/*_trig200.txt`.) That trigger variant is not adopted here.

**On a synthetic column** (measured, `tests/test_convection_closure.py`). The same sounding on 23 / 40 / 79 levels gives:
- this decision: $M_b$ = 0.0202 / 0.0191 / 0.0179 and 69 / 66 / 62 kg m⁻² moved per hour;
- before: 337 / 200 / 101 kg m⁻².

**Cadence.**
- On the same meteorology, one 3-h interval is exactly three 1-h intervals (derived; tested to 10⁻¹²). Before, a 3-h event moved 0.43× (measured).
- On real, changing meteorology the 3-hourly/hourly ratio is 1.48 (Jan) and 1.57 (Jul) for this decision, against 0.63 and 0.65 before (measured, `cadence/`).
  - Both are biased by the trigger's intermittency: a column that fires at t0 fires in 2.1 of the 3 hours on average.
  - Their ratio of 2.3 is the scheme effect (derived).

**Magnitude against FLEXPART.** FLEXPART 11's own `convect` was extracted from the local checkout and run on the stores' native L137 columns, with CBMF carried between 900 s calls (`flexpart/compare_{jan,jul}_L40.log`, `review/fp_thresholds.log`).
- In GLIDE's firing columns, the median CBMF is:
  - FLEXPART steady state: 0.0184 (Jan), 0.0197 (Jul);
  - this closure: 0.0179, 0.0188;
  - the old nominal: 0.177, 0.159;
  - the old effective, after the CFL scale: 0.022, 0.040.
- Over all columns where both are positive, the correlation is 0.955 / 0.952 and the median ratio is 1.013 / 1.022.

## Side effect: less deep transport in total

The per-column flux is now realistic, but the trigger still fires in about 1% of column-hours against FLEXPART's about 60% (measured):
- firing column-hours, GLIDE: 0.53% (Jan), 1.30% (Jul);
- FLEXPART, CBMF > 0: 55.7% (Jan), 61.2% (Jul).

So the domain total of sub-cloud air lifted above 3 km in an hour falls (measured, `review/fp_baseline_f3.log`; GLIDE columns with LCL < 2.5 km, FLEXPART baseline-corrected):
- **July:** from 35% to about 13% of FLEXPART's.
- **January:** from 0.9% to 0.6%.

The old closure's apparent agreement was an artefact of the CFL scale, not of the physics (argued). The trigger is where the remaining gap lies (`review/fp_trigger_split.log`).

**Footprint effect on CPU**: main 27cd98a against this decision, release seeds 42 and 43 (measured, `pr/after/{smoke,jfj}/compare*.log`).
- **Smoke** (Mace Head, January): bit-identical. No particle was ever in a firing column.
- **Jungfraujoch** (1,000 m AGL, July 2019, 3 days back):
  - 0–40 m changes by −2.1% (seed 42) and −0.1% (seed 43). Total changes by +0.05% and −0.01%.
  - Changing only the seed moves 0–40 m by 3.3%, so the effect is within Monte Carlo noise.
  - Convection displaces about half as many particle-events (2.2% against 4.2%).
  - The largest Courant number met is 1.5, and no column exceeds the cap.

## Deferred deliberately

These differences from FLEXPART's full scheme are known and are not changed here:
- **Detrainment profile.** GLIDE's is linear in level index. In January it lifts about 10× less sub-cloud air above 3 km than FLEXPART in the same columns (measured, medians 0.0009 vs 0.0084); in July the medians agree.
- **Downdrafts.** There is no saturated-downdraft branch.
- **Cloud top.** GLIDE uses the first LNB; FLEXPART uses the highest.
- **Parcel origin.** GLIDE starts from the surface; FLEXPART from NK, the maximum moist static energy below its minimum, with its updraft drawing from that one level.
- **The trigger.** Its form and frequency (Decision 6).

## Rejected alternatives

- **Keep the cap and the CFL scale** (0031 option a): transport set by grid layer mass and cadence.
- **`closure_c = 0.003` with the exponential** (0031 option b): grid-independent, but CAPE stays an on/off switch, and the calibration has no source.
- **FLEXPART's 900 s sub-steps, CFL-capped** (`substeps`): within 4% of the exponential on 40 levels, but the cap binds again on 80 levels with 3-hourly met (median 0.82 of the exact result, Jan).
- **`torch.linalg.matrix_exp`:** it reads the matrix norms back to the host, and it is not non-negative by construction.
- **FLEXPART's memory:** anti-causal in a backward run; see 1.

## Status

In force from the PR "CONV: buoyancy closure with exact exponential application (decision 0033)". It supersedes the closure of [0007](0007-reduced-emanuel-convection.md) and closes the open question of [0031](0031-cape-bounded-at-the-cloud-top.md). See `docs/convection.md` §4, §6 and §7.
