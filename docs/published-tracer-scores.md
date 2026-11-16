# Published tracer-experiment scores

_DATEM ranking scores that other models have published for CAPTEX and ANATEX,
and the ATMES-II statistics published for ETEX-1, with the meteorology
behind each, so that GLIDE's Tier 3 results can be set against them.
Compiled 2026-11-04 from Hegarty et al. (2013), Selvaratnam et al. (2023) and
Mosca et al. (1998); add new studies as they are found. The same numbers are
in [published-tracer-scores.csv](published-tracer-scores.csv) (ranking
scores) and [published-etex-atmes2.csv](published-etex-atmes2.csv) (ETEX-1),
one row per published value, which the tracer benchmark reads (queue task
B12)._

How GLIDE is scored against these numbers is in
[validation-plan.md](validation-plan.md) Tier 3. The data, file formats and
scoring code are described in
[../data/tracer-experiments.md](../data/tracer-experiments.md).

**Contents**

1. [Read this first](#1-read-this-first)
2. [How these scores were computed](#2-how-these-scores-were-computed)
3. [The meteorology behind each score](#3-the-meteorology-behind-each-score)
4. [Scores by experiment](#4-scores-by-experiment)
5. [Backward runs](#5-backward-runs)
6. [ETEX-1: the ATMES-II statistics](#6-etex-1-the-atmes-ii-statistics)
7. [Scores for whole experiments](#7-scores-for-whole-experiments)
8. [Gaps](#8-gaps)
9. [Where the numbers came from](#9-where-the-numbers-came-from)

---

## 1. Read this first

- **Most of these runs used meteorology that is finer than ERA5.** Hegarty et
  al. drove every model with WRF on a 10 km grid with hourly output, and their
  best configuration also nudged the boundary-layer winds towards NARR. ERA5
  is about 31 km. **GLIDE does not need to beat the WRF runs.** The aim is a
  score in the same range as established models, computed the same way.
- **The closest comparisons for GLIDE on ERA5 are the reanalysis-driven
  runs:** NAME on ERA-Interim (about 79 km, 3-hourly), and HYSPLIT and STILT on
  NARR (about 32 km, 3-hourly). ERA5 is about as fine as NARR horizontally,
  much finer than ERA-Interim, and hourly rather than 3-hourly. Note that NAME
  on the coarse ERA-Interim scored a higher mean (2.40) than any NAME run on
  WRF, so finer meteorology does not guarantee a better score.
- **The meteorology matters as much as the model.** For a single model, the
  mean rank moves by 0.22–0.29 between meteorology configurations; with the
  meteorology fixed, the four models' means differ by 0.09–0.22.
  Differences of a tenth or two from a published mean are not meaningful on
  their own.
- **ANATEX-STC and CAPTEX-3 are hard for every model.** One of the two has
  the lowest rank in every configuration. For ANATEX-STC, the correlation
  term is close to zero for NAME and, Selvaratnam et al. note, for the other
  models too.
- **The scores are per release, not per experiment.** They are not comparable
  with scores computed over a whole experiment, such as DATEM's HYSPLIT
  examples (§7).

## 2. How these scores were computed

Both papers use the same protocol; Selvaratnam et al. follow Hegarty et al.
explicitly and use DATEM's own statistics program.

| Setting | Value | Source |
| --- | --- | --- |
| Rank | R² + (1 − \|FB/2\|) + FMS/100 + (1 − KS/100), range 0–4 | both |
| Correlation | R² from DATEM's resampling option (`statmain -b`), which Hegarty et al. call "the bootstrap technique" | Hegarty §5 |
| Analysis | Every measured–modelled pair separately ("global"); no averaging in time or space | inferred from the pair counts |
| Zero threshold | 0 pg m⁻³, for the figure of merit in space and for the measurements | both |
| Negative measurements | Set to zero | Selvaratnam §4 |
| Zero–zero pairs | Included | inferred: excluding them would leave far fewer pairs than reported |
| CAPTEX experiments | Releases 1, 2, 3, 4, 5 and 7, each scored separately; release 6 (30 minutes) left out | both |
| ANATEX experiments | The first 10 releases only (5–16 January 1987), scored as two experiments: GGW (PTCH, DATEM tracer 1) and STC (PDCH, tracer 2); PMCH (tracer 3) left out | both |
| Pairs per experiment | 354, 365, 346, 299, 306 and 183 (CAPTEX-1 to 7), 562 (GGW) and 556 (STC) | Hegarty §6 |
| Modelled concentration | Mean over the lowest 100 m above ground, and over each sampling period (6 h for CAPTEX, 24 h for ANATEX) | both |
| Model runs | Forward, from the source, except Hegarty's STILT backward runs (§5) | both |
| Particles | HYSPLIT, STILT and NAME: 50,000 per CAPTEX release, 25,000 per ANATEX release; FLEXPART: 100,000 per release | both |
| Output grid | 0.25° latitude–longitude (FLEXPART: 25 km on the WRF projection) | both |

**What is not fully known.** The DATEM files do not reproduce Hegarty's pair
counts exactly. Counting DATEM's 6-hour CAPTEX samples, assigned to the most
recent release, gives 347, 352, 356, 281, 318 and 177 against Hegarty's 354,
365, 346, 299, 306 and 183. That is close, but not the same, so the exact
sample selection (3-hour samples, release windows, sites) is not recoverable
from the papers. Queue task B10 settles the selection rule from the data,
before any GLIDE run.

## 3. The meteorology behind each score

| Label | Meteorology | Horizontal | Vertical | Output interval | Used by |
| --- | --- | --- | --- | --- | --- |
| NARR | NCEP North American Regional Reanalysis | about 32 km | 45 levels | 3 h | HYSPLIT, STILT (Hegarty) |
| ERA-Interim | ECMWF global reanalysis (2006 IFS) | about 79 km | 60 levels | 3 h | NAME (Selvaratnam) |
| WRF avrg pbl1 | WRF-ARW 3.2.1, initial and boundary conditions from NARR; time-averaged, mass-coupled winds and convective mass fluxes; boundary-layer winds nudged to NARR | 10 km inner nest, 30 km outer | 43 levels, lowest about 33 m | 1 h | all four models |
| WRF avrg pbl0 | As above, boundary-layer winds not nudged | | | | all four |
| WRF inst pbl1 | As above, instantaneous winds, nudged | | | | all four |
| WRF inst pbl0 | As above, instantaneous winds, not nudged | | | | all four |
| **ERA5 (GLIDE)** | ECMWF global reanalysis (2016 IFS) | about 31 km (0.25°) | 137 model levels | 1 h | GLIDE |

All WRF runs were restarted daily at 0000 UTC and run for 30 hours, the
first 6 hours being replaced by the end of the previous day's run.
Temperature, moisture and winds were nudged to NARR every 3 hours in both
domains, except that winds inside the boundary layer were nudged only in the
"pbl1" runs. NAME's ANATEX runs used only the 10 km domain. The same WRF
fields were used by all four models; Hegarty et al.'s acknowledgments give
NOAA ARL as the source of the fields and of the models' outputs. NAME was
version 7.2.

## 4. Scores by experiment

Forward runs. Hegarty et al. Tables 1–3 (HYSPLIT, STILT, FLEXPART) and
Selvaratnam et al. Fig. 1 (NAME). C1–C7 are CAPTEX releases; A-GGW and A-STC
are the ANATEX experiments. "Mean" is over the eight experiments, as in the
papers' "Avg" rows.

| Model | Meteorology | C1 | C2 | C3 | C4 | C5 | C7 | A-GGW | A-STC | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HYSPLIT | NARR | 2.04 | 2.53 | 1.74 | 1.99 | 2.30 | 2.62 | 2.01 | 1.80 | 2.13 |
| STILT | NARR | 2.24 | 2.59 | 1.72 | 1.98 | 1.93 | 2.58 | 2.14 | 1.77 | 2.12 |
| NAME | ERA-Interim | 2.58 | 2.83 | 2.08 | 2.56 | 2.67 | 2.88 | 1.93 | 1.69 | 2.40 |
| HYSPLIT | WRF avrg pbl1 | 2.40 | 3.47 | 1.83 | 2.23 | 2.66 | 2.94 | 2.43 | 1.40 | 2.42 |
| STILT | WRF avrg pbl1 | 2.43 | 3.27 | 1.86 | 2.33 | 2.73 | 2.61 | 2.34 | 1.57 | 2.39 |
| FLEXPART | WRF avrg pbl1 | 2.42 | 3.17 | 1.91 | 1.95 | 2.54 | 3.14 | 2.24 | 1.30 | 2.33 |
| NAME | WRF avrg pbl1 | 2.49 | 3.02 | 2.09 | 2.28 | 2.75 | 2.41 | 2.10 | 1.44 | 2.32 |
| HYSPLIT | WRF avrg pbl0 | 2.43 | 2.92 | 1.96 | 2.14 | 2.37 | 2.75 | 2.36 | 1.76 | 2.34 |
| STILT | WRF avrg pbl0 | 2.48 | 2.77 | 1.81 | 2.04 | 2.43 | 2.52 | 2.13 | 1.90 | 2.26 |
| FLEXPART | WRF avrg pbl0 | 2.12 | 2.47 | 1.80 | 1.93 | 2.09 | 2.85 | 2.07 | 1.66 | 2.12 |
| NAME | WRF avrg pbl0 | 2.29 | 2.57 | 1.74 | 2.18 | 2.64 | 2.38 | 1.74 | 1.56 | 2.14 |
| HYSPLIT | WRF inst pbl1 | 2.48 | 3.25 | 1.89 | 2.29 | 2.56 | 2.40 | 2.06 | 1.40 | 2.29 |
| STILT | WRF inst pbl1 | 2.42 | 2.99 | 1.80 | 2.19 | 2.54 | 2.29 | 2.35 | 1.54 | 2.27 |
| FLEXPART | WRF inst pbl1 | 2.44 | 3.13 | 1.90 | 1.95 | 2.41 | 3.17 | 2.23 | 1.31 | 2.32 |
| NAME | WRF inst pbl1 | 2.50 | 3.02 | 2.11 | 2.17 | 2.72 | 2.43 | 2.35 | 1.55 | 2.36 |
| HYSPLIT | WRF inst pbl0 | 2.62 | 2.96 | 1.86 | 2.23 | 2.31 | 2.48 | 2.13 | 1.70 | 2.29 |
| STILT | WRF inst pbl0 | 2.67 | 2.84 | 1.76 | 2.07 | 2.31 | 2.49 | 2.27 | 1.76 | 2.27 |
| FLEXPART | WRF inst pbl0 | 2.07 | 2.50 | 1.78 | 1.95 | 2.10 | 2.82 | 2.05 | 1.64 | 2.11 |
| NAME | WRF inst pbl0 | 2.36 | 2.62 | 1.80 | 2.15 | 2.52 | 2.43 | 2.08 | 1.72 | 2.21 |

**Ranges.** Reanalysis-driven: the three runs on NARR or ERA-Interim.
WRF-driven: the sixteen runs on WRF.

| Experiment | Reanalysis-driven | WRF-driven | All 19 runs |
| --- | --- | --- | --- |
| CAPTEX-1 | 2.04–2.58 | 2.07–2.67 | 2.04–2.67 |
| CAPTEX-2 | 2.53–2.83 | 2.47–3.47 | 2.47–3.47 |
| CAPTEX-3 | 1.72–2.08 | 1.74–2.11 | 1.72–2.11 |
| CAPTEX-4 | 1.98–2.56 | 1.93–2.33 | 1.93–2.56 |
| CAPTEX-5 | 1.93–2.67 | 2.09–2.75 | 1.93–2.75 |
| CAPTEX-7 | 2.58–2.88 | 2.29–3.17 | 2.29–3.17 |
| ANATEX-GGW | 1.93–2.14 | 1.74–2.43 | 1.74–2.43 |
| ANATEX-STC | 1.69–1.80 | 1.30–1.90 | 1.30–1.90 |
| **Mean of the eight** | **2.12–2.40** | **2.11–2.42** | **2.11–2.42** |

The lowest mean is 2.11 (FLEXPART, WRF inst pbl0) and the highest 2.42
(HYSPLIT, WRF avrg pbl1).

**The four terms for NAME on ERA-Interim**, the closest analogue to GLIDE on
ERA5. Each term lies between 0 and 1; the four sum to the rank. The terms for
every NAME run are in the CSV. Hegarty et al. show the terms only as a figure
(their Fig. 2), which has not been extracted.

| Experiment | Rank | R² term | FB term | FMS term | KS term |
| --- | ---: | ---: | ---: | ---: | ---: |
| CAPTEX-1 | 2.58 | 0.24 | 0.83 | 0.56 | 0.95 |
| CAPTEX-2 | 2.83 | 0.21 | 0.99 | 0.70 | 0.92 |
| CAPTEX-3 | 2.08 | 0.50 | 0.31 | 0.35 | 0.91 |
| CAPTEX-4 | 2.56 | 0.24 | 0.95 | 0.43 | 0.94 |
| CAPTEX-5 | 2.67 | 0.56 | 0.94 | 0.31 | 0.86 |
| CAPTEX-7 | 2.88 | 0.61 | 0.79 | 0.52 | 0.96 |
| ANATEX-GGW | 1.93 | 0.12 | 0.65 | 0.34 | 0.82 |
| ANATEX-STC | 1.69 | 0.01 | 0.41 | 0.42 | 0.85 |

## 5. Backward runs

Hegarty et al. (Table 4) ran STILT backward from the samplers, the same
approach GLIDE uses: 500 particles per receptor, hourly receptors averaged
over each sampling period, and footprints multiplied by the known emissions
on a 1° grid around the source. All runs used WRF time-averaged winds. The
forward ranks for the same configuration are given after the slash.

| Experiment | Receptors at 50 m, pbl1 | 50 m, pbl0 | 10 m, pbl1 | 10 m, pbl0 |
| --- | ---: | ---: | ---: | ---: |
| CAPTEX-1 | 2.58 / 2.43 | 2.58 / 2.48 | 2.61 | 2.58 |
| CAPTEX-2 | 3.26 / 3.27 | 2.72 / 2.77 | 2.98 | 2.70 |
| CAPTEX-3 | 1.66 / 1.86 | 1.63 / 1.81 | 1.63 | 1.64 |
| CAPTEX-4 | 2.64 / 2.33 | 2.34 / 2.04 | 2.66 | 2.35 |
| CAPTEX-5 | 2.87 / 2.73 | 2.70 / 2.43 | 2.89 | 2.68 |
| CAPTEX-7 | 2.89 / 2.61 | 2.64 / 2.52 | 2.89 | 2.68 |
| CAPTEX mean | 2.65 / 2.54 | 2.44 / 2.34 | 2.61 | 2.44 |
| ANATEX-GGW | 2.23 / 2.34 | — | — | — |

Backward and forward agree to about 0.1 on average, which is the size of
difference to expect from the backward formulation alone. Receptors at 10 m
score about 0.04 lower than at 50 m.

## 6. ETEX-1: the ATMES-II statistics

ATMES-II (1996) had the participating groups rerun ETEX-1 with their own
dispersion models, all driven by ECMWF analyses: the same lineage of
meteorology as ERA5, which makes it the closest published comparison for
GLIDE. The ECMWF data's resolution and interval are not stated in the
sources read here. **Only four models' results are published in the journal
paper** (Mosca et al., 1998, Atmos. Environ. 32, 4307–4324,
doi:10.1016/S1352-2310(98)00179-4). They are identified only by number,
and were "selected to cover a wide range of performances": by the paper's
total ranking they came 6th, 7th, 33rd and 35th of 35. The statistics for
all models are in the JRC report EUR 17756 EN (Mosca et al., 1998, "ATMES-II —
Evaluation of long-range dispersion models using first ETEX release data"),
which is not online; the JRC's ETEX site offers a printed copy of the final
report by email. The participants included the UK Met Office and NOAA, but
which numbers they are is not given, except that the DATEM description
(§5.3) identifies **model 120 as HYSPLIT**, quoting the report's page 203.
All the values below are in
[published-etex-atmes2.csv](published-etex-atmes2.csv).

**The ATMES-II method differs from DATEM's,** so these statistics are
computed separately, not as a ranking score:
- 3-hour samples, paired in time and space; pairs with a lost measurement
  or a missing prediction excluded.
- **A filter on measured zeros**, applied to every index except the figure
  of merit in space: a pair is kept if the measurement is non-zero, or if it
  is zero but falls no earlier than two intervals (6 h) before the cloud's
  arrival at that station and no later than two intervals after its
  departure. This leaves **1,423 pairs**, 34% with a measured zero.
  Checked 2026-11-04 on the JRC file `pmch.dat` (version 1.1): taking the
  cloud's arrival and departure as each station's first and last non-zero
  sample, using all 30 intervals (90 h), and treating `-.88` ("detected, not
  quantifiable") as lost, gives 1,415 pairs, 34% zeros. Treating `-.88` as
  non-zero gives 1,892, and stopping at 72 h gives 1,209, so the rule above
  is almost certainly the one used; the last 8 pairs may be a data-version
  difference.
- Indices with logarithms set zeros to 0.01 ng m⁻³, the instrument's limit.
  The correlation (PCC) is computed on the logarithms of the concentrations.
- The Kolmogorov–Smirnov values (72–365) are not percentages; the paper says
  only that they are positive integers. Check the definition against the
  report before comparing.
- The figure of merit in space is computed from **interpolated contours**
  (logarithmic interpolation; an area-conserving projection is recommended)
  at 0.1 ng m⁻³ at fixed times, and for the time-integrated concentration
  ("dose") at 1 ng m⁻³ h. It depends on the interpolation, so it is not the
  station-count FMS of the DATEM score.
- Models are ranked on each index, and the ranks summed.

**Global statistics** (Mosca et al., Tables 3 and 5). 95% confidence
intervals in parentheses; the model's rank among the 35 in brackets. Bias in
ng m⁻³; FA2, FA5, FOEX and dose FMS in percent.

| Statistic | Model 101 | Model 127 | Model 122 | Model 117 | Best possible |
| --- | --- | --- | --- | --- | --- |
| Bias | 0.023 (−0.013 to 0.059) [7] | 0.795 (0.008 to 1.581) [33] | 0.789 (0.495 to 1.083) [32] | −0.056 (−0.104 to −0.009) [11] | 0 |
| Geometric mean bias MG | 1.258 [14] | 1.041 [3] | 0.843 [13] | 0.315 [33] | 1 |
| Geometric variance VG | 6.234 [2] | 6.19 [1] | 67.007 [31] | 111.372 [34] | 1 |
| NMSE | 9.965 [8] | 1091.147 [34] | 157.767 [32] | 25.938 [21] | 0 |
| PCC (of logarithms) | 0.679 [5] | 0.704 [2] | 0.421 [28] | 0.216 [33] | 1 |
| FA2 | 28 [14] | 37 [6] | 31 [12] | 33 [10] | 100 |
| FA5 | 48 [6] | 53 [1] | 41 [12] | 35 [15] | 100 |
| FOEX | 12 [12] | 1 [2] | −12 [12] | −41 [24] | 0 |
| KS | 138 [19] | 72 [9] | 86 [13] | 365 [32] | 0 |
| Dose FMS, 1 ng m⁻³ h | 73 [5] | 76 [2] | 56 [20] | 37 [22] | 100 |
| **Total ranking** | **6** | **7** | **33** | **35** | |

**Figure of merit in space at 0.1 ng m⁻³** (Table 6), hours after the start
of the release. Every model does best at about 36 h.

| Model | 12 h | 24 h | 36 h | 48 h | 60 h | 72 h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 34 | 48 | 71 | 55 | 46 | 34 |
| 127 | 24 | 55 | 66 | 55 | 56 | 29 |
| 122 | 28 | 35 | 49 | 39 | 37 | 0 |
| 117 | 8 | 4 | 1 | 2 | 8 | 0 |

**HYSPLIT, ATMES-II model 120** (from the DATEM description §5.3, which
quotes the report): correlation 0.44, NMSE 23.6, bias 0.180 ng m⁻³, FMS 61,
FA2 35, FA5 43. The same table gives HYSPLIT's later DATEM-scored values
(correlation 0.48, NMSE 23.25, bias 0.038 ng m⁻³, FMS 57.27, FA2 15.95,
FA5 34.64), which differ partly because DATEM applies no filter.

**What this means for GLIDE.** Four models, two near the top and two near
the bottom, show the spread of the exercise but not its full range or
median. A single statistic is a poor guide: model 127 ranks 1st–3rd on six
indices and 33rd–34th on bias and NMSE, because a few very high predictions
dominate those two. Mosca et al. recommend the logarithmic indices (MG, VG,
PCC) and FA5 for this reason.

## 7. Scores for whole experiments

These score every release and every sample of an experiment together, so they
are **not** comparable with §4. They are useful as a check that the scoring
code works, and for experiments with no per-release studies.

| Model | Meteorology | Experiment | All pairs | Time-averaged | Source |
| --- | --- | --- | ---: | ---: | --- |
| HYSPLIT | NCEP/NCAR reanalysis | CAPTEX | 1.36 | 3.35 | DATEM description (Draxler et al., 2001), Appendix D |
| HYSPLIT | NCEP/NCAR reanalysis | ANATEX tracer 1 | 1.69 | 2.69 | as above |
| HYSPLIT | NCEP/NCAR reanalysis | ANATEX tracer 2 | 1.74 | 2.92 | as above |
| HYSPLIT | NCEP/NCAR reanalysis | ACURATE | 1.46 | 3.43 | as above |
| HYSPLIT | NCEP/NCAR reanalysis | OKC80 | 1.77 | 3.36 | as above |
| HYSPLIT | ECMWF analyses of 1994, as in ATMES-II | ETEX-1 | **1.96** | — | DATEM description, Appendix C (R 0.48, FB 0.33, FMS 57.27, KS 68; 3,105 pairs); the meteorology is stated in §5.3 |
| HYSPLIT | not documented | CAPTEX | 2.23 | 3.03 | DATEM archive, `mdl_stat/` (run `007`) |
| HYSPLIT | not documented | ANATEX tracer 1 | 2.03 | 3.47 | as above |
| HYSPLIT | not documented | ANATEX tracer 2 | 1.73 | 2.48 | as above |
| HYSPLIT | not documented | ETEX-1 | 2.34 | 2.74 | as above |
| HYSPLIT | not documented | ACURATE | 1.96 | 3.44 | as above |
| HYSPLIT | not documented | OKC80 | 1.94 | 2.51 | as above |
| EASLEE | see the paper | ETEX-1, first 72 h | 2.76 | — | Brioude et al. (2026), doi:10.5194/egusphere-2026-5993 |
| EASLEE | see the paper | ETEX-1, four days | 2.41 | — | as above |
| EASLEE | see the paper | CAPTEX | 2.43 (paper), 2.51 (repository) | — | as above |

**ETEX-1 is a single release, so its whole-experiment score is also its
per-release score.** HYSPLIT on the 1994 ECMWF analyses (1.96) is the ETEX-1
threshold in the validation plan: it is the only ETEX-1 rank from an
established model with documented meteorology, scored on exactly DATEM's
file with DATEM's defaults.

"Time-averaged" is DATEM's `-at` option: each station's samples are averaged
over time, leaving one value per station, so only the differences between
stations remain. The EASLEE scores are reported alongside GLIDE's but set no
threshold ([validation-plan.md](validation-plan.md) Tier 3).

## 8. Gaps

To add when the papers are to hand:

- **The ATMES-II report, EUR 17756 EN:** the statistics for all 35 models,
  which models the numbers are, and the exact filter and KS definitions. It
  is not online; ask the JRC (the ETEX site offers the final report on
  request). Without it, ETEX-1 can only be compared with the four models in
  §6 and HYSPLIT.
- **Stohl et al. (1998):** FLEXPART on ETEX, CAPTEX and ANATEX with ECMWF
  global analyses at 1°, the closest meteorology to ERA5 in the literature.
  Its statistics are not the ranking score, so check which can be compared.
- **Ngan and Stein (2017)** and **Loughner et al. (2021):** HYSPLIT on CAPTEX
  and ANATEX, cited by Selvaratnam et al.
- **Selvaratnam et al. Fig. 5:** NAME's ranks as its meander parameters are
  varied (their runs M0–M6, Table 1). It is the prior for GLIDE's meander
  calibration (validation-plan §7) and can be extracted the same way as Fig. 1.
- **EASLEE:** the meteorology and scoring choices behind each score.

## 9. Where the numbers came from

- **Hegarty et al. (2013)**, J. Appl. Meteor. Climatol. 52, 2623–2637,
  doi:10.1175/JAMC-D-13-0125.1: Tables 1–4, transcribed. The means
  recomputed from the transcribed values match the tables' "Avg" rows.
- **Selvaratnam et al. (2023)**, J. Appl. Meteor. Climatol. 62, 1165–1174,
  doi:10.1175/JAMC-D-23-0021.1: the NAME values appear only as bar charts.
  The bars are vector graphics, so their heights were read from the PDF
  (with `pdfplumber`) rather than estimated by eye. Fig. 4 also plots
  Hegarty's HYSPLIT, STILT and FLEXPART values; fitting the scale to those 54
  published values reproduces every one to within 0.001. Fig. 1, read with
  its own axis, agrees with Fig. 4's NAME bars to within 0.01, and each
  "Average" bar equals the mean of its eight experiments. Treat the NAME
  values as accurate to ±0.01.
- **Mosca et al. (1998)**, Atmos. Environ. 32, 4307–4324: Tables 3, 5 and 6,
  transcribed and checked against the rendered page images, because the
  PDF's text layer garbles some rows. Each model's ranks in Table 5 sum to
  its printed "Sum of rankings".
- **DATEM** values are from the archive's files and description, as recorded
  in [../data/tracer-experiments.md](../data/tracer-experiments.md) §4.
- **EASLEE** values are as quoted in [validation-plan.md](validation-plan.md).

**The CSV** has one row per published value, with columns `source`, `model`,
`mode` (forward, or backward with the receptor height), `meteorology`,
`met_class` (`reanalysis` or `WRF`), `met_detail`, `experiment`, `rank`, the
four terms where published (`r2_term`, `fb_term`, `fms_term`, `ks_term`),
`how_obtained` and `notes`. Of section 7, only the two HYSPLIT ETEX-1 rows
are in it.
**published-etex-atmes2.csv** has one row per statistic per model:
`source`, `model`, `analysis` (`global` or `space`), `statistic`, `value`,
`ci95_low`, `ci95_high`, `atmes2_rank_of_35`, `how_obtained`, `notes`.
