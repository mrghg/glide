# Tracer-experiment data: ETEX, CAPTEX and ANATEX

_Where the validation tracer data comes from, how to get it, what the files
contain, and how GLIDE will use them. Written 2026-10-02 after checking both
archives by hand; re-check the access notes if a download fails._

These are the observations for validation Tier 3
([docs/validation-plan.md](../docs/validation-plan.md) §2). Like everything in
`data/`, the files are **not committed to this repository**. Code finds them
through environment variables, and falls back to `data/ETEX` and `data/DATEM`
in the checkout when a variable is unset. The `.gitignore` rule `data/**`
already keeps both directories out of git.

| Variable | Default | Contents |
| --- | --- | --- |
| `GLIDE_ETEX` | `data/ETEX` | the ETEX files below, in `ETEX_release1/` and `ETEX_release2/` |
| `GLIDE_DATEM` | `data/DATEM` | the DATEM files below, keeping DATEM's own directory names (`exp_data/captex/`, …) |

The public benchmark (roadmap item 1h) fetches them from the original archives
rather than redistributing them.

**Contents**

1. [ETEX (JRC)](#1-etex-jrc)
2. [CAPTEX and ANATEX (NOAA DATEM)](#2-captex-and-anatex-noaa-datem)
3. [Meteorology](#3-meteorology)
4. [Scoring: the DATEM ranking score](#4-scoring-the-datem-ranking-score)
5. [Processing plan](#5-processing-plan)

---

## 1. ETEX (JRC)

The European Tracer Experiment, 1994: two 12-hour releases from Monterfil,
Brittany, sampled at about 168 stations across Europe in 3-hour intervals.

| Release | Start (UTC) | End (UTC) | Tracer | Mass | Mean rate | Stack |
| --- | --- | --- | --- | --- | --- | --- |
| ETEX-1 | 1994-10-23 16:00 | 1994-10-24 03:50 | PMCH | 340 kg | 7.95 g/s | 8 m |
| ETEX-2 | 1994-11-14 15:00 | 1994-11-15 02:45 | PMCP | 490 kg | 11.56 g/s | 8 m |

Take the release coordinates from the experiment documentation, not from memory.

### Why the website download does not work

The site is `https://remon.jrc.ec.europa.eu/past_activities/etex/site/index.html`.
Its "Download → Datasets" page (`datasets.htm`) links to four **directories**, and
three things defeat a browser:

- The links point at directories, so a click shows a bare listing at best.
- A directory URL without a trailing slash redirects from `https` to plain
  `http`, which most browsers block or warn about.
- The inventory files `CONTENTS_1` and `CONTENTS_2` return 404 although the
  listing shows them, probably because the IIS server refuses files with no
  extension.

Every file needed below **is** served. Checked 2026-10-02.

### Files

Base URL: `https://remon.jrc.ec.europa.eu/past_activities/etex/site/database/`

| File | Contents |
| --- | --- |
| `ETEX_release1/pmch.dat` | PMCH concentrations |
| `ETEX_release1/pmch.cod` | quality flags, same layout |
| `ETEX_release1/pmch.readme` | format description |
| `ETEX_release1/release1.txt` | release details |
| `ETEX_release1/stationlist.950130` | station list |
| `ETEX_release1/letter.txt` | covering letter |
| `ETEX_release2/pmcp.datv1.2`, `pmcp.codv1.2`, `pmcp.readmev1.2`, `release2.txt`, `stationlist.950130` | the same for release 2 |

About 160 KB in total. To fetch them into `$GLIDE_ETEX`:

```bash
cd "$GLIDE_ETEX" && B=https://remon.jrc.ec.europa.eu/past_activities/etex/site/database && for f in ETEX_release1/{pmch.dat,pmch.cod,pmch.readme,release1.txt,letter.txt,stationlist.950130} ETEX_release2/{pmcp.datv1.2,pmcp.codv1.2,pmcp.readmev1.2,release2.txt,stationlist.950130}; do curl -fsS --create-dirs -o "$f" "$B/$f"; done
```

The `meteodat_1/` and `meteodat_2/` directories hold the original 1994
meteorology. GLIDE does not need them; it runs on ERA5.

### Concentration file format (from `pmch.readme`)

- Line 1: file information.
- Line 2: the 30 sampling-interval numbers, format `(9x,30I6)`. Interval 1 is
  23 Oct 1994 15–18 UTC, and each later interval is 3 hours on.
- Lines 3 to 170: station number, four-character station code, then 30
  concentrations, format `(1x,I3,2x,A4,30F6.2)`.
- Units ng m⁻³, ambient background already subtracted.
- `0.00` is a valid sample with no tracer; `-.99` means no sample or no
  analysis; `-.88` means tracer detected but not quantifiable.
- Release 2 has the same layout; check `pmcp.readmev1.2` for its interval
  origin.

### Terms of use

The 1996 readme says the data are "still restricted" to ATMES participants and
asks users to quote the data version (`etex1_v1.1.960505` for release 1). JRC now
publishes the files openly. Quote the version, cite the ETEX references, and do
not redistribute the files without confirming the terms with JRC.

---

## 2. CAPTEX and ANATEX (NOAA DATEM)

DATEM, the Data Archive of Tracer Experiments and Meteorology (Draxler, Heffter
and Rolph, 2001, revised 2002), holds several North American experiments in a
common text format, with Fortran statistics programs.

Page: `https://www.arl.noaa.gov/research/atmospheric-transport-and-dispersion/atd-programs-datem/`.
Description: `https://www.arl.noaa.gov/wp-content/uploads/documents/datem/datem.pdf`.

**Access.** The data directories under
`https://arl.noaa.gov/wp-content/uploads/documents/datem/` refuse scripted
requests: `curl` gets an empty "202 Accepted", which looks like bot protection.
**Download in a browser.** The landing page and the PDF do download by script.

**Acknowledgement, as NOAA requests:** "The authors gratefully acknowledge the
NOAA Air Resources Laboratory (ARL) for the provision of the Data Archive of
Tracer Experiments and Meteorology (DATEM) used in this publication."

### Which experiments

| Experiment | Use | Tracers | Releases and sampling |
| --- | --- | --- | --- |
| CAPTEX, Sep–Oct 1983 | **yes** | t1 PMCH | Seven releases (`emit-t1.txt`): five from Dayton OH, four of 3 hours and one of 30 minutes (28 Oct), and two of 3 hours from Sudbury ON; 84 sites at 300–800 km; 3- and 6-hour samples for about 48 h after each release |
| ANATEX, Jan–Mar 1987 | **yes** | t1 PTCH from Glasgow MT; t2 PDCH from St Cloud MN; t3 PMCH from St Cloud | 66 releases, every 2.5 days from each site; 75 sites over the eastern US and SE Canada; 24-hour samples. **Do not score t3**: its releases coincide with t2, so it is not independent |
| ETEX-1 | cross-check only | t1 PMCH | DATEM's reformatting of ETEX release 1. Use the JRC files (§1) as the observations; see "ETEX in DATEM" below |
| ACURATE, Mar 1982 – Sep 1983 | **reported, not graded (v1.x)** | t1 Kr-85 | Continuous emission from the Savannah River Plant, SC; five sites at about 300–1,000 km along the US east coast; mostly 12-hour samples, some 24-hour. See "ACURATE" below |
| OKC80, Jul 1980 | **benchmark test only** | PMCH, PDCH | One 3-hour release from Norman OK on 8 July; 697 samples of 45 minutes and 3 hours, on arcs at about 100 km and 600 km, over four days. The tracer benchmark (task B12) is tested on it before the pre-registration, because it is not graded |
| INEL74 | no | Kr-85 | Continuous source and one arc of samplers; the same background problem as ACURATE with less coverage |
| Project Sagebrush | no | — | Sub-kilometre scale, beyond a reanalysis-driven model |

### What to download

| DATEM directory | Files | Why |
| --- | --- | --- |
| `exp_data/<exp>/` | `emit-t?.txt`, `meas-t?.txt`, `stations.txt`, `readme.txt` | emissions, measurements, sites, units |
| `document/` | original experiment reports | release heights, sampler details |
| `stat_pgm/` | `statmain/statmain.f` and `c2datem/c2datem.f` | reference implementation of the scoring |
| `mdl_data/<exp>/` | `ct?_000.txt` | HYSPLIT predictions, to verify our scoring code |
| `mdl_stat/<exp>/` | `statA?_000.txt` (all pairs), `statT?_000.txt` (time-averaged), `run_stat.sh` | HYSPLIT statistics our code must reproduce, and the options that made them |

Here `<exp>` is `captex`, `anatex`, `etex` and `acurate`, and optionally
`okc80`. The `met_data/` files (NCEP/NCAR reanalysis on sigma levels) are not
needed.

**The DATEM readmes give the wrong file names.** `mdl_data/Readme.txt` and
`mdl_stat/Readme.txt` (dated 2013) describe `ct1_001.txt`, `gbl1_001.txt` and
`avg1_001.txt`. The files actually posted are `ct?_000.txt`, `statA?_000.txt`
and `statT?_000.txt`. The header of every stats file names itself
`stat?1_007.txt` with "Model variation: 007". `statmain` writes the run label
it was given into this header, so the files were produced as run `007` and
renamed afterwards. They do belong to the posted predictions. Checked
2026-11-04: `statmain`, compiled with gfortran 16.2 and run through each
`run_stat.sh` with run label `000` on a copy of the archive, reproduces all
twelve `statA?_000.txt` and `statT?_000.txt` files byte for byte, apart from
the run label in lines 2–3 (CAPTEX, ANATEX tracers 1 and 2, ETEX-1, ACURATE
and OKC80).

### What the scripts in `mdl_data` and `mdl_stat` do

None of them needs to be run. They record how DATEM made the posted files.

- **`stat_pgm/`** holds the Fortran source of two programs. The `compile.bat`
  files call the Intel compiler on Windows (`ifort /convert:big_endian`). The
  source is free-form Fortran in a `.f` file, so `gfortran` needs
  `-ffree-form`; `gfortran -O1 -ffree-form -ffree-line-length-none
  -fconvert=big-endian -o statmain statmain.f` builds it without warnings.
  Run it on a copy of the archive: it writes its results into the current
  directory, over the posted files. If an input file is missing it waits for
  a keypress, so redirect standard input from `/dev/null` in scripts.
  - `c2datem` reads a binary HYSPLIT concentration file (`cdump`) and writes
    one predicted value for each measurement, in the measurement format.
  - `statmain` reads the measurements (`<dir>/meas-t<tracer>.txt`) and the
    predictions (`<dir>/ct<tracer>_<run>.txt`) and writes the statistics and
    the ranking score to `statA<tracer>_<run>.txt` or `statT<tracer>_<run>.txt`.
- **`mdl_data/<exp>/bin2txt.sh`** runs `c2datem` on HYSPLIT's binary output
  (`ct?.bin`) to make `ct?_000.txt`, and renames the binary to `ct?_000.bin`.
  **`txt2bin.sh`** only renames the binary back, so the conversion can be
  rerun. The `.bin` files are HYSPLIT's raw output and are not needed; the
  `.txt` files are what we score.
- **`mdl_stat/<exp>/run_stat.sh`** runs `statmain` twice per tracer: once on all
  pairs (giving `statA`) and once with `-at`, time-averaged (giving `statT`).
  It uses `-z0.0` (measurements of zero count as zero) and does not pass `-x`,
  so zero–zero pairs are included. These are the options to reproduce in B10.

**CAPTEX specifics.** The CAPTEX HYSPLIT run is split across several binary
files, apparently one per release: `ct1`, `ct2`, `ct3`, `ct4`, `ct5` and `ct7`,
with no `ct6`. `bin2txt.sh` converts each one and appends them all to a single
`ct1_000.txt`. As posted, `ct1_000.bin` is empty (0 bytes). This does not
matter, because `ct1_000.txt` is complete: one prediction for each of the
2,281 measurements, covering all seven release periods.

### ETEX in DATEM

`exp_data/etex/` is ETEX release 1 only, reformatted from the ATMES data
(`mea95t1.txt`). Compared with the JRC files of §1:

- there is no emissions file and no release 2;
- concentrations are in pg m⁻³ rather than ng m⁻³, and the station codes are
  numbers (`1002`) rather than letters (`A02`): the country prefixes in
  alphabetical order are `1`–`8` (A to F), `90` (H) and `91`–`98` (N to UK),
  followed by the JRC station number, so `8002` is F02 and `9203` is NL03;
- the JRC quality flags are dropped, and so are the samples marked `-.88`
  (detected but not quantifiable);
- JRC has 3,104 valid samples, 939 of them non-zero; DATEM has 3,105, 941 of
  them non-zero.

**Matched sample by sample** (B10, `python -m tracers etex-match`), the two
files agree on 3,103 samples. Every difference is explained:

| Station, interval | JRC (v1.1) | DATEM | Explanation |
| --- | --- | --- | --- |
| D10, 2 (23 Oct 18–21 UTC) | `-.88`, flag 41 | 1.76 ng m⁻³ | JRC changed this sample from 1.76 to `-.88` after 5 May 1996 (`pmch.readme`, "changes made"). DATEM's copy predates the change. |
| D28, 3 (23 Oct 21–24 UTC) | 0.00, flag 01 | 0.60 ng m⁻³ | Not documented. Most likely an error in the ATMES-era file: the release began at 16 UTC in Brittany, and D28's next positive sample is 0.16 ng m⁻³ in interval 11. Treated as a reference issue. |
| D13, F10 (locations) | 50.12 N 8.73 E; 49.52 N 1.50 W | 50.05 N 8.60 E; 49.63 N 1.62 W | The JRC station list (30 Jan 1995) records "D13: change of station" and "F10: corrected"; DATEM keeps the old locations. F27 differs by 0.02°, rounding. |

The other JRC records absent from DATEM are the 1,566 not sampled (`-.99`)
and the 369 not quantifiable (`-.88`). So DATEM's ETEX file is the ATMES
version of the data (before v1.1), which is also what explains most of the
ATMES-II pair count (§5).

Use the JRC files as the ETEX observations. DATEM's version is kept for two
reasons. Its HYSPLIT predictions and statistics (`mdl_data/etex/`,
`mdl_stat/etex/`) are a third case for verifying the scoring code. And
matching the two ETEX-1 files sample by sample, and explaining the
differences, is a test of the readers.

### ACURATE (reported, not graded; v1.x)

The Atlantic Coast Unique Regional Atmospheric Tracer Experiment measured
krypton-85 from the Savannah River Plant (33.28° N, 81.66° W), emitted
continuously by nuclear-fuel reprocessing. Emissions are given hourly from
5 March 1982 to 30 September 1983 in curies, averaging 55 Ci h⁻¹ over all
hours and 110 Ci h⁻¹ while emitting. Five sites, from North Carolina to New
Jersey, took mostly 12-hour samples in pCi m⁻³.

It is not one of the graded Tier 3 tests, for three reasons:

- **Background.** Kr-85 has a large global background from reprocessing
  everywhere, so the plume is a small difference on top of it. DATEM gives
  two versions of the measurements: `meas-t0.txt` uses the original
  background analysis, and `meas-t1.txt` subtracts a variable background
  **from a global model** (Draxler, 2007, doi:10.1016/j.atmosenv.2006.08.052).
  The observations therefore partly depend on another model.
- **Few sites.** With five samplers, the spatial scores mean little.
- **Continuous emission.** Nearly every sample contains some tracer, so the
  figure of merit in space hardly discriminates between models, and the
  scores mostly test diluted, accumulated transport rather than where a plume
  went.

It is still worth running, because it is the experiment closest to how GLIDE
is used: a continuous point source seen for many months at a few sites
through long samples, which is a receptor-footprint, time-series problem like
a greenhouse-gas inversion. GLIDE scores it on both backgrounds, and the
difference between the two is reported as an uncertainty in the observations.
HYSPLIT's statistics in `mdl_stat/acurate/` (against `meas-t1.txt`) are
reported alongside. It is a v1.x item (roadmap 1i), so it is not part of the
v1 pre-registration.

### File formats (from the DATEM description)

All times are UTC; line 1 names the file and line 2 names the columns.

- **Emissions, `emit-t?.txt`:** year `I4`, month `I3`, day `I3`, start hour-minute
  `I5`, duration hour-minute `I5`, latitude `F6.2`, longitude `F8.2` (east
  positive), total emission over the duration `F8.0` in mass units.
- **Measurements, `meas-t?.txt`:** the same first seven fields for sample start,
  duration and location, then concentration `F8.1` in pg m⁻³ and site name `A8`.
- Concentrations are background-subtracted. ANATEX and CAPTEX volume
  concentrations were already converted to mass by DATEM.
- DATEM includes all measured data regardless of the original quality flags.
- Model predictions use the measurement format exactly, named `ct?_<variant>.txt`,
  one record per measurement.

---

## 3. Meteorology

GLIDE uses ERA5 for every experiment, so it runs through the normal
pipeline. Each backward run must reach back to the start of the release it is
scored against. ACURATE's source is continuous, so its backward runs need only
be long enough to cover the travel time from the plant to the furthest site
(a few days).

| Experiment | Period to cover | Domain |
| --- | --- | --- |
| ETEX-1 | 1994-10-23 to 1994-10-27 | Europe |
| ETEX-2 | 1994-11-14 to 1994-11-18 | Europe |
| CAPTEX | 1983-09-18 to 1983-10-30 | eastern US and SE Canada, including Dayton and Sudbury |
| ANATEX | 1987-01-05 to 1987-01-18 for v1 (the ten scored releases); the full experiment, to 1987-03-29, in v1.x | most of the US and southern Canada, including Glasgow MT |
| ACURATE (v1.x) | 1982-03-01 to 1983-09-30 | south-eastern US, from Georgia to New Jersey |
| OKC80 (benchmark test) | 1980-07-08 to 1980-07-12 | Oklahoma to Missouri, about 34–42° N, 99–90° W |

Prepare these with [met-bazarr](https://github.com/mrghg/met-bazarr)
([decision 0018](../dev/decisions/0018-met-bazarr.md)) on ERA5 model levels,
the default ([decision 0017](../dev/decisions/0017-model-levels-default.md)),
fetching only the levels below the domain top.

ANATEX covers about 2,000 hours over most of a continent, so it is the largest
cube. Size it before downloading; the compressed met cache (queue task B1) helps.
ACURATE's domain is small but its period is 19 months, so size it too.
Add each cube to the inventory in `dev/agent/PROGRESS.md`.

---

## 4. Scoring: the DATEM ranking score

From the DATEM description, Appendix B:

$$
\mathrm{RANK} = R^2 + \left(1 - \left\lvert \tfrac{FB}{2} \right\rvert\right) + \frac{FMS}{100} + \left(1 - \frac{KS}{100}\right)
$$

R is the correlation of paired values, FB the fractional bias, FMS the figure of
merit in space (percentage overlap of non-zero predicted and measured samples),
and KS the Kolmogorov–Smirnov parameter (maximum difference between the
unpaired cumulative distributions, in percent). Each term lies between 0 and 1,
so the best score is 4. The appendix text describes the first term as |R|, but
the formula and later papers use R²; follow the formula and DATEM's code.

The options change scores a lot, so the pre-registration must fix them:

- the percentile below which measurements count as zero (`-z`);
- whether zero–zero pairs are included (`-x` excludes them);
- unaveraged ("global") versus time- or space-averaged analysis (`-at`, `-as`);
- whether the correlation is computed by resampling (`-b`).

**The settings that match the published studies** (Hegarty et al., 2013;
Selvaratnam et al., 2023; details and evidence in
[../docs/published-tracer-scores.md](../docs/published-tracer-scores.md) §2)
are: `-z0`, zero–zero pairs included, global analysis, and `-b`, with each
CAPTEX release (1, 2, 3, 4, 5, 7) and the first ten ANATEX releases (five from each
site, 5–16 January 1987) scored by site as separate experiments. The DATEM description
recommends the same defaults: "use all the data and rely upon more robust
statistical parameters".

`-b` replaces the correlation with the root-mean-square of the correlations
from N passes (N is the number of records), in each of which every pair is
dropped independently with probability 0.2, so about 20% of the pairs. It was
added to `statmain` in October 2012, for Hegarty et al. Repeated runs differ
in the second decimal place (three CAPTEX runs gave ranks of 2.25, 2.25 and
2.24, against 2.23 without `-b`). The comment in the code says it seeds the
generator from the clock, but it does not: the clock value is stored in the
variable that the first `RANDOM_NUMBER` call overwrites, and `RANDOM_SEED`
is never called, so the sequence is whatever the compiler's default is
(random per run with gfortran 7 and later; with some compilers, the same
every run). B10's version takes a seed.

For scale, HYSPLIT's ranks from two sources are below. They differ, and
are not meant to agree: the DATEM description's Appendix D (2001, revised
2002) reports HYSPLIT with NCEP reanalysis, while the posted `mdl_stat` files
come from a later HYSPLIT run, `007`, whose configuration the archive does not
document. **The posted files are the ones the B10 scoring code must
reproduce.** Neither is a threshold for GLIDE; the thresholds come from the
published per-release comparisons (validation-plan §3). Both are for whole
experiments, so they cannot be compared with the per-release scores.

| Experiment | Appendix D, all pairs | Appendix D, time-averaged | `statA?_000`, all pairs | `statT?_000`, time-averaged |
| --- | --- | --- | --- | --- |
| CAPTEX | 1.36 | 3.35 | 2.23 | 3.03 |
| ANATEX tracer 1 | 1.69 | 2.69 | 2.03 | 3.47 |
| ANATEX tracer 2 | 1.74 | 2.92 | 1.73 | 2.48 |
| ETEX-1 | — | — | 2.34 | 2.74 |
| ACURATE | — | — | 1.96 | 3.44 |
| OKC80 | — | — | 1.94 | 2.51 |

Published comparisons on the same score, per release: NAME driven by
ERA-Interim and by WRF (Selvaratnam, Thomson and Webster, 2023, J. Appl.
Meteor. Climatol. 62(9), doi:10.1175/JAMC-D-23-0021.1); HYSPLIT and STILT
driven by NARR and by WRF, and FLEXPART by WRF (Hegarty et al., 2013,
doi:10.1175/JAMC-D-13-0125.1). All are forward runs, apart from Hegarty et
al.'s backward STILT runs. Every value, with its meteorology, is in
[../docs/published-tracer-scores.md](../docs/published-tracer-scores.md).

For ETEX, also compute the ATMES-II statistics (Mosca et al., 1998) so results
compare with the original exercise. They use their own method, not DATEM's:
the zero filter (measured zeros kept only from 6 h before the cloud arrives
at a station to 6 h after it leaves, which leaves 1,423 pairs), correlation
of the logarithms, zeros set to 0.01 ng m⁻³ in logarithmic indices, and the
figure of merit in space from interpolated contours at 0.1 ng m⁻³. The method
and the published values are in
[../docs/published-tracer-scores.md](../docs/published-tracer-scores.md) §6.
Use the JRC files: the filter needs the quality flags and the `-.99` missing
samples, which DATEM's copy drops. The graded ETEX-1 score, by contrast, is
the DATEM rank on **DATEM's** copy with DATEM's defaults, so that it is
computed exactly as HYSPLIT's published 1.96 was (validation-plan §3).

---

## 5. Processing plan

Implemented by queue tasks B10 (readers and scoring), B12 (the benchmark
runner) and A11 (the registered runs). Update this section with what was
actually built.

Steps 1 and 2 are built (B10), in the package `scripts/validation/tracers/`.
Its command line runs from `scripts/validation` as `python -m tracers
<command>`; the commands are listed in `tracers/__main__.py`. Tests:
`tests/test_tracer_scoring.py` (synthetic files, and checks against the
archives that skip when `$GLIDE_DATEM` or `$GLIDE_ETEX` is unset).

1. **Readers** (`tracers/readers.py`) for both formats into one table of
   releases (start, end, duration, location, height, mass) and one of samples
   (ID, site, start, end, duration, location, height, concentration,
   quality flag), in SI units: kg and kg m⁻³, or Bq and Bq m⁻³ for ACURATE's
   Kr-85. The raw DATEM fields are kept, because `statmain` pairs predictions
   with measurements on them. JRC station coordinates are degrees and
   minutes and are converted. `read_acurate` reads `meas-t0.txt` and
   `meas-t1.txt` and labels each record with its background (t0: the original
   background analysis, arl-080; t1: the variable background from a global
   model). `write_datem_predictions` writes predictions in the measurement
   format with the measurement keys unchanged, and refuses any non-finite
   prediction (written as `nan`, `statmain` would read it as missing and drop
   the pair without a word; B10b). The JRC files do not give the source
   coordinates; `ETEX_RELEASES` takes ETEX-1's from the JRC ATMES-II pages
   ("The first ETEX tracer release",
   https://remon.jrc.ec.europa.eu/past_activities/atmes_2/site/11a.htm):
   Monterfil, 48°03'30" N, 2°00'30" W, ground 90 m above sea level, release
   8 m above ground (B10b). ETEX-2 was released from the same site (JRC ETEX
   overview, https://remon.jrc.ec.europa.eu/past_activities/etex/site/index.html). The ETEX reader is checked by
   matching JRC's ETEX-1 against DATEM's sample by sample (§2, "ETEX in
   DATEM").
2. **Scoring** (`tracers/datem_stats.py`) re-implements `statmain` routine by
   routine, with every option the pre-registration must fix as an argument
   of `StatOptions`: `zero_percentile` (`-z`), `exclude_zero_pairs` (`-x`),
   `model_zero` (`-y`), `plume_only` (`-p`), `averaging` (`-at`, `-as`),
   `bootstrap` with a required `seed` (`-b`), and `contingency_level` (`-l`).
   It works in single precision with sequential sums, as `statmain` does;
   in double precision six of the twelve files differed in one printed
   digit (factor-of-5 counts at ratio boundaries, averages in `-at`).
   - **Verified** (`python -m tracers verify-datem`): on DATEM's HYSPLIT
     predictions with `run_stat.sh`'s options it reproduces every printed
     value of all twelve posted files, `statA` and `statT` for CAPTEX, ANATEX
     tracers 1 and 2, ETEX-1, ACURATE and OKC80 (ranks 2.23/3.03, 2.03/3.47,
     1.73/2.48, 2.34/2.74, 1.96/3.44, 1.94/2.51).
   - **Resampled correlation.** Over seeds 0–19, the mean rank with `-b` lies
     inside the range of 20 runs of DATEM's own `statmain -b` (gfortran 7.5,
     rank printed to five decimals) for every experiment:

     | Experiment | B10, mean of 20 seeds | `statmain -b`, 20 runs |
     | --- | --- | --- |
     | CAPTEX | 2.2449 | 2.2424–2.2476 |
     | ANATEX tracer 1 | 2.0818 | 2.0787–2.0833 |
     | ANATEX tracer 2 | 1.7323 | 1.7323–1.7324 |
     | ETEX-1 | 2.3366 | 2.3340–2.3393 |
     | ACURATE | 1.9635 | 1.9627–1.9644 |

   - **The published experiments** (`tracers/experiments.py`). Neither paper
     says which samples belong to which release. The candidate rules and
     their pair counts (`python -m tracers rules`; "6 h" is the duration
     rule `6h_or_24h`, which keeps CAPTEX's 6-hour samples, dropping its
     3-hour ones, and ANATEX's samples, which are all 24 h):

     | Rule | C1 | C2 | C3 | C4 | C5 | C7 | GGW | STC | Σ\|diff\| |
     | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
     | Hegarty et al. (2013) | 354 | 365 | 346 | 299 | 306 | 183 | 562 | 556 | 0 |
     | 6 h, most recent release | 347 | 352 | 356 | 281 | 318 | 177 | 813 | 800 | 561 |
     | 6 h, start within 48 h of release | 341 | 315 | 344 | 279 | 318 | 176 | 676 | 668 | 330 |
     | 6 h, start within 60 h | 346 | 347 | 354 | 281 | 318 | 177 | 813 | 800 | 565 |
     | 6 h, whole sample within 60 h | 344 | 334 | 352 | 281 | 318 | 177 | 469 | 465 | 267 |
     | 6 h, whole sample within 72 h | 347 | 352 | 356 | 281 | 318 | 177 | 676 | 668 | 292 |
     | all, most recent release | 395 | 400 | 404 | 367 | 325 | 288 | 813 | 800 | 821 |
     | all, start within 60 h | 394 | 395 | 402 | 367 | 325 | 288 | 813 | 800 | 813 |
     | all, whole sample within 60 h | 392 | 382 | 400 | 367 | 318 | 288 | 469 | 465 | 478 |
     | 6 h, most recent; ANATEX samples starting 5–12 Jan | 347 | 352 | 356 | 281 | 318 | 177 | 507 | 501 | 176 |
     | 6 h, most recent; ANATEX 5–12 Jan, sites inside WRF d02 | 347 | 352 | 356 | 281 | 318 | 177 | 484 | 478 | 222 |
     | **6 h, most recent; ANATEX samples starting 5–13 Jan** | **347** | **352** | **356** | **281** | **318** | **177** | **577** | **570** | **95** |
     | 6 h, most recent; ANATEX 5–13 Jan, sites inside WRF d02 | 347 | 352 | 356 | 281 | 318 | 177 | 551 | 544 | 89 |
     | 6 h, most recent; ANATEX samples starting 5–14 Jan | 347 | 352 | 356 | 281 | 318 | 177 | 646 | 638 | 232 |
     | 6 h, most recent; ANATEX 5–14 Jan, sites inside WRF d02 | 347 | 352 | 356 | 281 | 318 | 177 | 617 | 609 | 174 |

     **Adopted** (`experiments.ADOPTED_RULE`): for CAPTEX, the 6-hour
     samples, each assigned to the most recent release (release 6's samples
     are left out with it); for ANATEX, every 24-hour sample starting from
     5 to 13 January inclusive, the last ending at 14 January 14 UTC. CAPTEX
     is within 6–18 pairs of Hegarty et al. for every release, with no
     consistent sign; no release-window rule does better. For ANATEX no
     release-based rule comes near, because the five releases of each tracer
     run to 15 January and Hegarty et al. have fewer pairs than samples in
     that period; the date cut is empirical, and leaves 15 and 14 pairs more
     than published. The full table, with 48- and 72-hour windows, is
     printed by `rules`.
   - **The ANATEX surplus is not sites outside the WRF domain** (B10b,
     24 Nov 2026). Hegarty et al.'s ANATEX runs used AER's WRF inner
     domain, which NOAA ARL posts with DATEM as
     `https://www.ready.noaa.gov/data/archives/datem/anatex/aer_anatex_PBL1A_*.d02`
     (5–16 January). The first index record of those files (only the header
     was read) gives the grid: Lambert conformal, true latitudes 33° and
     45° N, reference longitude 100° W, 10 km, 454 × 334 points, point
     (227, 167) at 41.2° N, 98.02° W (`experiments.ANATEX_WRF_D02`;
     spherical projection on ARL's 6371.2 km earth). Three ANATEX sites lie
     outside it: 2601 (25.90° N, 97.43° W; 3.3 cells south), 3052 (53.30° N,
     60.37° W; 19 cells east, 26 north) and 3007 (41.67° N, 69.97° W;
     0.35 cells east, outside under every variant of earth radius and
     half-cell origin tried). Under the adopted date cut they hold 26 pairs
     per tracer (9, 9 and 8), so dropping them gives 551 and 544 against
     Hegarty et al.'s 562 and 556: 11 and 12 too few instead of 15 and 14
     too many. The total difference falls only from 95 to 89, because the
     cut overshoots rather than closes the gap, and no other date cut
     combined with it does better. Dropping only the two sites well outside would give 559 and
     552, but there is no reason to keep 3007, and whether a site outside
     the meteorology loses its pairs at all depends on each model's
     concentration grid, which DATEM pairs on (a zero is still a pair). The
     domain cut is therefore **not** adopted; it is available as
     `AssignmentRule(..., anatex_wrf_domain=True)` and listed by `rules`.
     The remaining difference is unexplained.
   - **The sample list for A11** (`python -m tracers export-samples DIR`)
     writes `tracer_samples.csv` (experiment, assigned release, tracer,
     sample ID, site, latitude, longitude, UTC start and end) and
     `tracer_receptors.csv` (the unique receptor windows, with the
     experiments each serves): 2,978 samples and 2,387 receptor windows,
     because the two ANATEX tracers share 591 sites and times.
   - **ATMES-II** (`tracers/atmes2.py`) on the JRC ETEX-1 file. The zero
     filter, taking arrival and departure as each station's first and last
     positive sample over all 30 intervals, keeping zeros up to two intervals
     either side, and treating `-.88` as lost, gives **1,415 pairs, 33.6%
     measured zeros**, against the published 1,423 and 34%. The difference is
     the data version: with DATEM's (ATMES-era) values for D10 interval 2
     and D28 interval 3 (§2), the same rule gives 1,424. Other rules are far
     off (a 1-interval margin 1,240; 3 intervals 1,575; stopping at 72 h
     1,209; `-.88` counted as tracer 1,522). The statistics: bias with a 95%
     confidence interval, NMSE, MG = exp(⟨ln P⟩ − ⟨ln M⟩) (the published
     signs fix the direction), VG, the correlation of the logarithms, FA2,
     FA5, FOEX, KS, and FMS in space from log-linear interpolation on the
     stations' Delaunay triangulation in an equal-area projection, at 12–72 h
     and for the dose. **KS:** the published values (72–365) cannot be
     percentages. The code returns the maximum difference D of the two
     empirical distributions and D × N (on 1,423 pairs the published range
     is D = 0.05–0.26), but this is a hypothesis: the definition is in the
     JRC report EUR 17756 EN, not in the paper. The same holds for whether a
     zero counts within a factor of 2 for FA2 (`fa_zeros`). `python -m
     tracers atmes2-hysplit` runs the whole chain on DATEM's HYSPLIT
     example, as a check of the plumbing only.
3. **Receptor-oriented backward runs**: one backward release per measured
   sample, particles released uniformly over the sample period and between the
   ground and 100 m, the layer the published forward runs averaged over; a
   second set at sampler height as a sensitivity test. The predicted
   concentration is the release's emission rate times the surface sensitivity
   in the footprint cell that contains the source, summed over emission
   intervals. Hegarty et al.'s backward runs spread the source over 1°; that
   is reported as a sensitivity test. Both ANATEX tracers are sampled at the same sites and times, so one
   backward footprint per sample scores both source locations. ACURATE (v1.x)
   is run the same way, with each sample's sensitivity multiplied by the hourly
   emissions; it is scored on both backgrounds.
4. **Expected differences**, declared in the pre-registration: GLIDE runs
   backward and the published scores are forward; ERA5 versus the comparison
   models' meteorology. The backward-against-forward agreement is also a
   real-case test of the backward formulation.
5. **The benchmark command** (task B12) wraps steps 2–3. Given a GLIDE
   configuration and a meteorology store, it runs the backward releases for
   the chosen experiments, scores them, and writes a scorecard beside the
   published values. It is `tracers/benchmark.py`, run from
   `scripts/validation` as `python -m tracers.benchmark <command>`; tests in
   `tests/test_tracer_benchmark.py` and `tests/test_sample_release.py`.
   See "Running the benchmark" below.

### Running the benchmark

**Before the pre-registration, run it on OKC80 only.** `run` refuses CAPTEX,
ANATEX and ETEX unless `--registration-record` names a pre-registration
record whose `**Registration.**` line is filled in (validation-plan Tier 3);
`plan` prints sizes for any experiment and writes nothing. The experiments are
`OKC80` (PMCH, DATEM tracer 1), `OKC80-PDCH` (tracer 2, sampled at the same
windows, so the same footprints serve both), and the eight published ones
(`CAPTEX-1` … `CAPTEX-7`, `ANATEX-GGW`, `ANATEX-STC`, with B10's adopted
sample rule). ETEX is not wired in yet; ETEX-1's source coordinates are now in
`readers.ETEX_RELEASES` (step 1).

From `scripts/validation`, with `PYTHONPATH` pointing at the GLIDE `src` to
use and, on a GPU node, `scripts/gpu_env.sh` sourced:

```bash
python -m tracers.benchmark plan --experiments OKC80 OKC80-PDCH
python -m tracers.benchmark run --experiments OKC80 OKC80-PDCH \
    --met "$GLIDE_DATA/era5-arco/NAM_19800708-19800712_ERA5-ARCO.zarr" --out RUN_DIR
```

**What `run` does.**

- **Releases.** One backward release per unique sample window (location,
  start, end), release kind `samples`: particles uniform over the sampling
  period and between the ground and 100 m (`--receptor-layer 0 100`, the
  layer the published forward runs averaged over; give e.g. `10 10` for
  receptors at sampler height). `--particles` per window (default 2,000),
  `--seed` for the releases (per window: seed + index).
- **GLIDE runs.** One footprint grid per source site: 3 × 3 cells of
  `--source-deg` (default 0.25°; 1° is Hegarty et al.'s backward set-up)
  centred on the source, and one layer, the lowest 40 m (`--source-layer-m`,
  the registered footprint layer). Every particle is followed for one length
  from its own release, so the windows are split into GLIDE runs whose starts
  span at most `--max-span-h` (24 h), each with the length its latest window
  needs to reach back to the first emission, and each needing meteorology no
  earlier than the store's start. Age bins are the widest width (≤ 1 h) that
  every sample and emission time is a multiple of (5 minutes for OKC80), so
  the emission windows fall on bin edges.
- **Physics** comes from `tracers/benchmark_base.yaml` (the reference
  configuration's physics: Hanna turbulence with meander 0.16, reduced
  Emanuel convection, 60 s steps), or from `--config`, with any value
  changed by `--set key.path=value`. The horizontal met domain defaults to
  the store's extent less 0.5°.
- **Predictions** (`tracers/concentration.py`). For a source emitting mass M
  uniformly over [E₀, E₁] (rate Q) into the source box of volume V (area ×
  40 m), the predicted value for a sample is
  Q / V × (M_air / V_m) × Σₖ φₖ τₖ / ρₖ: τₖ is the residence time in the box
  in age bin k (from the end of the sample window), φₖ the fraction of the
  bin's absolute interval inside [E₀, E₁], and ρₖ the dry-air density at the
  source. This is the dry-air mole fraction converted with the observations'
  fixed factor M / V_m (decision 0022; V_m = 22.40 L mol⁻¹, assumed also for
  OKC80), not the ambient mass concentration (`--basis ambient`, Q / V ×
  Σₖ φₖ τₖ), which is 10–15% lower in warm air. GLIDE's default footprint is
  density-weighted (`output_grid.density_weighting: dry_air`, task A04,
  decision 0022: s m³ kg⁻¹, already τ/ρ per particle), which the "stp" basis
  uses as it is; the benchmark reads the store's `density_weighting` and
  `units` attributes, never assumes them. An ambient run needs plain seconds:
  `run ... --basis ambient --set output_grid.density_weighting=none` (`run`
  refuses the ambient basis on a weighted configuration before GLIDE starts).
  On a `none` store the "stp" basis divides by ρₖ read from the meteorology at
  the grid point nearest the source (surface pressure, and temperature and
  humidity on the lowest model level), interpolated to each bin's centre. The
  derivation is in the module's docstring.
- **Scoring.** Predictions are written in DATEM format
  (`predictions/<experiment>.txt`, pg m⁻³, the measurement keys unchanged)
  and scored with `datem_stats` under the published protocol: `-z0`,
  zero–zero pairs kept, unaveraged, R² from `-b` with `--bootstrap-seed`;
  the rank with the plain correlation is reported too. These options are
  provisional until the pre-registration fixes them.
- **Scorecard.** `scorecard.md`, `scorecard.csv` and `scorecard.json`: rank
  and its four terms per experiment, the published values for the same
  experiment grouped by class of meteorology
  (`docs/published-tracer-scores.csv`), the mean over the eight published
  experiments when all are scored, DATEM's own HYSPLIT example scored on the
  same samples and options where the archive has one, and the provenance:
  physics tag, GLIDE commit, scoring-code commit, configuration hash (physics
  and numerics only), the overrides, the meteorology store and its
  met-bazarr attributes, including any repairs recorded in the store's
  `glide_repairs` attribute (a summary such as "met-bazarr 0.3.0, repaired
  0.4.0 (era5-surface-geopotential)", then each repair's version, commit,
  date and size), and the seeds.

**Rerunning.**

- *With a changed parameter:* `run ... --out RUN_B --set
  turbulence.meander.coefficient=0.3`, then `python -m tracers.benchmark
  compare RUN_A RUN_B`. `compare` lists the settings that differ, every
  score and term side by side with the differences from the first run, each
  run's meteorology store with its producer version and repairs, and
  whether the predictions are identical.
- *With another meteorology store:* `run ... --met OTHER.zarr --out RUN_C`.
  The store must cover the samples and reach back to a few hours before the
  first emission; the scorecard records its met-bazarr version and commit.
- *On a parameter ensemble (A7):* one `run` per member, each with its own
  `--set` values and `--out`, then `compare` over all of them. Keep the
  release and bootstrap seeds fixed across members so that only the
  parameters differ.
- *Another bootstrap seed, or the other concentration basis where the store
  allows it, without rerunning GLIDE:* `python -m tracers.benchmark score
  RUN_DIR --bootstrap-seed 1 --out RUN_DIR_B1`. A `dry_air` store serves only
  "stp"; a `none` store serves both. Scoring is written to a temporary
  directory and published (metadata, predictions and scorecard together) only
  when it succeeds, so a refused rescoring leaves the run's record unchanged;
  an existing `--out` must already link to the same run.

**Longitudes.** The benchmark takes the meteorology's extent from GLIDE's
reader (`ArcoEra5ZarrReader.get_lon_lat_coverage`, after its normalisation:
NAM stores, ascending 0..360 on disk, are read as −120..−55) and writes every
longitude inside it, so the configuration uses the convention the runtime sees
and passes its preflight. (Before PR #56 the reader kept such stores in 0..360
and clamped negative longitudes to the western edge; STATUS.md.)

A rerun with the same seeds reproduces the scorecard (checked on OKC80, see
below). Every run before the pre-registration is exploratory and belongs in
its development history.

### OKC80 (exploratory, 2026-11-24, task B12)

Not a graded experiment; run before the pre-registration to test the
benchmark, and to be listed as exploratory in the pre-registration's
development history. Run on the code merged with `main` (PRs #56 and #57:
longitudes as the reader presents them, −180..180, with no 0..360
workaround; footprints dry-air weighted per particle, s m³ kg⁻¹).
Configuration: `benchmark_base.yaml` unchanged, 2,000 particles per window,
release seed 42, bootstrap seed 0, receptors 0–100 m, 0.25° source box,
lowest 40 m, basis "stp", on `era5-arco/NAM_19800708-19800712_ERA5-ARCO.zarr`
(met-bazarr 0.3.0, surface-geopotential repair by met-bazarr 0.4.0 recorded
in its attributes). 700 sample windows in four GLIDE runs on one GH200,
321 s of GLIDE wall time in all.

| Experiment | Pairs | Rank | R² | 1−\|FB/2\| | FMS/100 | 1−KS/100 | Rank, plain R | Mean meas. | Mean calc. (pg m⁻³) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OKC80 (PMCH) | 697 | 1.641 | 0.091 | 0.905 | 0.124 | 0.520 | 1.637 | 1,106 | 914 |
| OKC80-PDCH | 683 | 2.215 | 0.107 | 0.960 | 0.237 | 0.910 | 2.210 | 987 | 911 |
| DATEM's HYSPLIT example, PMCH, same options | 697 | 1.942 | 0.068 | 0.962 | 0.262 | 0.650 | | | |

- These supersede the branch-local numbers recorded before the merge (ranks
  1.63 and 2.21, means 907 and 905 pg m⁻³, run with 0..360 longitudes and the
  density applied afterwards at the source). The 0.8% rise in the means is
  the per-particle density (decision 0022) against one density at the source.
- A second run with the same seeds gave identical particle trajectories
  (`endpoint_particles.parquet`) and an identical scorecard, but footprints
  that differ in the last bit (at most 5 × 10⁻⁷ relative, in under 1% of cells):
  CUDA's `scatter_add_` accumulates particles in a nondeterministic order,
  which is invisible when every particle adds the same amount (`none`) and
  shows once each adds its own density weight (`dry_air`). Two runs with
  `--set output_grid.density_weighting=none` were bit-identical.
- On the ambient basis (`run --basis ambient --set
  output_grid.density_weighting=none`) the means are 759 and 757 pg m⁻³ (17%
  lower) and the ranks 1.549 and 2.122: the ratio of the dry-air density to
  the 1.293 kg m⁻³ behind DATEM's conversion.
- Before the merge, with the meander coefficient doubled (`--set
  turbulence.meander.coefficient=0.32`), `compare` shows the ranks moving
  by +0.010 (PMCH) and −0.010 (PDCH), with R² down 0.006 in both: changes
  of the size the published studies call not meaningful. (That run's
  scorecard flags the tree as modified, because a doc file was being edited
  while it ran; the check now looks only at `src` and the scoring package.)
- The PMCH and PDCH scores differ mostly in KS and FMS, although the same
  footprints predict both: 56% of DATEM's PMCH samples are non-zero (185 of
  them below 10 pg m⁻³, which looks like background), against 17% for PDCH.
  GLIDE predicts tracer in 8% of the samples. Not investigated further.
