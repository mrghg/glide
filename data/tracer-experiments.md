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
| OKC80, Jul 1980 | optional | PMCH, PDCH | One 3-hour release; 3-hour samples at 100 km and 600 km |
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
  numbers (`8002`) rather than letters (`A02`);
- the JRC quality flags are dropped, and so are the samples marked `-.88`
  (detected but not quantifiable);
- JRC has 3,104 valid samples, 939 of them non-zero; DATEM has 3,105, 941 of
  them non-zero. The small difference suggests a different data version.

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
| ANATEX | 1987-01-05 to 1987-03-29 | most of the US and southern Canada, including Glasgow MT |
| ACURATE (v1.x) | 1982-03-01 to 1983-09-30 | south-eastern US, from Georgia to New Jersey |

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

- the percentile below which measurements count as zero;
- whether zero–zero pairs are included;
- unaveraged ("global") versus time- or space-averaged analysis.

For scale, HYSPLIT's ranks from two sources are below. They differ, and
are not meant to agree: the DATEM description's Appendix D (2001, revised
2002) reports HYSPLIT with NCEP reanalysis, while the posted `mdl_stat` files
come from a later HYSPLIT run, `007`, whose configuration the archive does not
document. **The posted files are the ones the B10 scoring code must
reproduce.** Neither is a threshold for GLIDE; the thresholds come from the
published model comparisons below (validation-plan §3).

| Experiment | Appendix D, all pairs | Appendix D, time-averaged | `statA?_000`, all pairs | `statT?_000`, time-averaged |
| --- | --- | --- | --- | --- |
| CAPTEX | 1.36 | 3.35 | 2.23 | 3.03 |
| ANATEX tracer 1 | 1.69 | 2.69 | 2.03 | 3.47 |
| ANATEX tracer 2 | 1.74 | 2.92 | 1.73 | 2.48 |
| ETEX-1 | — | — | 2.34 | 2.74 |
| ACURATE | — | — | 1.96 | 3.44 |
| OKC80 | — | — | 1.94 | 2.51 |

Published comparisons on the same score: NAME driven by ECMWF reanalysis and by
WRF (Selvaratnam, Thomson and Webster, 2023, J. Appl. Meteor. Climatol. 62(9),
doi:10.1175/JAMC-D-23-0021.1); HYSPLIT, STILT and FLEXPART driven by WRF
(Hegarty et al., 2013, cited there). All of these are **forward** runs.

For ETEX, also compute the ATMES-II statistics (Mosca et al., 1998) so results
compare with the original exercise.

---

## 5. Processing plan

Implemented by queue tasks B10 (readers and scoring) and A11 (model runs). Update
this section with what was actually built.

1. **Readers** for both formats into one table of releases (start, duration,
   location, height, mass) and one of samples (start, duration, location,
   height, concentration, site, quality flag), with units converted to SI.
   The ETEX reader is checked by matching JRC's ETEX-1 against DATEM's,
   sample by sample, and explaining every difference. The ACURATE reader reads
   both `meas-t0.txt` and `meas-t1.txt`, and keeps track of which background
   each uses.
2. **Scoring** in Python: the DATEM statistics and ranking score, and the ATMES-II
   statistics for ETEX. **Verify** against DATEM's own code: run on DATEM's
   HYSPLIT predictions (`mdl_data/<exp>/ct?_000.txt`) with the options in each
   `run_stat.sh`, and reproduce `statA?_000.txt` and `statT?_000.txt` to their
   printed precision, for CAPTEX, both ANATEX tracers, ETEX-1 and ACURATE.
3. **Receptor-oriented backward runs**: one backward release per measured
   sample, particles released uniformly over the sample period at sampler
   height. The predicted concentration is the release's emission rate times the
   sensitivity in a small volume around the source, summed over emission
   intervals. Both ANATEX tracers are sampled at the same sites and times, so one
   backward footprint per sample scores both source locations. ACURATE (v1.x)
   is run the same way, with each sample's sensitivity multiplied by the hourly
   emissions; it is scored on both backgrounds.
4. **Expected differences**, declared in the pre-registration: GLIDE runs
   backward and the published scores are forward; ERA5 versus the comparison
   models' meteorology. The backward-against-forward agreement is also a
   real-case test of the backward formulation.
