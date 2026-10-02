# Tracer-experiment data: ETEX, CAPTEX and ANATEX

_Where the validation tracer data comes from, how to get it, what the files
contain, and how GLIDE will use them. Written 2026-10-02 after checking both
archives by hand; re-check the access notes if a download fails._

These are the observations for validation Tier 3
([docs/validation-plan.md](../docs/validation-plan.md) §2). Like everything in
`data/`, the files are **not committed to this repository**. They live outside it,
located by environment variables:

| Variable | Contents |
| --- | --- |
| `GLIDE_ETEX` | the ETEX files below, in `ETEX_release1/` and `ETEX_release2/` |
| `GLIDE_DATEM` | the DATEM files below, keeping DATEM's own directory names (`exp_data/captex/`, …) |

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
| CAPTEX, Sep–Oct 1983 | **yes** | t1 PMCH | Six 3-hour releases, four from Dayton OH and two from Sudbury ON; 84 sites at 300–800 km; 3- and 6-hour samples for about 48 h after each release |
| ANATEX, Jan–Mar 1987 | **yes** | t1 PTCH from Glasgow MT; t2 PDCH from St Cloud MN; t3 PMCH from St Cloud | 66 releases, every 2.5 days from each site; 75 sites over the eastern US and SE Canada; 24-hour samples. **Do not score t3**: its releases coincide with t2, so it is not independent |
| OKC80, Jul 1980 | optional | PMCH, PDCH | One 3-hour release; 3-hour samples at 100 km and 600 km |
| ACURATE, INEL74 | no | Kr-85 | Continuous sources, few sites |
| Project Sagebrush | no | — | Sub-kilometre scale, beyond a reanalysis-driven model |

### What to download

| DATEM directory | Files | Why |
| --- | --- | --- |
| `exp_data/captex/`, `exp_data/anatex/` | `emit-t?.txt`, `meas-t?.txt` | emissions and measurements |
| `document/` | original experiment reports | release heights, sampler details |
| `stat_pgm/` | `statmain` and `c2datem` source | reference implementation of the scoring |
| `mdl_data/captex/`, `mdl_data/anatex/` | `ct?_001.txt` | HYSPLIT predictions, to verify our scoring code |
| `mdl_stat/captex/`, `mdl_stat/anatex/` | `gbl?_001.txt`, `avg?_001.txt` | HYSPLIT statistics our code must reproduce |

The `met_data/` files (NCEP/NCAR reanalysis on sigma levels) are not needed.

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

GLIDE uses ERA5 for all three experiments, so it runs through the normal
pipeline. Each backward run must reach back to the start of the release it is
scored against.

| Experiment | Period to cover | Domain |
| --- | --- | --- |
| ETEX-1 | 1994-10-23 to 1994-10-27 | Europe |
| ETEX-2 | 1994-11-14 to 1994-11-18 | Europe |
| CAPTEX | 1983-09-18 to 1983-10-30 | eastern US and SE Canada, including Dayton and Sudbury |
| ANATEX | 1987-01-05 to 1987-03-29 | most of the US and southern Canada, including Glasgow MT |

ANATEX covers about 2,000 hours over most of a continent, so it is the largest
cube. Size it before downloading; the compressed met cache (queue task B1) helps.
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

For scale, HYSPLIT with NCEP reanalysis scored, per the DATEM description
Appendix D:

| Experiment | Rank, all pairs | Rank, time-averaged |
| --- | --- | --- |
| CAPTEX | 1.36 | 3.35 |
| ANATEX tracer 1 | 1.69 | 2.69 |
| ANATEX tracer 2 | 1.74 | 2.92 |

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
2. **Scoring** in Python: the DATEM statistics and ranking score, and the ATMES-II
   statistics for ETEX. **Verify** against DATEM's own code: run on DATEM's
   HYSPLIT predictions in `mdl_data` and reproduce the `mdl_stat` files to their
   printed precision.
3. **Receptor-oriented backward runs**: one backward release per measured
   sample, particles released uniformly over the sample period at sampler
   height. The predicted concentration is the release's emission rate times the
   sensitivity in a small volume around the source, summed over emission
   intervals. Both ANATEX tracers are sampled at the same sites and times, so one
   backward footprint per sample scores both source locations.
4. **Expected differences**, declared in the pre-registration: GLIDE runs
   backward and the published scores are forward; ERA5 versus the comparison
   models' meteorology. The backward-against-forward agreement is also a
   real-case test of the backward formulation.
