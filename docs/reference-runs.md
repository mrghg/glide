# Reference runs: specification for FLEXPART and NAME

This is the exact definition of the FLEXPART and NAME runs that GLIDE is
validated against ([validation-plan.md](validation-plan.md), Tier 2). It is
written for whoever sets up and runs those two models. It needs no knowledge
of GLIDE.

**The aim is that the two sets of runs are consistent with each other.** They
are built for model comparison, not to match observations. The receptor
positions and heights below are definitions: use them as written, even where
the real site or inlet differs.

Three rules apply throughout:

- **Use the numbers here exactly.** If a model cannot do something as
  written, stop and report it before running. Do not substitute a near
  equivalent.
- **Keep every setting the same in all runs of one model.** Only the
  receptor, the times and the random seed change between runs. The only
  exceptions are the named sensitivity runs of §9, which each change the
  one setting stated there.
- **Settings that both models have are set to the same value** (§7), even
  where that differs from how either model is usually run.
- **Do not tune.** Do not adjust a setting after seeing output from the other
  model, from GLIDE or from measurements.

Where this file and the validation plan differ, this file is correct for the
reference runs, and the plan is brought into line.

## 1. Summary

| | FLEXPART | NAME |
| --- | --- | --- |
| Meteorology | ERA5, hourly, 0.25°, all 137 model levels, prepared by flex_extract | Met Office global Unified Model analyses, Mk10 series, at the archive's own resolution and interval |
| Direction | Backward in time | Backward in time |
| Tracer | Inert: no deposition, decay or chemistry | The same |
| Release | One hour long, from a point | The same |
| Particles | 21,600 per point release; 100,800 per column release (§4) | The same, to within a few particles |
| Time followed | Each particle for 240 hours from its own release time, to within one 300 s step | The same |
| Main time step | 300 s, for synchronising particles and for sampling the footprint | The same |
| Deep convection | On | On |
| Boundary-layer depth limits | 100 to 4,500 m | The same |
| Unresolved mesoscale motions | Off, as its developers recommend (§7) | On, NAME's standard scheme |
| Particle domain | The EUR or NAM box (§2); particles are removed when they leave it | The same |
| Footprint layer | 0–40 m above the model's ground | The same |
| Output grid | 0.25°, cells centred on ERA5 grid points (§2) | The same grid, so nothing is regridded |
| Output | One footprint per release, summed over the 240 hours, in (mol/mol)/(mol m⁻² s⁻¹) | The same |

## 2. Domains and output grids

There are two domains. Each has one box that is used for three things: the
meteorology given to FLEXPART, the region NAME follows particles in, and the
edge at which particles are removed.

| Domain | Particle domain | Output cells | Output grid edges | First and last cell centres |
| --- | --- | --- | --- | --- |
| **EUR** | 30° W – 40° E, 32 – 72° N | 279 × 159 | 29.875° W – 39.875° E, 32.125 – 71.875° N | 29.75° W … 39.75° E; 32.25 … 71.75° N |
| **NAM** | 120 – 55° W, 22 – 58° N | 259 × 143 | 119.875 – 55.125° W, 22.125 – 57.875° N | 119.75 … 55.25° W; 22.25 … 57.75° N |

- **Cells are 0.25° × 0.25° on a regular latitude–longitude grid,** each
  centred on an ERA5 grid point.
- **The output grid stops half a cell inside the particle domain.** FLEXPART
  refuses an output grid that extends beyond its meteorology
  (`readoptions_mod.f90`, "PART OF OUTPUT GRID IS OUTSIDE MODEL DOMAIN"), and
  a cell that straddled the edge would be half empty. The strip 0.125° wide
  inside the edge is not counted in any model.
- **A particle is removed as soon as it leaves the particle domain,** and
  never comes back. FLEXPART does this by itself when its meteorology covers
  only the box. NAME reads global fields, so its computational domain must
  be set to the box.
- **The top.** No particle may be removed below 20 km above the ground.
  The two models differ above that, and the difference is declared:
  - NAME's computational domain has its top at 20,000 m above the ground. A
    particle that rises above it is removed, and its end record (§8) says
    it left through the top.
  - FLEXPART uses all 137 levels, to about 80 km. It does not remove a
    particle at its top; it holds it just below the top level
    (`advance_mod.f90`, `set_z(ipart,height(nz)-100.*eps)`).

  Report where each model's top is and what happens to a particle that
  reaches it.

In FLEXPART's terms:

| | EUR | NAM |
| --- | --- | --- |
| flex_extract area (N/W/S/E), `GRID = 0.25` | 72 / −30 / 32 / 40 | 58 / −120 / 22 / −55 |
| Meteorology grid points | 281 × 161 | 261 × 145 |
| `OUTLON0`, `OUTLAT0` (lower-left corner) | −29.875, 32.125 | −119.875, 22.125 |
| `NUMXGRID`, `NUMYGRID` (number of cells) | 279, 159 | 259, 143 |
| `DXOUT`, `DYOUT` | 0.25, 0.25 | 0.25, 0.25 |
| `OUTHEIGHTS` | 40.0 | 40.0 |

More output levels may be added above 40 m. Only the 0–40 m layer is used.

## 3. Receptors

Release from exactly these coordinates. Heights are **metres above the
model's own ground** at that point, the same number in every model. No model
converts a height above sea level.

| Code | Site | Latitude | Longitude | Release height | Domain |
| --- | --- | --- | --- | --- | --- |
| MHD | Mace Head | 53.3267° N | 9.9046° W | 10 m | EUR |
| JFJ | Jungfraujoch | 46.5475° N | 7.9851° E | 1,000 m | EUR |
| CBW-20 | Cabauw, low | 51.9703° N | 4.9264° E | 20 m | EUR |
| CBW-200 | Cabauw, high | 51.9703° N | 4.9264° E | 200 m | EUR |
| TAC | Tacolneston | 52.5176° N | 1.1387° E | 185 m | EUR |
| LEF | Park Falls | 45.9451° N | 90.2732° W | 30 m | NAM |
| SGP | Lamont | 36.6040° N | 97.4860° W | column (§5) | NAM |

- **The release is a point:** no width in the horizontal and no depth in the
  vertical (in FLEXPART, `LON1 = LON2`, `LAT1 = LAT2`, `Z1 = Z2`,
  `ZKIND = 1`).
- **Jungfraujoch is 1,000 m above the model's ground,** not 3,580 m above
  sea level. Both models smooth the Alps, by different amounts, so a height
  above sea level would put the release at a different height above each
  model's surface.
- **Report what the model sees at each receptor:** the height of its ground
  above sea level there, and the land or sea type of that grid cell. These
  go in the run record (§8) and explain differences later.

## 4. Release times and how long particles are followed

All times are UTC.

| Block | Domain | First release starts | Last release starts | Releases per receptor | Run must cover |
| --- | --- | --- | --- | --- | --- |
| EUR-Jan | EUR | 1 Jan 2019 00:00 | 14 Jan 2019 23:00 | 336 | 22 Dec 2018 00:00 – 15 Jan 2019 00:00 |
| EUR-Jul | EUR | 1 Jul 2019 00:00 | 14 Jul 2019 23:00 | 336 | 21 Jun 2019 00:00 – 15 Jul 2019 00:00 |
| NAM-Jan | NAM | 1 Jan 2019 00:00 | 14 Jan 2019 23:00 | 336 | 22 Dec 2018 00:00 – 15 Jan 2019 00:00 |
| NAM-Jul | NAM | 1 Jul 2019 00:00 | 14 Jul 2019 23:00 | 336 | 21 Jun 2019 00:00 – 15 Jul 2019 00:00 |

- **One release every hour, on the hour, from every receptor in the block's
  domain.** Each release lasts one hour: release *t* covers *t* to
  *t* + 1 h. Particles are released evenly through that hour.
- **A release is named by its start time.** The footprint for "8 Jan 12:00"
  comes from particles released between 12:00 and 13:00.
- **Each particle is followed for 240 hours from the moment it was
  released,** then stopped. It is not followed to a fixed end time. Both
  models check a particle's age once per 300 s step (§7), so a particle may
  be followed for up to 300 s longer than 240 hours, and no more.
  - FLEXPART: `LAGESPECTRA = 1`, and an `AGECLASSES` file with
    `NAGECLASS = 1` and `LAGE = 864000`. FLEXPART then stops each particle
    at that age (`timemanager_mod.f90`, near line 639). With
    `LAGESPECTRA = 0` particles run to the end of the simulation, which is
    wrong here.
  - NAME: maximum travel time 240 hours.
- **Particles per release: 21,600 for a point release and 100,800 for a
  column release.** These are 6 and 28 particles per second for one hour.
  NAME takes a release rate, not a total, and may release a few more or
  fewer than rate × 3,600. The number each model actually released is read
  from its output, stored in `n_particles` (§8) and used as $N$ in the
  footprint (§6).
- **That is 4,704 releases:** five receptors × 336 × two blocks in EUR
  (3,360), and two receptors × 336 × two blocks in NAM (1,344).

How the releases are divided into runs (by receptor, by day) is up to the
runner, as long as every setting is identical.

## 5. The column release at Lamont

Every Lamont release is a column, not a point.

- **Particles start at pressures drawn evenly between the model's surface
  pressure and 100 hPa** (about 16 km), at the Lamont coordinates. Even in
  pressure means even in air mass, which is what a column-average
  measurement weights by.
- **The surface pressure is the model's own,** at the Lamont coordinates
  (interpolated from the four surrounding grid points) at the release's
  start time. List the value used for every release in the run record.
- **100,800 particles per release** (28 per second).
- **FLEXPART:** an ordinary `RELEASES` entry with `ZKIND = 3`,
  `Z1` = the surface pressure in hPa and `Z2 = 100`. If `Z1` is higher than
  the true surface pressure, FLEXPART puts the extra particles in its lowest
  level, so `Z1` must be the model's value, not a round number. The
  `SATELLITES` option is not used.
- **NAME:** if NAME cannot draw starting heights evenly in pressure, use 28
  stacked sources. Each spans one twenty-eighth of the pressure range from
  the surface pressure to 100 hPa, releases 1 particle per second (3,600
  particles), and is even in height within its own layer. Record which method was used and the
  layer boundaries in metres.

## 6. What the footprint is

The footprint of a release is the rise in mole fraction at the receptor,
averaged over the release hour, for a unit emission from the surface of a
grid cell that is constant through the 240 hours. Its unit is
(mol/mol)/(mol m⁻² s⁻¹), which is m² s mol⁻¹.

In terms of particles, for one output cell:

$$
S = \frac{M_\mathrm{air}}{h\ N} \sum_{p=1}^{N} \int \frac{I_p(t)}{\rho_p(t)}\ \mathrm{d}t
$$

- $N$ is the number of particles the model actually released (§4);
- $I_p(t)$ is 1 while particle $p$ is inside the cell horizontally **and**
  between 0 and 40 m above the model's ground, and 0 otherwise;
- $\rho_p(t)$ is the model's air density at the particle (kg m⁻³);
- $h$ = 40 m;
- $M_\mathrm{air}$ = 0.02897 kg mol⁻¹;
- the integral runs over the particle's 240 hours.

Points to hold to:

- **The layer is a fixed 40 m above the ground.** It is not the boundary
  layer, and it does not change with the boundary-layer depth.
- **Each particle counts only in the cell that contains it.** No kernel or
  smoothing spreads it over neighbouring cells. FLEXPART is built with
  `lusekerneloutput = .false.` (`src/par_mod.f90`). NAME uses no output
  smoothing.
- **The footprint is summed over the whole 240 hours.** One two-dimensional
  field per release is delivered. Footprints resolved in time are not
  needed.
- **Particles that leave the domain stop contributing.**

FLEXPART's native backward output with `IND_SOURCE = 1` and
`IND_RECEPTOR = 2` is in s m³ kg⁻¹. Summed over all output times and
multiplied by $M_\mathrm{air} / h$, it gives $S$. NAME's output is converted
with the group's usual conversion, with the 40 m layer and the same
$M_\mathrm{air}$. Each runner records the conversion used, and keeps the
native output.

## 7. Model settings

### FLEXPART

Version 11, one commit for all runs, with the default terrain-following
("eta") build, on flex_extract's ERA5.

| Setting | Value |
| --- | --- |
| `LDIRECT` | −1 |
| `IBDATE`/`IBTIME`, `IEDATE`/`IETIME` | the "run must cover" span of the block (§4) |
| `LOUTSTEP`, `LOUTAVER` | 3600, 3600 |
| `LSYNCTIME`, `LOUTSAMPLE` | 300, 300 |
| `CTL`, `IFINE` | 10, 4 |
| `IOUT`, `LNETCDFOUT` | 1, 1 |
| `IPOUT` | 0 (1 for the single-release runs, §9) |
| `LSUBGRID` | 0 |
| `LCONVECTION` | 1 |
| `LTURBULENCE` | 1 |
| `LTURBULENCE_MESO` | 0 |
| `LAGESPECTRA` | 1, with `NAGECLASS = 1`, `LAGE = 864000` |
| `IOUTPUTFOREACHRELEASE` | 1 |
| `IND_SOURCE`, `IND_RECEPTOR` | 1, 2 |
| `CBLFLAG` | 0 |
| `LINIT_COND` | 0 |
| Species | a passive air tracer with no wet or dry deposition, no decay and no OH reaction |
| `PARTS` | 21,600 (100,800 for Lamont) |
| `hmixmin`, `hmixmax` (`par_mod.f90`) | 100, 4500: the defaults, not to be edited |
| `lusekerneloutput` (`par_mod.f90`) | `.false.` |
| Random seeds | recorded for every run (§9) |

`LTURBULENCE_MESO = 0` is FLEXPART's default and its developers'
recommendation (the option's comment in `COMMAND` reads "off
(recommended)"). The reference runs keep it. Its effect is measured by the
`meso` sensitivity run (§9).

Any option not listed stays at the v11 default. The time-step settings
(`LSYNCTIME`, `CTL`, `IFINE`) are the one choice here that the FLEXPART
runner may challenge. Do so before the first run; they are then fixed.

**Never give FLEXPART the vertical velocity from the ARCO ERA5 copy.** That
field is ω, and FLEXPART would read it as η̇ ∂p/∂η. Use flex_extract's
fields only.

### NAME

The settings below are chosen so that NAME and FLEXPART share every setting
they both have. Several differ from the group's site template (deep
convection, the sync time, the boundary-layer limits, the random seed and
the particle cap). That is intended; the `group` run (§9) measures the
effect.

Record the version. If a patched build is used (for example, to add the
particle index and stop reason to the particle-fate output), name the patch
in the run record, and show with one `Fixed`-seed run that the patched and
unpatched builds give the same footprints, to within rounding (§10).

| Setting | Value |
| --- | --- |
| Meteorology | Met Office global analyses, Mk10 series, for the "run must cover" span (§4). One series version per block |
| Computational domain | the particle domain of §2, top at 20 km or higher |
| Source | a point at the §3 coordinates and height above ground; one hour long; 6 particles per second (28 for Lamont), which is 21,600 (100,800) per release |
| Sync time | 300 s. This is NAME's main time step: particles are released through it, the footprint is sampled at it, and the 240-hour limit is checked at it |
| Max # Particles | at least 6,000,000 for a block run of one point receptor, and 25,000,000 for Lamont (see below) |
| Maximum travel time | 240 hours |
| Output grid | §2; one layer, 0–40 m above ground |
| Output quantity | air concentration integrated over the 240 hours, one field per release |
| Turbulence | on. Record the scheme and its settings |
| Unresolved mesoscale motions | on, with NAME's standard settings for these fields: since NAME 8 its velocity variances and timescales come with the met definition of each series, so the input sets none. Record the values used |
| Deep convection | `New`, NAME's mass-flux scheme (Meneguz and Thomson, 2014), which NAME's user guide recommends for global met. NAME's other choices are `No` and `Old`; the group's site template uses `No` (measured by the `group` run, §9) |
| Boundary-layer depth | from the meteorology, limited to 100 to 4,500 m (`Min B L Depth`, `Max B L Depth`) |
| Deposition, decay, chemistry | off |
| Random seed | `Fixed` for `main` and the single-release runs; `Random` for the repeat-seed runs (§9) |

**The particle cap.** `Max # Particles` limits the number of particles alive
at once. In a block run of one receptor, up to 240 hourly releases are alive
together: 240 × 21,600 = 5,184,000 particles for a point receptor, and
240 × 100,800 = 24,192,000 for Lamont. If NAME reaches the cap it stops
releasing, prints a warning and carries on, so the output looks normal.
- Set the cap above these numbers.
- If the memory needed is not available, divide the releases between more
  runs (§4). Do not lower the number of particles per release.
- Confirm from the output that every release has its full number of
  particles (§10), and state the cap in the run record.

## 8. What to deliver

### Footprint files

One NetCDF file per receptor, block and run, in the layout of the group's
existing footprint files:

- `srr(time, latitude, longitude)`, float32, in (mol/mol)/(mol/m2/s);
- `time` is the release start, one entry per release;
- `latitude` and `longitude` are the cell centres of §2;
- `release_lon(time)`, `release_lat(time)`, and `release_height(time)` in
  metres above the model's ground (for Lamont, `release_pressure_bottom`
  and `release_pressure_top` in hPa);
- `n_particles(time)`: the number of particles the model actually released,
  counted from its output;
- `fraction_left_sides(time)` and `fraction_left_top(time)`: the fraction of
  the release's particles removed through the side edges and through the
  top before reaching 240 hours. If a model cannot give these for the block
  runs, say so; they are then taken from the single-release runs (§9). In
  FLEXPART `fraction_left_top` is zero, because FLEXPART does not remove
  particles at its top (§2).

### End records

An end record says where each particle was when the model stopped following
it. It is required for the seven single-release runs (§9), and for the
block runs from any model that can write it per release. NAME can.

Exactly one row per particle, so the number of rows for a release equals
its `n_particles`. If the model's raw output repeats a particle, remove the
repeats and record how in the run record. Each row has:

- the longitude, latitude and height above the model's ground at the last
  time the particle was followed;
- the particle's age at that time, in hours;
- why it stopped: it left through a side edge (and which: north, east,
  south or west), it left through the top, or it reached 240 hours inside
  the domain. If the model gives more than one reason for a particle, record
  them all.

How each model gets it:

- **FLEXPART** does not record where a particle is removed. Write particle
  positions every hour (`IPOUT = 1`) in the single-release runs, and take
  each particle's last recorded position. That is within one hour of where
  it left. Say in the run record whether the particle index stays the same
  from one output time to the next; if it does not, report that before
  running.
- **NAME** takes the record from its particle-fate output, for the
  single-release runs and all the block runs.

File name: as the footprint file, with `_end` before `.nc`.

Footprint file name:
`{receptor}_{model}_{meteorology}_{domain}_{YYYYMM}_{run}.nc`, for example
`CBW-20_FLEXPART_ERA5_EUR_201901_main.nc` or
`MHD_NAME_UMG-Mk10_EUR_201901_seed3.nc`. `run` is `main`, `seed1` to
`seed5`, `kernel`, `meso`, `nomeso` or `group` (§9).

### Run record

One text file (YAML) per model, with:

- model version and commit; compiler, flags and edited constants (FLEXPART);
  any patch to the model, and what it changes (NAME);
- the complete input files of one run (FLEXPART's `COMMAND`, `OUTGRID`,
  `AGECLASSES`, `SPECIES` and one `RELEASES` entry; NAME's input file);
- the meteorology: source, version (flex_extract version and `CONTROL`
  file; Met Office series number), resolution, levels, interval between
  fields;
- for each receptor: the model's ground height above sea level, and its
  surface type if the model can report it;
- the main time step, the boundary-layer depth limits, and whether deep
  convection and unresolved mesoscale motions are on, with their settings;
- NAME's particle cap, and confirmation that it was never reached;
- for each Lamont release: the surface pressure used;
- where the model's top is, and what happens to particles there;
- the unit conversion applied to the native output;
- the random seeds of every run. NAME does not report a seed, so for NAME
  record the `Random Seed` setting (`Fixed` or `Random`);
- wall-clock time, threads or cores, and hardware of every run;
- every departure from this file, with the reason, written down before the
  run.

The files go to the locations named in
[dev/agent/PROGRESS.md](../dev/agent/PROGRESS.md) (`GLIDE_FLEXPART_OUTPUTS`
and `GLIDE_NAME_FOOTPRINTS`). NAME output and Met Office fields are
restricted and never go into the repository. Nor does NAME's run record:
NAME's input file contains the NAME team's met declarations, which are
licensed with NAME. It is kept with the NAME output.

## 9. The runs

| Run | Releases | Models | Purpose |
| --- | --- | --- | --- |
| `main` | All 4,704 (§4) | Both | The reference footprints |
| `seed1` … `seed5` | 1 Jan 2019 00:00 – 5 Jan 23:00 (120 releases) at MHD, JFJ and CBW-20 | Both | How much footprints differ from random sampling alone |
| Single-release runs | The seven standard cases below, one release per run | Both (NAME: if it can write particle positions) | Particle positions |
| `kernel` | Standard case 1 only | FLEXPART | The effect of FLEXPART's default output kernel (`lusekerneloutput = .true.`) |
| `meso` | The seven standard cases, one release per run | FLEXPART | The effect of FLEXPART's mesoscale term: `LTURBULENCE_MESO = 1`, all else as `main` |
| `nomeso` | The seven standard cases, one release per run | NAME | The effect of NAME's unresolved mesoscale motions: switched off, all else as `main` |
| `group` | MHD, block EUR-Jan, all 336 releases | NAME | How far the shared settings moved NAME: deep convection off, sync time 3 minutes and boundary-layer limits 40 to 4,000 m, as in the group's site template; all else as `main` |

**Repeat-seed runs.** Five separate runs with five different seeds, all
other settings as `main`.
- FLEXPART: change `idummy` (`FLEXPART.f90`) and `iseed1` and `iseed2`
  (`random_mod.f90`) for each run, one rebuild per seed. Use the same
  thread count as `main`.
- NAME: `Random Seed` takes only `Fixed` or `Random`, and NAME does not
  report the seed it used. Use `Random` for all five runs. They cannot be
  repeated, so keep all their output. Start them at different times, and
  confirm that the five differ from one another.

**NAME's seed for `main`.** Use `Fixed`, so the run can be repeated, unless
check 6 of §10 shows that two identical sources in one `Fixed` run give
identical output. In that case the hourly releases of a `Fixed` run share
random numbers, so use `Random` for `main` as well, keep all its output, and
say so in the run record before running.

**The sensitivity runs** (`kernel`, `meso`, `nomeso`, `group`) are reported
and never graded. Each changes only the setting in the table. The seven
cases of `meso` and `nomeso` deliver the same files as the single-release
runs.

**Single-release runs.** The same settings as `main`, with one release per
run, because FLEXPART's particle files do not say which release a particle
came from. Write every particle's position at 6, 24, 72 and 120 hours before
the release's start time: longitude, latitude, height above the model's
ground, and whether it is still in the domain. Also deliver the end record
of every particle (§8).

| # | Case | Receptor | Release start (UTC) |
| --- | --- | --- | --- |
| 1 | Coastal | MHD | 8 Jan 2019 12:00 |
| 2 | Mountain | JFJ | 8 Jul 2019 12:00 |
| 3 | Stable night | CBW-20 | a night in EUR-Jan, 02:00; supplied |
| 4 | Convective afternoon | LEF | a day in NAM-Jul, 20:00; supplied |
| 5 | Frontal passage | TAC | an hour in EUR-Jan; supplied |
| 6 | Continental winter | LEF | 8 Jan 2019 18:00 |
| 7 | Column | SGP | 8 Jul 2019 19:00 |

The times for cases 3 to 5 are picked from ERA5 by fixed rules
([validation-plan.md](validation-plan.md), §4) and supplied before the
single-release runs. They are all releases that `main` already contains, so
`main` does not wait for them.

## 10. Checks before the full runs

Run one day (8 January 2019, 24 releases at MHD) in each model and confirm:

1. The output grid has the cell count and centres of §2.
2. No particle is older than 240 hours plus one 300 s step, and particles
   released at 00:30 are still being followed at 239 hours.
3. The footprint summed over all cells has the expected size. In the
   group's earlier Mace Head files it is mostly 5 to 17 m² s mol⁻¹ for a
   10 m release. Compare the two models' totals; a difference of a factor
   of ten or more means a unit or layer error.
4. The receptor's ground height is in the run record, and its surface type
   if the model can report it.
5. Two runs with different seeds differ, and two with the same seed do not
   (FLEXPART: single thread). For NAME: two `Fixed` runs are identical, and
   a `Fixed` and a `Random` run differ.
6. NAME only: put two identical sources (same position, height and time) in
   one `Fixed` run, and report whether their footprints are identical. This
   decides the seed setting of `main` (§9).
7. NAME only: `n_particles` of every release is within a few particles of
   21,600, and the end records have exactly one row per particle.
8. NAME only, if a patched build is used: one `Fixed` run with the patched
   build and one with the unpatched build give identical particle-fate
   records, and footprints that differ in any cell by at most one unit in
   the last digit NAME prints (about one part in a million; two separately
   built executables may round that digit differently).

Send the run record and the one-day files for checking before starting the
blocks.
