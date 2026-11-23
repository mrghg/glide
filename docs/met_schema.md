# Meteorology input contract

Everything a meteorology store must satisfy to be read by GLIDE
(`src/lpdm/met_reader.py`). ARCO ERA5 is the reference source, but **any** store
conforming to this page will run — the point of writing it down is that other
reanalyses, NWP output and regional models can be used.

This page is the authoritative definition of the `glide` output format of
[met-bazarr](https://github.com/mrghg/met-bazarr), the separate repository that
downloads and prepares meteorology for GLIDE
([dev/decisions/0018](../dev/decisions/0018-met-bazarr.md)). A change here is a
change to that format: note it in the decision records and tell met-bazarr.

**Format version: 2.0** (see [Format versions](#format-versions)). Stores record the version they implement in
`glide_format_version` (see [Provenance attributes](#provenance-attributes)).
The history is under [Format versions](#format-versions).

**What is prescribed:** container format, coordinate conventions, variable names,
dimensions, units, sign conventions, a single grid shared by all fields, and the
provenance attributes. Get these exact. Stores used as validation inputs
satisfy [Validation inputs](#validation-inputs) as well.

**What is not:** horizontal resolution, vertical resolution, domain extent
(beyond covering the run's domain, see [Domain and extent](#domain-and-extent)),
time span, chunk sizes. GLIDE adapts. In particular it runs on **either** pressure
levels **or** model (hybrid) levels — see [Vertical coordinate](#vertical-coordinate).

`scripts/download_sample_cube.py` produces conforming stores from ARCO ERA5 and
is the best worked example to imitate. If your source is not ERA5, skip to
[Preparing meteorology from a non-ERA5 source](#preparing-meteorology-from-a-non-era5-source)
for the gaps you will most likely have to close yourself.

---

## Container format

- **Zarr**, with **consolidated metadata** (`to_zarr(..., consolidated=True)`).
  Zarr v2 and v3 are both fine.
- **One logical dataset per store.** A long time series may be split across
  several stores and handed to GLIDE as a list (or a local glob); they are
  stitched along `time`. **All stores in a set must share byte-identical
  `latitude`, `longitude` and vertical coordinates** — only `time` may differ.
- **dtype** `float32` or `float64`; GLIDE casts to `float32` on read. CF
  `scale_factor`/`add_offset` into `int16` also conforms — xarray decodes it
  before GLIDE sees anything — and roughly halves a store; see
  [Storage precision](#storage-precision).
- **No non-finite values** in the required variables over the domain you intend
  to run. Non-finite geopotential is rejected outright, and the downloader's
  validator refuses to install a store containing any.
- **Every required variable must carry a non-empty CF-style `units` attribute.**
  A missing or empty `units` is a hard error — GLIDE will not guess.

## Coordinates

| Coordinate | Default name | Requirement |
| --- | --- | --- |
| Time | `time` | `datetime64`, **UTC**, any *uniform* cadence. Must cover the run's backward window. |
| Latitude | `latitude` | Degrees north. Ascending or descending both accepted. |
| Longitude | `longitude` | Degrees east. Either $-180..180$ or $0..360$; GLIDE detects and normalises. |
| Vertical | `level` | 1-D. Its *meaning* depends on the vertical mode below. |

**One grid for every field.** All variables share one set of `latitude` and
`longitude` points, and all 3-D variables share one vertical coordinate. GLIDE
has no notion of staggering. A source that stores fields at different points
(for example the Met Office UM: winds offset horizontally on an Arakawa C grid,
and potential temperature, humidity and vertical velocity on different levels
from the winds and pressure) is interpolated onto one grid by the converter.
The target grid and the interpolation method are the converter's choice. They
are recorded in the `processing` attribute, and for validation inputs they are
fixed before conversion (see [Validation inputs](#validation-inputs)).

Coordinate names are configurable on the reader (`lon_name`, `lat_name`,
`level_name`, `time_name`), but the defaults above are the path of least
resistance. The one you *will* set is `level_name` for model-level stores — and
even that is auto-corrected when the store is tagged (see below).

**Cadence is measured, not declared.** GLIDE reads the spacing of the time
coordinate and makes one met window exactly one source timestep wide — hourly,
3-hourly, 15-minute, whatever the store holds. The spacing must be **uniform**: a
ragged coordinate would give some windows the wrong duration and mis-weight the
linear interpolation inside them, so GLIDE refuses it and names the gap. The
window grid is anchored on the store's own first timestamp, so a store sampled at
:30 past brackets on :30 rather than on the hour.

Cadence and flux accumulation are independent: the accumulated-flux period comes
from the store's `glide_flux_accumulation_seconds` attr (see
[Sensible heat flux](#sensible-heat-flux)), not from the time coordinate. An
hourly-accumulated ERA5 cube thinned to 3-hourly has a 10800 s cadence and a
3600 s accumulation period.

### What a timestamp means

A field's value at a timestamp is one of two things:

- **A snapshot**, valid at the timestamp. This is the default, and it covers
  the winds, temperature, humidity, pressure and boundary-layer depth of ERA5
  and of the UM.
- **A mean or an accumulation over a period that ends at the timestamp.** For
  example, ERA5's surface sensible heat flux and surface stresses are
  accumulated over the hour ending at their timestamp. The UM's heat flux and
  surface stresses are means over the 3 hours ending at theirs. A friction
  velocity derived from mean stresses is itself a period variable.

Each variable of the second kind carries two attributes of its own:

| Variable attribute | Content |
| --- | --- |
| `cell_methods` | `"time: mean"` or `"time: sum"` (CF convention) |
| `glide_period_seconds` | Length of the period, in seconds; the period ends at the timestamp |

A variable with neither attribute is a snapshot.

**How GLIDE uses them.** Snapshots are interpolated linearly in time across
each meteorology window, as before. A period variable whose period equals the
store's time step is held constant through each window, at its value at the
window's end, which is the mean over exactly that window. This places it at
the middle of the period it represents, as a snapshot is placed at its
timestamp. GLIDE refuses a period variable whose period differs from the time
step (for example ERA5 thinned to 3-hourly), rather than guess. This is how
NAME treats the UM's time-mean fluxes
([decision 0020](../dev/decisions/0020-time-meaning-of-period-fields.md)).

Validation inputs derive `friction_velocity` from the surface stresses, as NAME
and FLEXPART do, rather than taking a source's own friction velocity (ERA5's
`zust`).

**Stores written before format 2.0** carry no variable attributes. GLIDE then
treats `surface_sensible_heat_flux` as a period variable when the store has
`glide_flux_accumulation_seconds` or a non-zero
`glide_flux_averaging_seconds`, with that period, and every other variable as
a snapshot.

Coarser meteorology is read directly, but it is not free of consequence: GLIDE
interpolates the wind linearly across a window, so anything varying inside one —
the diurnal boundary-layer cycle, a frontal passage — is smoothed. The physics is
tuned against hourly ERA5, and a run coarser than that logs a warning at startup.
Two settings that are tied to the source interval and do **not** follow it
automatically:

- `turbulence.meander_timescale_seconds` defaults to 1800 s, half the hourly
  interval. Meander parameterises the mesoscale motion the met does not resolve,
  so on a coarser source consider setting it to half your interval. It is left as
  an explicit knob because changing it moves results.
- `memory.met_cache_max_hours` counts met **windows**, not hours. At 3-hourly
  cadence a value of 48 caches six days of meteorology, and the host RAM it uses
  scales with the window count, not the wall-clock span.

## Required variables

3-D fields are functions of `(time, level, latitude, longitude)`; surface fields
of `(time, latitude, longitude)`. `geopotential_at_surface` may be 2-D or 3-D (a
singleton level is fine). Names are GLIDE's defaults; they can be remapped via
the reader's `variable_map`, but matching them means no remap is needed.

| Variable | Role | Dims | Units | Notes |
| --- | --- | --- | --- | --- |
| `u_component_of_wind` | zonal wind | 3-D | `m s**-1` | |
| `v_component_of_wind` | meridional wind | 3-D | `m s**-1` | |
| `vertical_velocity` | vertical motion | 3-D | `Pa s**-1` **or** `m s**-1` | omega is converted to geometric $w$; see below |
| `temperature` | air temperature | 3-D | `K` | ω→w conversion, stability, density, $\theta$ |
| `geopotential` | geopotential | 3-D | `m**2 s**-2` | **Not geopotential height.** See below. |
| `specific_humidity` | specific humidity | 3-D | `kg kg**-1` | deep convection; also model-level pressure |
| `boundary_layer_height` | BL depth | surface | `m` | |
| `surface_pressure` | surface pressure | surface | `Pa` | |
| `geopotential_at_surface` | orography | surface (2-D or 3-D) | `m**2 s**-2` | the terrain reference for the AGL conversion |
| `friction_velocity` | $u_\ast$ | surface | `m s**-1` | Hanna turbulence |
| `surface_sensible_heat_flux` | sensible heat flux | surface | `W m**-2` **or** `J m**-2` | **sign convention matters** — see below |

A conforming store contains exactly this set plus coordinates; there are no
optional-but-recognised extras.

---

## Sign and unit conventions

These are the easy things to get wrong when converting a non-ERA5 source, and
each fails silently rather than loudly.

### Vertical velocity

`vertical_velocity` may be **omega** (a pressure tendency, `Pa s**-1`) or a
**geometric velocity** (`m s**-1`). If it is omega, GLIDE converts it:

$$
w = -\frac{R_d\  T}{g\  p}\ \omega
$$

which needs the pressure $p$ at each level.

- On **pressure levels** the level coordinate *is* $p$, so omega works directly.
- On **model levels** the coordinate is an index, so GLIDE reconstructs per-level
  pressure hydrostatically (below) and uses that. **Omega works on model levels
  too** — no pre-conversion needed, which matters because ARCO's model-level
  product ships omega. Supplying `m s**-1` directly is also accepted and skips
  the conversion.

### Sensible heat flux

GLIDE's boundary-layer physics uses **positive = upward**, but expects the
**input** in the **ECMWF/ERA5 convention: positive = downward** (into the
surface). GLIDE negates internally.

So: a daytime, upward surface sensible heat flux should be stored **negative**.
If your source is already positive-upward, negate it before writing. Getting this
wrong inverts the stability classification on every field — GLIDE will run
happily and produce nonsense.

Accumulated (`J m**-2`) fluxes are de-accumulated before use; instantaneous
(`W m**-2`) fluxes pass through unscaled.

A `W m**-2` field may be an instantaneous value or a mean over an interval
(the UM writes means over its output interval). Record which it is in the store
attribute `glide_flux_averaging_seconds`: `0` for instantaneous, otherwise the
length of the averaging interval, which ends at the timestamp. How GLIDE places
a mean or an accumulation in time is under
[What a timestamp means](#what-a-timestamp-means).

**Tag the accumulation period.** `J m**-2` says the field is accumulated but not
over how long, and the period is not recoverable from the data. GLIDE therefore
**refuses** to de-accumulate an accumulated flux unless the store says:

```
glide_flux_accumulation_seconds = 3600
```

(`download_sample_cube.py` writes it; the reader's `accumulation_seconds`
argument overrides it for a store whose attr is wrong.) A store whose fluxes are
already instantaneous never needs the attr.

This is the **source's** accumulation period, and it does not change when you
subset or subsample the store in time: ERA5 accumulates over one hour, so an ERA5
cube thinned to 3-hourly still carries `3600`. Getting it wrong is a silent
rescaling of the surface heat flux — which sets the Obukhov length, and so the
stable/convective regime the turbulence scheme picks for every column.

A cube downloaded before this attr existed will be refused. Retagging one is a
metadata-only edit — no data is rewritten:

```python
import zarr

g = zarr.open_group("EUROPE_202401.zarr", mode="r+")
g.attrs["glide_flux_accumulation_seconds"] = 3600  # ERA5
zarr.consolidate_metadata(g.store)
```

As a backstop, GLIDE checks the converted flux on the first window it reads and
rejects a peak magnitude above 2000 W m⁻² (real values peak near 600–800 over hot
dry land; an undivided hourly accumulation lands near 3.6 × 10⁵). That catches a
missing or order-of-magnitude-wrong divisor whatever the attrs claim. It cannot
catch a small wrong factor — reading hourly-accumulated data as 3-hourly puts
midday flux at 300 W m⁻² instead of 100, and both are plausible — which is why
the attr is mandatory rather than defaulted.

### Geopotential, not geopotential height

`geopotential` and `geopotential_at_surface` are true geopotential (m² s⁻²), not
height in metres. GLIDE derives geometric height as

$$
z_{\mathrm{AGL}} = \frac{\Phi - \Phi_s}{g}
$$

Do **not** pre-divide by $g$.

**Sub-surface levels are allowed and expected.** Where a pressure level sits
below the local terrain, the AGL height is legitimately negative. Do not clamp
them to zero or mask them — GLIDE excludes them properly during the
terrain-following resample, and clamping would corrupt that.

---

## Vertical coordinate

Two modes, distinguished only by what the level coordinate *means*. Both require
3-D `geopotential` and `geopotential_at_surface`; height always comes from
geopotential, so both share the same AGL machinery.

### Pressure levels

The level coordinate holds **pressure**. Units come from its `units` attribute —
`hPa` (or `mbar`/`millibar`) or `Pa`. With no units attribute, values ≤ 2000 are
assumed hPa, otherwise Pa. Any ordering and any number of levels.

### Model / hybrid levels (the default, decision 0017)

The level coordinate is a **level index**. Per-level heights come directly from
the 3-D `geopotential` field, exactly as on pressure levels — but pressure has to
be derived, and GLIDE reconstructs it **hydrostatically** from the archive's own
fields by integrating the hypsometric relation upward from the surface:

$$
p(k) = p(k-1)\ \exp\left(-\frac{\Phi(k) - \Phi(k-1)}{R_d\  T_v^{\text{layer}}}\right),
\qquad
T_v = T\ (1 + 0.6077\ q)
$$

with the first segment running from the surface. It is exact for an isothermal
column, and it needs **no hybrid $a/b$ coefficients** — deliberately, because
ARCO does not ship them and a third-party L137 table cannot be guaranteed to
match the data this cube was built from. A mismatch would silently corrupt every
pressure. `specific_humidity` and `surface_pressure` are therefore mandatory on
model levels. See
[dev/decisions/0010](../dev/decisions/0010-model-level-met-reader.md).

Model levels require the terrain-following path (`terrain_following=True`, the
default).

**Tag your model-level cubes.** GLIDE **refuses** to read a store as pressure
levels when the vertical coordinate does not look like pressures — when it is
named `hybrid`/`model_level`, declares a `hybrid_sigma_pressure` `standard_name`,
or holds consecutive integers. Without that guard the indices would be read as
pressures, corrupting the ω→w conversion, air density and convection with no
error. Fix it by setting the store attribute

```
glide_vertical_coordinate = "pressure_level" | "model_level"
```

(`download_sample_cube.py` does this), or by constructing the reader with
`vertical_coordinate="model"`. When tagged, GLIDE also auto-corrects `level_name`
to the store's actual vertical coordinate.

### The internal AGL grid sets the resolution the physics sees

Whichever mode the source uses, GLIDE resamples every meteorology hour onto one
**fixed terrain-following AGL ladder** shared by all columns. (That shared 1-D
ladder is what makes the vertical interpolation cheap enough for the per-step hot
path.) So the *source's* level count does not set the model's effective vertical
resolution — this grid does, and it also sets the meteorology cache size.

The default is 40 levels stretched from `first_layer_m` (decision 0017, task
A01; interim until the convergence study, A5), which roughly matches the
near-surface density of ERA5's model levels (~20 below 1.5 km). Before A01 the
default was a fixed 23-level ladder, 13 below 1.5 km; pin it explicitly with
`vertical_levels: <list>` to reproduce older runs. To set the grid yourself:

```yaml
met_domain:
  alt_max_m: 15000.0
  vertical_levels: 40        # a count → geometrically stretched grid
  first_layer_m: 10.0        # lowest layer thickness
```

`vertical_levels` accepts either a **count** (layers stretch geometrically from
`first_layer_m` to `alt_max_m`, concentrating resolution near the surface, with
the ratio solved so they span the domain exactly) or an **explicit ascending
list** of AGL heights in metres. Omit it for the 40-level default.

Guidance: ~40 levels roughly matches ERA5 model levels' near-surface density (~23
below 1.5 km); beyond that you are interpolating rather than resolving new
structure. **The host meteorology cache scales linearly with the level count** —
at 192 cached hours on the EUROPE domain, 23 levels is ≈52 GiB but 40 levels is
≈90 GiB, so raise SLURM `--mem` accordingly.
`make_multisite_config.py --vertical-levels N` does this arithmetic for you.

---

## Units

Unit strings are matched case- and whitespace-insensitively, so `"m s**-1"`,
`"m/s"` and `"ms-1"` are equivalent.

| Quantity | Accepted |
| --- | --- |
| Velocities (`u`, `v`, `friction_velocity`, geometric `vertical_velocity`) | `m s**-1`, `m/s`, `ms-1`, equivalents |
| Omega (`vertical_velocity` as a pressure tendency) | `Pa s**-1`, `Pa/s`, equivalents |
| Pressure (`surface_pressure`) | `Pa`, `pascal`, `pascals` — plus `hPa`/`mbar`/`millibar` for the pressure-level vertical coordinate |
| Temperature | `K`, `kelvin` |
| Geopotential | `m**2 s**-2`, `m2/s2`, equivalents |
| Length (`boundary_layer_height`) | `m`, `meter(s)`, `metre(s)` |
| Heat flux | `W m**-2` (instantaneous) or `J m**-2` (accumulated), equivalents |
| Specific humidity | `kg kg**-1` or `1` (must be non-empty) |

The safest choice is to reproduce the ERA5 CF `units` strings from the
[required-variables table](#required-variables).

---

## Preparing meteorology from a non-ERA5 source

Archives built for other models tend to miss this contract in a small number of
predictable ways. The checklist below came out of assessing a terrain-following
NWP archive (the Met Office UM) against the schema; most of it generalises.

| Gap in the source | What to do |
| --- | --- |
| No `geopotential` / `geopotential_at_surface` | On a terrain-following source with a $z = z_{\text{lev}} + \sigma\ h_s$ coordinate, solve for the orography hydrostatically from the store's own 3-D pressure, then form both fields from the coordinate definition |
| Coarser than hourly | Nothing — GLIDE reads the source cadence directly. Check the caveats under [Coordinates](#coordinates) first |
| No `friction_velocity` | $u_\ast = \sqrt{\lvert\tau\rvert/\rho}$ from the two surface stress components, with $\rho = p_s/(R_d T_v)$ and $T_v = T(1 + 0.6077q)$ |
| Heat flux is positive-**up** | Negate it to the ECMWF positive-down convention |
| Non-GLIDE variable names | Rename to GLIDE's defaults — `main.py` exposes no `variable_map` override |
| Level coordinate is 1..N indices | Tag the store `glide_vertical_coordinate="model_level"`, or GLIDE will refuse it rather than read the indices as pressures |
| Heat flux is accumulated (`J m**-2`) | Tag the store `glide_flux_accumulation_seconds=<source period>`, or GLIDE will refuse to guess the divisor |
| Domain-spanning chunks | Rechunk — see [Chunking](#chunking) |

Three things worth knowing before you start:

- **Time-invariant geopotential.** On a terrain-following source the 3-D
  geopotential does not change with time, and GLIDE reads it fine written as
  `(level, latitude, longitude)` with no time dimension — turning a per-timestep
  3-D field into a few hundred MB. ERA5's varies, so `download_sample_cube.py`
  cannot do this.
- **You no longer need to interpolate to hourly.** GLIDE brackets on the source's
  own timestep, so a 3-hourly archive is read as-is — no 3× storage cost for an
  upsampled copy. Interpolating up would have been exactly equivalent
  numerically (GLIDE interpolates linearly within the bracket anyway, and
  composing the two is exact), so this is a storage and bookkeeping win rather
  than an accuracy one: 3-hourly met still resolves less than hourly met.
- **Crop before you convert.** An hourly, full-domain rewrite is easily several
  times the source archive's own volume, and NWP archive domains are usually far
  wider than a run domain. Drop levels above the run's `alt_max_m` and crop to
  the bounding box you actually intend to run.

If you control the *extraction* rather than converting after the fact, build to
this schema at extraction time — it avoids a full read-rewrite pass over the
archive, and the derivations above are cheaper where the native fields still
exist.

---

## Domain and extent

- **Horizontal.** The store's grid points must cover the run's `met_domain`
  box, including at least one point on or beyond each edge. GLIDE removes a
  particle when it leaves `met_domain`, not the store's extent. Where no grid
  point lies on a `met_domain` bound, GLIDE's reader also takes the first
  point beyond it, so particles between the outermost interior point and the
  bound interpolate instead of taking the edge value (decision 0019). Where a
  grid point lies on the bound (ERA5 on the EUR and NAM boxes), nothing beyond
  it is read, so a store wider than the box does not change results.
- **Vertical.** Keep the source's levels up to at least 20 km above the
  ground everywhere in the domain; for ERA5 and the UM, keep all levels. The
  column releases need a model top of at least 20 km (validation-plan §4). Dropping levels above the run's
  `alt_max_m` (as under
  [Preparing meteorology from a non-ERA5 source](#preparing-meteorology-from-a-non-era5-source))
  is for development stores only.
- **Time.** The source's own timestamps, without interpolation in time,
  covering the run's backward window.

---

## Provenance attributes

Store-level attributes that say what the store is and how it was made. GLIDE
copies them into each run's `run_metadata.json` (task B1). Stores written by
met-bazarr carry all of them; they are required in validation inputs. All
values are strings unless stated.

| Attribute | Content |
| --- | --- |
| `glide_format` | `"glide"` |
| `glide_format_version` | The version of this page the store implements, e.g. `"1.0"` |
| `glide_vertical_coordinate` | `"pressure_level"` or `"model_level"` (see [Vertical coordinate](#vertical-coordinate)) |
| `glide_flux_accumulation_seconds` | Accumulation period of a `J m**-2` heat flux (see [Sensible heat flux](#sensible-heat-flux)); omit for `W m**-2` |
| `glide_flux_averaging_seconds` | Averaging interval of a `W m**-2` heat flux; `0` for instantaneous; omit for `J m**-2` |
| `source_dataset` | The source product, precisely enough to fetch it again: e.g. `"ERA5 model levels, ARCO gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"`, or `"Met Office global UM analyses, NAME met archive series UMG_Mk10PT"` |
| `source_access` | How and when it was read: host or archive path, and the UTC date |
| `producer` | The tool that wrote the store, e.g. `"met-bazarr"` |
| `producer_version` | Its version |
| `producer_commit` | Its git commit |
| `processing` | Every derivation and interpolation applied, in order, as plain text: derived fields and their formulas, destaggering (target grid and method), unit and sign changes, cropping |
| `compression` | Codec and level, and any lossy encoding per variable; `"lossless"` if none |
| `domain` | A name for the box, e.g. `"EUR"` or `"NAM"`, if it is one of the validation domains |
| `glide_filled_values` | Only if values were filled ([Validation inputs](#validation-inputs)): a JSON list with one object per filled value, with keys `variable`, `time` (ISO, UTC), `latitude`, `longitude`, `level` (3-D variables only), `before`, `after` and `filled` |

---

## Validation inputs

The meteorology for the registered validation runs (validation-plan §3–§4)
meets these rules in addition to the rest of this page:

- **Lossless.** Variables are stored as the source gives them or as
  `float32`, with lossless compression only: no `int16` scale and offset, no
  bit-rounding (decision [0018](../dev/decisions/0018-met-bazarr.md)). A
  lossy encoding is allowed only after a decision record that compares it
  against the golden fixtures.
- **All provenance attributes** above are present.
- **Isolated missing values in the source are filled by a fixed rule.**
  Source archives occasionally hold a missing value (NaN). ARCO's ERA5, for
  example, has one in `surface_sensible_heat_flux` at 59.0° N, 35.0° E at
  2018-12-28 21:00. GLIDE rejects non-finite input, so:
  - **What may be filled:** a non-finite value in a time-varying variable,
    where the same grid cell (and level) has finite values at the timesteps
    immediately before and after it. Each fill covers a single timestep, and
    a store has no more than 10 filled values per variable.
    `geopotential` and `geopotential_at_surface` are never filled.
  - **How:** the mean of the values at the same cell at the timesteps before
    and after, that is, linear interpolation in time.
  - **Record:** every filled value is listed in the store attribute
    `glide_filled_values`, and the fill is noted in `processing`.
  - **Otherwise:** a gap longer than one timestep, a gap at the first or last
    timestep, or more than 10 per variable is not filled. The values are
    fetched from the original archive (for ERA5, the Copernicus Climate Data
    Store or MARS), or the conversion stops and the case is reported.
- **The conversion is fixed before it runs.** For a non-ERA5 source, the
  choices that affect results (the vertical coordinate route, the destaggering
  target and method, which source fields each GLIDE variable comes from) are
  written down and agreed before the case-library windows are converted, and
  are the same in every window.
- **Produced by a tagged met-bazarr release,** whose version is pinned in the
  pre-registration record.

---

## Storage precision

Meteorology stores are large enough that encoding matters. Measured on a
converted EUROPE crop, writing the time-varying fields as `int16` with CF
`scale_factor`/`add_offset` and compressing with Blosc/zstd5:

| Encoding | Relative size |
| --- | --- |
| `float32` + lz4 | 1.00× |
| `float32` + zstd5 | 0.70× |
| `float32` bitround(12) + zstd5 | 0.55× |
| **`int16` + zstd5** | **0.48×** |

Round-trip errors at `int16`: 0.003 m s⁻¹ (winds), 0.0015 K, 0.6 Pa,
5×10⁻⁷ kg kg⁻¹ — orders of magnitude below what the physics resolves.

Two rules if you do this:

- **Never quantise `geopotential` or `geopotential_at_surface`.** GLIDE forms
  near-surface layer thicknesses by differencing them, so quantisation error
  enters twice on a 20 m bottom layer. They are static on a terrain-following
  source and a fraction of a percent of the store anyway.
- **Choose ranges that cannot be exceeded, and check.** Out-of-range values
  *wrap* under CF scale/offset rather than clipping, turning one anomalous wind
  speed into a large negative one. Validate each block against its declared range
  as you write, and abort rather than clip.

One known cost: xarray infers the decoded dtype from the Python type of the
stored `scale_factor`, and Zarr keeps attributes as JSON, so scale/offset
variables always decode to `float64`. GLIDE casts to `float32` when it builds the
channel tensor, so this is a transient 2× on the per-hour bounding-box subset,
not on the store.

---

## Chunking

GLIDE reads **one hour at a time, over the particle cloud's bounding box**, up to
its vertical ceiling. That box is usually far smaller than the archive domain, so
the on-disk chunk shape decides how much of the store has to be decompressed to
serve it.

Chunk shapes are **storage-neutral** (measured within 0.5% across tile 64/96/128
and whole-domain), so there is nothing to trade off against.

Recommended: **one hour per chunk, a 128×128 horizontal tile, levels whole when
shallow and split around 24 when deep.** `download_sample_cube.py --chunk-tile`
and `--chunk-levels` set this.

Read amplification (bytes decompressed ÷ bytes used), worst-case straddling
placement, for a 274×551 ERA5 EUROPE crop on 37 pressure levels:

| Chunk `(lev, lat, lon)` | Chunk size | 2° box | 10° box | 40° box |
| --- | --- | --- | --- | --- |
| `(37, 274, 392)` (dask auto) | 15.90 MB | 1678× | 67× | 8× |
| **`(37, 128, 128)`** | **2.42 MB** | **256×** | **10×** | **3×** |
| `(37, 96, 96)` | 1.36 MB | 144× | 6× | 3× |
| `(37, 64, 64)` | 0.61 MB | 64× | 10× | 1× |

Two caveats before you re-chunk an existing archive:

- **This is a cold-cache argument.** With the store already in page cache, tiling
  is a small *loss* — per-chunk decompression overhead — and the domain-spanning
  shape won every warm timing tested (tile 128 cost ~4–10%, tile 64 measurably
  more). The amplification table governs a cold read on a shared filesystem.
- **Chunking is not the dominant cost of a fetch.** Profiling a window fetch:
  the AGL resample is ~74% (of which computing the regrid weights alone is ~50%)
  and the actual dask read only ~17%. Re-chunk when you are writing a store
  anyway; it is rarely worth a re-download on its own.

Deep vertical coordinates need the level split: ERA5's 137 model levels in a
single chunk is 8.98 MB and forces a full-depth read even though GLIDE only needs
levels below its ceiling. `geopotential` is the exception — the AGL mask cannot
be built without reading it at full depth — but it is one variable among many,
and on a terrain-following source it should be static anyway.

---

## Verifying a store

Confirm a prepared store round-trips before committing to a large conversion:

```python
from lpdm.met_reader import ArcoEra5ZarrReader

# pressure-level store
reader = ArcoEra5ZarrReader("path/to/your_met.zarr")

# model-level store (tagged; level_name is auto-corrected)
reader = ArcoEra5ZarrReader("path/to/your_met_ml.zarr")

# Fetch one window over a small box. This exercises the unit checks, the
# omega→w conversion, and the terrain-following AGL resample.
```

A missing variable, a missing or empty `units` attribute, an unrecognised unit
string, non-finite geopotential, or an untagged model-level coordinate will each
raise a clear error at this point.

---

## Format versions

| Version | Date | Change |
| --- | --- | --- |
| 1.0 | 2026-11-17 | First versioned definition ([decision 0019](../dev/decisions/0019-met-format-versioning.md)). Adds the format version, the provenance attributes, `glide_flux_averaging_seconds`, the one-grid rule, the domain and extent rules, and the rules for validation inputs (including the fill rule for isolated missing source values and `glide_filled_values`). No change to what GLIDE reads: stores that conformed before still run. |

| 2.0 | 2026-11-18 | Per-variable `cell_methods` and `glide_period_seconds` mark means and accumulations; GLIDE holds a period variable constant through each window at its end value, as NAME does, instead of treating it as a snapshot at its timestamp; validation inputs derive `friction_velocity` from the surface stresses ([decision 0020](../dev/decisions/0020-time-meaning-of-period-fields.md)). Stores of format 1.0 still read; their heat flux is placed by the store-level flux attributes. Changes results: ERA5 heat flux moves 30 minutes earlier, UM heat flux and friction velocity 1.5 hours earlier. |

A change that GLIDE's reader depends on, or that alters what a conforming store
contains, takes a new version and a decision record. Additions that only record
information take a minor version; anything else takes a major one.
