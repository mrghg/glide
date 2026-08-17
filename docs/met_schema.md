# Meteorology input contract

Everything a meteorology store must satisfy to be read by GLIDE
(`src/lpdm/met_reader.py`). ARCO ERA5 is the reference source, but **any** store
conforming to this page will run — the point of writing it down is that other
reanalyses, NWP output and regional models can be used.

**What is prescribed:** container format, coordinate conventions, variable names,
dimensions, units, and sign conventions. Get these exact.

**What is not:** horizontal resolution, vertical resolution, domain extent, time
span, chunk sizes. GLIDE adapts. In particular it runs on **either** pressure
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
| Time | `time` | `datetime64`, **UTC**, hourly cadence. Must cover the run's backward window. |
| Latitude | `latitude` | Degrees north. Ascending or descending both accepted. |
| Longitude | `longitude` | Degrees east. Either $-180..180$ or $0..360$; GLIDE detects and normalises. |
| Vertical | `level` | 1-D. Its *meaning* depends on the vertical mode below. |

Coordinate names are configurable on the reader (`lon_name`, `lat_name`,
`level_name`, `time_name`), but the defaults above are the path of least
resistance. The one you *will* set is `level_name` for model-level stores — and
even that is auto-corrected when the store is tagged (see below).

Hourly cadence is assumed by the accumulated-flux de-accumulation
(`accumulation_seconds`, default 3600 s). If your source uses a different
accumulation window, set `accumulation_seconds` to match, or supply instantaneous
fluxes.

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

Accumulated (`J m**-2`) fluxes are de-accumulated by dividing by
`accumulation_seconds`; instantaneous (`W m**-2`) fluxes pass through unscaled.

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

### Pressure levels (default)

The level coordinate holds **pressure**. Units come from its `units` attribute —
`hPa` (or `mbar`/`millibar`) or `Pa`. With no units attribute, values ≤ 2000 are
assumed hPa, otherwise Pa. Any ordering and any number of levels.

### Model / hybrid levels

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

The default is 23 levels, 13 of them below 1.5 km. That suits 37 pressure levels
(~5 below 1.5 km) but **under-uses a model-level source**, which carries ~20
there. Raise it:

```yaml
met_domain:
  alt_max_m: 15000.0
  vertical_levels: 40        # a count → geometrically stretched grid
  first_layer_m: 10.0        # lowest layer thickness
```

`vertical_levels` accepts either a **count** (layers stretch geometrically from
`first_layer_m` to `alt_max_m`, concentrating resolution near the surface, with
the ratio solved so they span the domain exactly) or an **explicit ascending
list** of AGL heights in metres. Omit it for the built-in default.

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
| Coarser than hourly | Interpolate the time-varying fields to hourly (see below) |
| No `friction_velocity` | $u_\ast = \sqrt{\lvert\tau\rvert/\rho}$ from the two surface stress components, with $\rho = p_s/(R_d T_v)$ and $T_v = T(1 + 0.6077q)$ |
| Heat flux is positive-**up** | Negate it to the ECMWF positive-down convention |
| Non-GLIDE variable names | Rename to GLIDE's defaults — `main.py` exposes no `variable_map` override |
| Level coordinate is 1..N indices | Tag the store `glide_vertical_coordinate="model_level"`, or GLIDE will refuse it rather than read the indices as pressures |
| Domain-spanning chunks | Rechunk — see [Chunking](#chunking) |

Three things worth knowing before you start:

- **Time-invariant geopotential.** On a terrain-following source the 3-D
  geopotential does not change with time, and GLIDE reads it fine written as
  `(level, latitude, longitude)` with no time dimension — turning a per-timestep
  3-D field into a few hundred MB. ERA5's varies, so `download_sample_cube.py`
  cannot do this.
- **Sub-hourly bracketing is not supported.** GLIDE brackets on whole hours, so a
  3-hourly source must be interpolated up to hourly before it is read. Linear
  interpolation costs nothing in accuracy — GLIDE interpolates linearly within
  the bracket anyway, and composing the two is exact — but it costs 3× the
  storage. Reading the source cadence directly is a known future change.
- **Crop before you convert.** An hourly, full-domain rewrite is easily several
  times the source archive's own volume, and NWP archive domains are usually far
  wider than a run domain. Drop levels above the run's `alt_max_m` and crop to
  the bounding box you actually intend to run.

If you control the *extraction* rather than converting after the fact, build to
this schema at extraction time — it avoids a full read-rewrite pass over the
archive, and the derivations above are cheaper where the native fields still
exist.

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
