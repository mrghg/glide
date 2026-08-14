# GLIDE meteorology schema

The contract every meteorology store must satisfy to be read by GLIDE's
`ArcoEra5ZarrReader` (`src/lpdm/met_reader.py`). ERA5 from the ARCO archive is the
reference source, but **any** dataset that conforms to this schema will run —
this page is written for preparing met from other sources (other reanalyses, NWP
output, regional models).

**What is prescribed:** the container format, coordinate conventions, variable
names, dimensions, units, and sign conventions. Get these exact.

**What is *not* prescribed:** horizontal resolution, vertical resolution (number
of levels), domain extent, time span, and chunk sizes. GLIDE adapts to whatever
grid you provide. In particular it runs on **either** pressure-level **or**
model-level (hybrid) met — see [Vertical coordinate](#vertical-coordinate).

`scripts/download_sample_cube.py` produces conforming stores from ARCO ERA5 and is
the best worked example to imitate. If your source is not ERA5, see
[Preparing met from a non-ERA5 source](#preparing-met-from-a-non-era5-source) for
the gaps you will most likely have to close yourself.

## Container format

- **Zarr**, with **consolidated metadata** (written via
  `to_zarr(..., consolidated=True)`; opened with `xr.open_zarr(..., consolidated=True)`).
  Zarr v2 or v3 are both fine.
- **One logical dataset per store.** You may split a long time series across
  several stores and hand GLIDE the list (or a local glob); they are stitched
  along `time`. **All stores in a set must share byte-identical `latitude`,
  `longitude`, and vertical coordinates** — only `time` may differ.
- **dtype:** `float32` or `float64` (GLIDE casts to `float32` on read by default).
  **CF `scale_factor`/`add_offset` into `int16` also conforms** — xarray decodes it
  before GLIDE sees anything — and roughly halves a store for errors far below what
  the physics resolves — see [Storage precision](#storage-precision).
- **No non-finite values** in the required variables over the domain you intend to
  run. NaNs/Infs in geopotential are rejected outright; the downloader's
  validator (`_validate_written_store`) refuses to install a store containing any.
- **Every required variable must carry a non-empty CF-style `units` attribute**
  (see [Units](#units)). A missing/empty `units` attr is a hard error — GLIDE will
  not guess.

## Coordinates

| Coordinate | Default name | Requirement |
| --- | --- | --- |
| Time | `time` | `datetime64`, **UTC**, hourly cadence. Must cover the run's backward window. |
| Latitude | `latitude` | Degrees north. Ascending or descending both accepted. |
| Longitude | `longitude` | Degrees east. Either `-180..180` or `0..360` accepted (GLIDE detects and normalises internally). |
| Vertical | `level` | 1-D. Meaning depends on the vertical mode below. |

Coordinate **names** are configurable on the reader (`lon_name`, `lat_name`,
`level_name`, `time_name`), but the defaults above are the path of least
resistance — name your coordinates this way and nothing needs overriding. The one
exception you *will* set is `level_name` for model-level met (see below).

Hourly time cadence is assumed by the accumulated-flux de-accumulation
(`accumulation_seconds`, default 3600 s). If your source uses a different
accumulation window, set `accumulation_seconds` to match, or supply instantaneous
fluxes (see `shf` below).

## Required variables

All are functions of `(time, level, latitude, longitude)` for 3-D fields or
`(time, latitude, longitude)` for surface fields. `geopotential_at_surface` may be
2-D or 3-D (a singleton level is fine). Names below are GLIDE's defaults; they can
be remapped via the reader's `variable_map`, but matching these names means no
remap is needed.

| Variable name | Role | Dims | Recommended `units` | Notes |
| --- | --- | --- | --- | --- |
| `u_component_of_wind` | zonal wind | 3-D | `m s**-1` | |
| `v_component_of_wind` | meridional wind | 3-D | `m s**-1` | |
| `vertical_velocity` | vertical motion | 3-D | `Pa s**-1` **or** `m s**-1` | omega (Pa/s) is converted to geometric w; see [Vertical velocity](#vertical-velocity). |
| `temperature` | air temperature | 3-D | `K` | Needed for the omega→w conversion. |
| `geopotential` | geopotential | 3-D | `m**2 s**-2` | **Geopotential, not geopotential height.** GLIDE forms height as `(z − z_sfc)/g`. |
| `specific_humidity` | specific humidity | 3-D | `kg kg**-1` | Deep convection (Emanuel). Use `kg kg**-1` or `1`; must be non-empty. |
| `boundary_layer_height` | BL depth | surface | `m` | |
| `surface_pressure` | surface pressure | surface | `Pa` | |
| `geopotential_at_surface` | surface geopotential (orography) | surface (2-D or 3-D) | `m**2 s**-2` | The terrain reference for the AGL conversion. |
| `friction_velocity` | u\* | surface | `m s**-1` | Hanna turbulence. |
| `surface_sensible_heat_flux` | sensible heat flux | surface | `W m**-2` **or** `J m**-2` | Sign convention matters — see [Sensible heat flux](#sensible-heat-flux). |

There are no optional-but-recognised extras beyond these; a conforming store
contains exactly this set (plus coordinates).

## Sign and unit conventions (read this)

These are the easy things to get wrong when converting a non-ERA5 source.

### Vertical velocity

`vertical_velocity` may be supplied as **omega** (pressure tendency, `Pa s**-1`) or
as **geometric velocity** (`m s**-1`). If it is omega, GLIDE converts it with
`w = −(R_d·T / (g·p))·omega`, which needs the pressure `p` at each level.

- On **pressure levels**, `p` is the level coordinate — omega works directly.
- On **model levels**, the level coordinate is not a pressure, so GLIDE
  reconstructs per-level pressure hydrostatically from `geopotential`,
  `surface_pressure`, `temperature` and `specific_humidity` (see
  [Vertical coordinate](#vertical-coordinate)) and uses that. **Omega works on
  model levels too** — no need to pre-convert (ARCO's model-level product ships
  omega). Supplying `m s**-1` directly is also accepted and skips the conversion.

### Sensible heat flux

GLIDE's boundary-layer physics uses **positive = upward**, but expects the
**input** to follow the **ECMWF/ERA5 convention: positive = downward** (into the
surface). GLIDE negates internally. Match ERA5: a daytime, upward surface sensible
heat flux should be stored **negative**. If your source is already positive-upward,
negate it before writing (or remap through a source that isn't).

Accumulated (`J m**-2`) fluxes are de-accumulated by dividing by
`accumulation_seconds`; instantaneous (`W m**-2`) fluxes pass through unscaled.

### Geopotential, not geopotential height

`geopotential` and `geopotential_at_surface` are true geopotential (`m² s⁻²`), not
height in metres. GLIDE derives geometric height as `(geopotential −
geopotential_at_surface) / g`. Do not pre-divide by g.

Sub-surface levels (where a level sits below local terrain, giving negative AGL)
are **allowed and expected** — do not clamp them to zero or mask them. GLIDE
handles them in the terrain-following resample.

## Vertical coordinate

GLIDE supports two vertical modes, distinguished only by what the level
coordinate *means*. Both require `geopotential` (3-D) and
`geopotential_at_surface`; height always comes from geopotential, so both modes
share the same AGL machinery.

**Pressure levels** (default, `level_name="level"`):
- The level coordinate holds **pressure**. Units are read from its `units` attr —
  `hPa` (a.k.a. `mbar`/`millibar`) or `Pa`. With no units attr, values `≤ 2000`
  are assumed hPa, otherwise Pa.
- Any ascending/descending order and any number of levels.

### The internal AGL grid sets the resolution the physics sees

Whichever mode a source uses, GLIDE resamples every met hour onto one **fixed
terrain-following AGL ladder** shared by all columns (that shared 1-D ladder is
what makes the vertical interpolation cheap enough for the per-step hot path). So
the *source's* level count does not set the model's effective vertical resolution
— this grid does, and it also sets the met-cache size.

The default is a 23-level ladder resolving ~13 levels below 1.5 km. That suits the
37 pressure levels (~5 below 1.5 km) but **under-uses a model-level source**, which
carries ~20 there. Raise it via `met_domain.vertical_levels`:

```yaml
met_domain:
  alt_max_m: 15000.0
  vertical_levels: 40        # count -> geometrically stretched grid
  first_layer_m: 10.0        # lowest layer thickness (default: ERA5's lowest model level)
```

`vertical_levels` accepts either a **count** (levels are geometrically stretched
from `first_layer_m` to `alt_max_m`, concentrating resolution near the surface) or
an **explicit ascending list** of AGL heights in metres for full control. Omit it
to keep the built-in default.

Guidance: ~40 levels roughly matches ERA5 model levels' near-surface density
(~23 below 1.5 km); beyond that you are interpolating rather than resolving new
structure. **The host met cache scales linearly with the level count** — at 192
cached hours on the EUROPE domain, 23 levels ≈ 52 GiB but 40 levels ≈ 90 GiB, so
raise SLURM `--mem` accordingly (`make_multisite_config.py --vertical-levels N`
does this arithmetic for you).

**Model / hybrid levels** (auto-detected, or `vertical_coordinate="model"`):
- The level coordinate is a **level index**, not a pressure. Per-level heights
  come directly from the 3-D `geopotential` field, exactly as on pressure levels.
- **Per-level pressure is reconstructed hydrostatically** from `geopotential`,
  `surface_pressure`, `temperature`, and `specific_humidity` (via the hypsometric
  relation, integrated from the surface). This is needed for the omega→w
  conversion, air density, and convection. **No hybrid a/b coefficients are
  required** — deliberately, because ARCO does not ship them and a third-party
  coefficient table cannot be guaranteed to match the data. `specific_humidity`
  and `surface_pressure` are therefore mandatory for model-level met.
- GLIDE auto-detects model mode from the `glide_vertical_coordinate` store attr
  and auto-corrects `level_name` to the store's vertical coordinate (e.g.
  `hybrid`). Override explicitly with the reader's `vertical_coordinate` /
  `level_name` if your store lacks the attr.
- Model levels require the terrain-following path (`terrain_following=True`, the
  default).

**Tag your model-level cubes.** GLIDE refuses to read a store as pressure levels
when the vertical coordinate doesn't look like pressures — it is named `hybrid` /
`model_level`, declares a `hybrid_sigma_pressure` `standard_name`, or holds
consecutive integers (level indices). Without that guard the indices would be read
as pressures, silently corrupting the omega→w conversion, air density, and
convection. Fix it by tagging the store (below) or constructing the reader with
`vertical_coordinate="model"`.

Store which mode a cube uses in a `glide_vertical_coordinate` attr
(`"pressure_level"` or `"model_level"`) so it is self-describing on disk;
`download_sample_cube.py` does this.

## Units

Unit strings are matched case-insensitively and whitespace-insensitively, so
`"m s**-1"`, `"m/s"`, and `"ms-1"` are equivalent. Accepted forms per variable:

- **Velocities** (`u`, `v`, `friction_velocity`, and `vertical_velocity` if
  geometric): `m s**-1`, `m/s`, `ms-1`, and equivalents.
- **Omega** (`vertical_velocity` if a pressure tendency): `Pa s**-1`, `Pa/s`, and
  equivalents.
- **Pressure** (`surface_pressure`): `Pa`, `pascal`, `pascals`. The vertical
  coordinate on pressure levels also accepts `hPa`/`mbar`/`millibar`.
- **Temperature** (`temperature`): `K`, `kelvin`.
- **Geopotential** (`geopotential`, `geopotential_at_surface`): `m**2 s**-2`,
  `m2/s2`, and equivalents.
- **Length** (`boundary_layer_height`): `m`, `meter(s)`, `metre(s)`.
- **Heat flux** (`surface_sensible_heat_flux`): `W m**-2` (instantaneous) or
  `J m**-2` (accumulated), and equivalents.
- **Specific humidity** (`specific_humidity`): `kg kg**-1` or `1` (must be
  non-empty).

The safest choice is to reproduce the ERA5 CF `units` strings shown in the
[required-variables table](#required-variables).

## Preparing met from a non-ERA5 source

Archives built for other models tend to miss this contract in a small number of
predictable ways. The checklist below came out of assessing a terrain-following NWP
archive (the Met Office UM) against the schema; most of it generalises.

| Gap in the source | What to do about it |
| --- | --- |
| No `geopotential` / `geopotential_at_surface` | On a terrain-following source with a `z = level_height + sigma·orog` coordinate, solve for the orography hydrostatically from the store's own 3-D pressure, then form both fields from the coordinate definition |
| Coarser than hourly | Interpolate the time-varying fields to hourly (see below) |
| No `friction_velocity` | `u* = sqrt(\|τ\|/ρ)` from the two surface stress components, with `ρ = p_sfc/(R_d·T_v)` and `T_v = T(1 + 0.6077q)` |
| Heat flux is positive-**up** | Negate it to the ECMWF positive-down convention this page requires |
| Non-GLIDE variable names | Rename to GLIDE's defaults — `main.py` exposes no `variable_map` override |
| Level coord is 1..N indices | Tag the store `glide_vertical_coordinate="model_level"`, or GLIDE will refuse it rather than read the indices as pressures |
| Domain-spanning chunks | Rechunk — see [Chunking](#chunking) |

Three things worth knowing before you start:

- **Time-invariant geopotential.** On a terrain-following source the 3-D
  geopotential does not change with time, and GLIDE reads it fine written as
  `(level, latitude, longitude)` with no time dimension — turning a per-timestep
  3-D field into a few hundred MB. ERA5's varies, so `download_sample_cube.py`
  cannot do this.
- **Sub-hourly bracketing is not yet supported.** GLIDE brackets met on whole hours
  (`_canonicalize_hour_bounds`), so a 3-hourly source has to be interpolated up to
  hourly before it is read. Linear interpolation costs nothing in accuracy — GLIDE
  interpolates linearly within the bracket anyway, and composing the two is exact —
  but it costs 3× the storage. Reading the source cadence directly is a known
  future change.
- **Crop before you convert.** An hourly, full-domain rewrite is easily several
  times the source archive's own volume, and NWP archive domains are usually far
  wider than a run domain. Drop levels above the run's `alt_max_m` and crop to the
  bounding box you actually intend to run.

If you control the *extraction* rather than converting after the fact, build to
this schema at extraction time — it avoids a full read-rewrite pass over the
archive, and the derivations above are cheaper where the native fields still exist.

## Storage precision

Met stores are large enough that encoding matters. Measured on a converted EUROPE
crop, writing the time-varying fields as `int16` with CF `scale_factor`/`add_offset`
and compressing with Blosc/zstd5:

| encoding | relative size |
| --- | --- |
| `float32` + lz4 | 1.00× |
| `float32` + zstd5 | 0.70× |
| `float32` bitround(12) + zstd5 | 0.55× |
| **`int16` + zstd5** | **0.48×** |

Round-trip errors at `int16`: 0.003 m s⁻¹ (winds), 0.0015 K, 0.6 Pa, 5e-7 kg kg⁻¹ —
orders of magnitude below what the physics resolves.

Two rules if you do this:

- **Never quantise `geopotential` / `geopotential_at_surface`.** GLIDE forms
  near-surface layer thicknesses by differencing them, so quantisation error enters
  twice on a 20 m bottom layer. They are static on a terrain-following source and a
  fraction of a percent of the store anyway.
- **Choose ranges that cannot be exceeded, and check.** Out-of-range values *wrap*
  under CF scale/offset rather than clipping, turning one anomalous wind speed into
  a large negative one. Validate each block against its declared range as you
  write, and abort rather than clip.

One known cost: xarray infers the decoded dtype from the Python type of the stored
`scale_factor`, and Zarr keeps attrs as JSON, so scale/offset variables always decode
to `float64`. GLIDE casts to `float32` when it builds the channel tensor, so this is
a transient 2× on the per-hour bounding-box subset, not on the store.

## Chunking

GLIDE reads **one hour at a time, over the particle cloud's bounding box**, up to its
vertical ceiling. That box is usually far smaller than the archive domain, so the
on-disk chunk shape decides how much of the store has to be decompressed to serve it.

Chunk shapes are **storage-neutral** (measured within 0.5% across tile 64/96/128 and
whole-domain), so there is nothing to trade off against.

Recommended: **one hour per chunk, a 128×128 horizontal tile, levels whole when
shallow and split around 24 when deep.** `download_sample_cube.py --chunk-tile`
and `--chunk-levels` set this.

Read amplification (bytes decompressed ÷ bytes used), worst-case straddling
placement, for a 274×551 ERA5 EUROPE crop on 37 pressure levels:

| chunk `(lev, lat, lon)` | chunk size | 2° box | 10° box | 40° box |
| --- | --- | --- | --- | --- |
| `(37, 274, 392)` (dask auto) | 15.90 MB | 1678× | 67× | 8× |
| **`(37, 128, 128)`** | **2.42 MB** | **256×** | **10×** | **3×** |
| `(37, 96, 96)` | 1.36 MB | 144× | 6× | 3× |
| `(37, 64, 64)` | 0.61 MB | 64× | 10× | 1× |

Two caveats worth knowing before you re-chunk an existing archive:

- **This is a cold-cache argument.** With the store already in page cache, tiling is
  a small *loss* — per-chunk decompression overhead — and the domain-spanning shape
  won every warm timing tested (tile 128 cost ~4–10%, tile 64 measurably more). The
  amplification table is what governs a cold read on a shared filesystem.
- **Chunking is not the dominant cost of a fetch.** Profiling a window fetch:
  `_resample_hour_to_agl` 74% (of which `compute_agl_regrid_weights` alone is ~50%),
  the actual dask read only ~17%. Re-chunk when you are writing a store anyway; it
  is rarely worth a re-download on its own.

Deep vertical coordinates need the level split: ERA5's 137 model levels in a single
chunk is 8.98 MB and forces a full-depth read even though GLIDE only needs levels
below its ceiling. `geopotential` is the exception — the AGL mask cannot be built
without reading it at full depth — but it is one variable among many, and on a
terrain-following source it should be static (no `time` dim) anyway.

## Verifying a store

Confirm a prepared store round-trips before committing to a large conversion:

```python
from lpdm.met_reader import ArcoEra5ZarrReader

# pressure-level store
reader = ArcoEra5ZarrReader("path/to/your_met.zarr")
# model-level store
reader = ArcoEra5ZarrReader("path/to/your_met_ml.zarr", level_name="hybrid")

# Fetch one window over a small box; this exercises unit checks, the omega→w
# conversion, and the terrain-following AGL resample.
```

A missing variable, a missing/empty `units` attr, an unrecognised unit string, or
non-finite geopotential will each raise a clear error at this point.
