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
the best worked example to imitate.

## Container format

- **Zarr**, with **consolidated metadata** (written via
  `to_zarr(..., consolidated=True)`; opened with `xr.open_zarr(..., consolidated=True)`).
  Zarr v2 or v3 are both fine.
- **One logical dataset per store.** You may split a long time series across
  several stores and hand GLIDE the list (or a local glob); they are stitched
  along `time`. **All stores in a set must share byte-identical `latitude`,
  `longitude`, and vertical coordinates** — only `time` may differ.
- **dtype:** `float32` or `float64` (GLIDE casts to `float32` on read by default).
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
