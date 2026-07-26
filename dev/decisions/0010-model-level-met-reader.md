# 0010 — Model-level met: reconstruct pressure hydrostatically, not from a/b

**Context.** GLIDE is adding support for ERA5's native 137 hybrid model levels
(finer near-surface, terrain-following; see [STATUS.md](../../STATUS.md) "what's
next"). Heights come from the archive's `geopotential` field either way, but the
per-step physics also needs **pressure** at each level — for the omega→w
conversion, air density, potential temperature, and convection. On pressure levels
the vertical coordinate *is* pressure; on model levels it is just a level index, so
pressure must be derived. ERA5 model levels are *defined* by hybrid coefficients
`a(k), b(k)` via `p = a + b·p_s`, so the a/b table is the exact route.

**Decision.** Reconstruct per-level pressure **hydrostatically** from the archive's
own fields — geopotential, surface pressure, temperature, and specific humidity —
via the hypsometric relation integrated from the surface
(`vertical_grid.model_level_pressure_pa`). Do **not** use hybrid a/b coefficients.
The reader auto-detects model mode from the `glide_vertical_coordinate` store attr,
auto-corrects the vertical coordinate name (e.g. `hybrid`), and reconstructs
pressure; a run just points `io.zarr_store` at a model-level cube.

**Rationale.** The a/b coefficients are **not in the ARCO archive** — verified: the
`ar/model-level` store's `hybrid` coord has no attrs, and the raw `co/model-level-*`
stores label it a CF hybrid-sigma-pressure coordinate but drop the `formula_terms`
and the `ap`/`b` variables. So a/b could only come from a third party (ECMWF's site,
a mirror), and there is no guarantee a third-party L137 table matches the
coefficients this data was actually built on — a mismatch would silently corrupt
every pressure. Deriving pressure from the same fields as everything else keeps the
whole cube self-consistent and source-agnostic (works for any model-level source
that provides geopotential, which the schema already requires), with no external
table to trust or maintain.

**Rejected alternatives.**
- **Hybrid a/b table** (exact for ERA5): not in the archive; third-party provenance
  can't be trusted to match; ERA5-specific.
- **Require a `pressure` field in the schema** (push to met-prep): every model-level
  source's prep would have to compute pressure; more schema surface for no gain over
  deriving it from fields already required.
- **Require `vertical_velocity` in m/s** (skip omega→w): sidesteps pressure for w
  only — density, θ, and convection still need it — and forces a pre-conversion.

**Status.** In force. Download (`--levels model`) and reader both implemented;
heights unchanged (`(z−z_sfc)/g`). Small approximation aloft from the layer-mean
virtual temperature; exact for an isothermal column (tested). Open: confirm the
approximation is tight enough for convection during the model-level validation run.
See [docs/met_schema.md](../../docs/met_schema.md) and
[0003](0003-terrain-following-agl-coordinate.md).
