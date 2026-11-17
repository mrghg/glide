# 0019 — A versioned `glide` meteorology format, with provenance and rules for validation inputs

**Context.** `docs/met_schema.md` defines what GLIDE reads, and decision
[0018](0018-met-bazarr.md) makes it the `glide` output format of met-bazarr. It
said met-bazarr would record which version of the format it implements, but the
format had no version, and the provenance attributes 0018 requires had no
names. A Met Office UM converter is about to be written for task B14 by an
agent on another system. Several choices that change results were not covered
by the page: how staggered fields are put on one grid, what a mean (not
instantaneous) heat flux is, how far a store must extend, and whether a
validation input may be lossy.

**Decision.** `docs/met_schema.md` becomes format version 1.0 and gains:
- `glide_format_version` and a history table;
- named provenance attributes (`source_dataset`, `source_access`, `producer`,
  `producer_version`, `producer_commit`, `processing`, `compression`, `domain`,
  `glide_format`), with every derivation and interpolation listed in
  `processing`;
- an explicit rule that every field is on one grid, with destaggering left to
  the converter and recorded;
- `glide_flux_averaging_seconds` for `W m**-2` heat fluxes (recorded, not yet
  read);
- domain and extent rules: cover `met_domain` with a point on or beyond each
  edge, keep levels to at least 20 km, and do not interpolate in time;
- rules for validation inputs: lossless, all provenance present, the conversion
  choices fixed and agreed before conversion, produced by a tagged met-bazarr
  release.

None of this changes what GLIDE reads, so existing stores still run.

**Rationale.** A converter written on another system should not have to make
choices that change results without recording them, and the pre-registration
needs to state exactly what the inputs were. Versioning gives met-bazarr
something to target and GLIDE something to check.

**Open, for B14.** Whether UM levels are read as model levels with derived
geopotential or through a new height-based vertical mode; whether GLIDE uses
the UM's own pressure; how GLIDE interpolates in the strip between a non-ERA5
store's outermost interior point and the `met_domain` edge (the reader
currently cuts the store at the bounds). Each that changes the reader takes a
new format version.

**Rejected alternatives.**
- Leave the details to met-bazarr: the choices would be made without the
  GLIDE context, and recorded only in another repository.
- Require GLIDE to read the provenance now: that is task B1's scope.

**Status.** Proposed 2026-11-17. Docs: [docs/met_schema.md](../../docs/met_schema.md).
