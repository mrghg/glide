# 0018 — Meteorology preparation lives in met-bazarr, not in GLIDE

**Context.** GLIDE needs meteorology from several centres (ECMWF now; the Met
Office and NOAA later), as compressed Zarr stores that meet its contract
([docs/met_schema.md](../../docs/met_schema.md)). GATES reads meteorology as
Zarr stores too. Each source has its own download client and file format
(`cdsapi` or earthkit for ECMWF, `iris` for Met Office files, others for NOAA),
and keeping those clients in GLIDE or GATES would bloat both with heavy
dependencies that neither model needs at run time. FLEXPART keeps its
preparation tool, flex_extract, separate for the same reason.

**Decision.** Meteorology preparation moves to a separate repository,
[met-bazarr](https://github.com/mrghg/met-bazarr) (package `metbazarr`), built
for GLIDE first. GATES's meteorology processing (currently
[extract_UM_met](https://github.com/elenafillo/extract_UM_met)) may also be
incorporated; that is still to be discussed with the team and the GATES
maintainer. The boundary:

- **met-bazarr does data preparation:**
  - download, by wrapping the existing clients, never by writing new ones;
  - subsetting, optional regridding, renaming;
  - units, sign conventions, and converting accumulated fields to rates;
  - chunking and compression;
  - an ensemble-member dimension where the source has one;
  - provenance attributes and a validator for each output format.
- **GLIDE keeps everything that is physics or run time:**
  - resampling onto its terrain-following levels;
  - the vertical-velocity conversion;
  - the pressure reconstruction for model levels;
  - the GLIDE-ready cache (roadmap 7d), which depends on GLIDE's vertical grid;
  - combining sources during a run (roadmap 3b);
  - its read-time checks (heat-flux sign, accumulation period and so on),
    kept as a second line of defence.

  These are covered by the physics freeze and the audit table, so they stay
  where the freeze applies.
- **Named output formats, not arbitrary ones,** to begin with: a `glide`
  format, and a `gates` format if GATES's processing is incorporated. GLIDE's
  `docs/met_schema.md` remains the authoritative definition of the `glide`
  format; met-bazarr implements it and records which version it implements.
  Generalise once a third consumer exists.
- **Provenance.** Every store records the met-bazarr version and commit, the
  source dataset, the output format and its version, and the compression
  settings. GLIDE copies these into each run's `run_metadata.json`.
- **Lossless by default.** Bit-rounding or other lossy compression is off by
  default. It is enabled for GLIDE only after a comparison against the golden
  fixtures shows its effect is negligible, with a decision record, because it
  changes the model's inputs. MPTRAC is prior art here: its current release
  (v3.2) reads meteorology compressed with ZFP, ZSTD, SZ3, LZ4, a multiscale
  scheme and netCDF quantisation, with precision set per variable (see
  `met_type` and the `met_*_prec` settings in its `src/mptrac.h`). Per-variable
  precision is the model for what met-bazarr records.
- **Pinning.** The pre-registration record and the v1 release pin the
  met-bazarr version used for the validation inputs, alongside the physics
  tag, and the release archives it.
- **No run-time dependency.** GLIDE reads met-bazarr's stores through its
  existing reader and does not import `metbazarr`.

**Rationale.** The download clients are the strongest reason: they are large,
change often and have nothing to do with transport. It also gives one place to
test the conversions where errors are silent (signs, units, accumulations). If
GATES's processing is incorporated, the two models would get identical inputs
from one tool, which would help the GATES training campaign (roadmap 5a).

**Rejected alternatives.**
- Keep preparation in GLIDE: GLIDE would carry every download client.
- Move GLIDE's vertical resampling and other transforms into met-bazarr as
  well: part of the frozen physics would then live outside the freeze and the
  audit.
- Support arbitrary user-defined formats from the start: a large framework
  neither model needs yet.

**Status.** Proposed 2026-10-19. Adopted in GLIDE by queue task B1; roadmap 3a
and the Met Office converter in 3c move to met-bazarr.
