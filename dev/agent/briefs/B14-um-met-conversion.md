# Brief: Met Office UM analyses to GLIDE's meteorology format (task B14, met-bazarr part)

_For an agent working in [met-bazarr](https://github.com/mrghg/met-bazarr) on a
system that holds the Met Office meteorology used by NAME. Drafted 2026-11-17
by Claude Opus 5.5 for Matt Rigby, who decides at each stop below._

## Goal

Convert the Met Office global UM analyses that NAME runs on into
[GLIDE](https://github.com/mrghg/glide)'s meteorology format, so that GLIDE can
run on the same meteorology as NAME. This is GLIDE's second reference
comparison (validation-plan Tier 2, "NAME and Met Office meteorology").

The work is in **two stages, with a stop between them.** The stop exists
because some conversion choices change GLIDE's results, and they have to be
agreed and recorded before the validation inputs are produced. GLIDE's
validation is pre-registered.

## Read first

In GLIDE (`main`, or the branch `docs/met-schema-v1` until it merges):
- `docs/met_schema.md`: the format, **version 1.0**. Read all of it,
  especially "One grid for every field", "Sign and unit conventions",
  "Vertical coordinate", "Preparing meteorology from a non-ERA5 source",
  "Domain and extent", "Provenance attributes" and "Validation inputs".
- `dev/decisions/0018-met-bazarr.md` (what met-bazarr does and what stays in
  GLIDE) and `dev/decisions/0019-met-format-versioning.md` (what is still open).
- `dev/agent/QUEUE.md`, task **B14**: the full task. This brief covers its
  met-bazarr part.
- `docs/validation-plan.md` §4 (domains, blocks, windows) and the paragraph
  "What GLIDE needs to read UM fields".
- `docs/reference-runs.md` §2 (domain edges) and §7 (NAME settings).

In met-bazarr: `AGENTS.md`, `STATUS.md` and `docs/roadmap.md`. Milestone M1,
the `glide` format, is not yet built. A writer and validator for format 1.0
are part of this work.

## What we already know about the source

From the NAME run record for the reference runs (16–17 Nov 2026):
- series `UMG_Mk10PT` for every block, from the NAME met archive on JASMIN;
- regular latitude–longitude grid, 2,560 points around the globe:
  0.140625° in longitude by 0.09375° in latitude;
- 59 levels; fields every 3 hours;
- NAME reports the land fraction as missing for this series.

Check all of these against the files themselves. Do not rely on figures quoted
for other series.

## Stage 1: assessment (then stop)

Read the Mk10 files for **one day of block EUR-Jan** (8 Jan 2019) and write a
short report, in a met-bazarr progress note or a draft PR description, with:

1. **Grid and staggering.** Horizontal grid of each field, and any offsets
   between fields (Arakawa C or other). Vertical levels: how they are defined
   (heights above the terrain, `z = a + b × orography` or other), the level
   definition constants, where the top is, and which fields sit on which levels
   (rho or theta levels).
2. **Fields available**, against GLIDE's required variables: winds, vertical
   velocity (units), temperature or potential temperature, pressure or Exner
   function on levels, specific humidity, boundary-layer depth, surface
   pressure, orography, surface stress components or friction velocity,
   surface sensible heat flux. For the heat flux, give its sign convention and
   whether it is instantaneous or a mean over the 3-hour interval (and which
   interval: the one ending or the one starting at the timestamp).
3. **The route for each gap.** For each gap, say whether it can be closed in
   conversion (met-bazarr) or would need a change to GLIDE's reader. Address
   these in particular:
   - **Vertical coordinate.** Route A: write the UM levels as GLIDE model
     levels (`glide_vertical_coordinate = "model_level"`), with 3-D
     geopotential formed from the level heights and orography. GLIDE then
     rebuilds pressure hydrostatically from geopotential, temperature and
     humidity, and ignores the UM's own pressure. Route B: a new
     height-based vertical mode in GLIDE's reader. Recommend one. Estimate,
     for one column, how far GLIDE's rebuilt pressure would differ from the
     UM's pressure, if the files hold it.
   - **Destaggering.** Propose the target grid (which horizontal points and
     which levels all fields go to) and the interpolation method, with the
     reasons. NAME's own treatment, if you can find it, is the best guide.
   - **Vertical velocity.** It is probably in m s⁻¹. GLIDE accepts
     `m s**-1` (`docs/met_schema.md`, "Vertical velocity"); confirm that its
     definition matches GLIDE's (geometric `w`).
   - **The domain edge.** UM points do not fall on the EUR and NAM edges.
     GLIDE's reader currently cuts the store at exactly the `met_domain`
     bounds (`docs/met_schema.md`, "Domain and extent"). Say what a store
     cropped one point beyond each edge would contain. The GLIDE side of this
     is ours to check.
4. **Sizes.** Size of one converted day and of the four case-library windows
   (EUR-Jan, EUR-Jul, NAM-Jan, NAM-Jul; 24 days each), lossless `float32`.
   The stores have to be transferred to Isambard.

**Stop here** and hand the report to Matt. The choices in item 3 are then
agreed, written into GLIDE (a new format version and decision record if the
reader changes), and only then is stage 2 built against them.

## Stage 2: converter (after the agreement)

- In met-bazarr, a writer and validator for the `glide` format version
  agreed at the stop (1.0, or its successor if the reader changes), and a UM
  converter to it, covering what `docs/met_schema.md` lists for non-ERA5
  sources:
  - geopotential from the level definition and orography;
  - friction velocity from the surface stresses;
  - the heat-flux sign (store positive-down) and
    `glide_flux_averaging_seconds`;
  - temperature from potential temperature, if needed;
  - GLIDE's variable names and units strings;
  - destaggering as agreed;
  - every provenance attribute, with each step in `processing`.
- Rules for these validation inputs (`docs/met_schema.md`, "Validation
  inputs"):
  - lossless;
  - every level kept;
  - no interpolation in time (3-hourly stays 3-hourly);
  - cropped to cover the EUR or NAM box with one point beyond each edge.
- Checks on one day, reported in the PR:
  - converted pressure, temperature and winds against the source at a sample
    of points;
  - hydrostatic consistency of the geopotential;
  - boundary-layer depth and friction velocity within plausible ranges;
  - GLIDE's reader opening the store (`docs/met_schema.md`, "Verifying a
    store"), on CPU if GLIDE can be installed on that system.
- Tests on small synthetic stores with the same structure, never on Met
  Office data.
- Then the four case-library windows, from a tagged met-bazarr release:
  - EUR and NAM, 22 Dec 2018 00:00 – 15 Jan 2019 00:00;
  - EUR and NAM, 21 Jun 2019 00:00 – 15 Jul 2019 00:00.

## Rules

- **Restricted data.** Met Office fields, and anything derived from them, never
  go into any repository, test or issue. Tests use synthetic stores.
- **Nothing in GLIDE.** Do not change GLIDE's reader. Report what it needs. The
  GLIDE side of B14 is done on Isambard and must merge before GLIDE's physics
  freeze.
- **Ask before** choosing a destaggering method or vertical route that was not
  agreed at the stop, and before any lossy encoding.
- Record which model did the work in every commit and PR, as both repositories
  do.
- If something in GLIDE's format page is wrong or ambiguous for this source,
  report it rather than working around it. The page is meant to be fixed.
