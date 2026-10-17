# 0017 — ERA5 model levels as the default meteorology

**Context.** GLIDE reads ERA5 on either the 37 pressure levels or the 137
native model levels ([0010](0010-model-level-met-reader.md)). The download
tooling, the shipped configurations and the built-in vertical grid (23
terrain-following levels, 13 below 1.5 km) were all set up for pressure levels.
Pressure levels have only about 5 levels below 1.5 km, and they cut through
terrain, so the lowest levels over high ground are ERA5's below-ground
extrapolation. The model-level reader is implemented but not yet validated
(STATUS.md, "what's next" item 1).

**Decision.** Model levels become the default for all production and
validation runs, before the physics freeze:

- the download tooling fetches model levels by default, and only the levels
  needed below the domain top;
- the shipped configurations read model-level cubes;
- the built-in vertical grid is raised to resolve the model levels near the
  surface (about 40 levels as an interim value; the convergence study, task A5,
  sets the final one);
- the validation runs (Tiers 0 to 3) use model levels.

Pressure levels stay supported, for comparison and for sources that only
provide them.

**Rationale.** Model levels follow the terrain and are about four times denser
in the boundary layer, which is where footprint magnitudes are set. FLEXPART is
driven by ERA5 model levels through flex_extract, so the Tier 2 comparison
starts from the same native fields. The switch changes footprints everywhere,
so it must land before the freeze and before the golden fixtures and
benchmarks are recorded.

**Rejected alternatives.** Keeping pressure levels as the default and
validating on them (the validated configuration would not be the one
recommended for use). Validating on both (doubles the reference runs for no
gain in v1).

**Costs.** About 3.7 times the vertical data to download and store before
subsetting; a larger host meteorology cache, roughly in proportion to the
number of internal levels (about 90 GiB instead of 52 GiB at 192 cached hours on
the EUROPE domain); a somewhat slower vertical interpolation.

**Status.** Proposed 2026-10-16. Implemented by queue task A01 (roadmap 6d).
Docs: [docs/met_schema.md](../../docs/met_schema.md).
