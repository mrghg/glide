# 0013 — A composite meteorological source rather than a nesting feature

**Context.** Two needs point the same way: nesting high-resolution regional
meteorology (UKV at 1.5 km) inside global fields, and driving transport with
ensembles or with ML weather-model output. Both are "more than one source of
wind for the same particle cloud".

**Decision.** Build a composite met source: an ordered list of readers, each
with its own bounding box, cadence, vertical mode and per-domain physics
options (meander, convection), composed into one object that answers the same
per-window request the runtime makes today. Per particle, sample from the
innermost domain containing it; select by mask so shapes stay static. Hard
switch at the interface by default (as FLEXPART), optional blend zone. AGL is
relative to each source's own terrain, so a particle keeps its AGL across the
boundary.

**Rationale.** The abstraction covers nesting, ensemble members, mixed-period
sources and ML-NWP input with one mechanism, and is the first customer for
compressed met stores. UKV-in-UMG is its first test case, which also gives a
clean NAME comparison.

**Rejected alternatives.** A UK-specific nesting path; interpolating the fine
source onto the coarse grid (loses the point); rewriting the step loop per
source.

**Status.** Proposed 2026-09-16; v1.x (roadmap 3b, 3c). Docs: architecture.md
meteorology pipeline once implemented.
