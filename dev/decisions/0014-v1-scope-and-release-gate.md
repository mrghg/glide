# 0014 — v1 scope: a validated, fast footprint model without uncertainty quantification

**Context.** The roadmap spans efficiency, validation, uncertainty
quantification, nesting and emulator co-development. A release needs a
boundary that the community can evaluate against something familiar.

**Decision.** v1 is a validated backward LPDM comparable to FLEXPART and NAME,
with literature-default parameters aligned to FLEXPART v11, GPU speed,
Zarr-streamed or cached meteorology, point, column and satellite releases,
nested footprint output, and an installable, reproducible package. Excluded:
tangent footprints, transport-error covariance, calibrated parameters,
composite met source, and all research lines. The release gate is the
checklist in dev/roadmap.md §3. Efficiency changes that alter numerics land
before the physics freeze so that validation happens once.

**Rationale.** A v1 with literature defaults is like for like with FLEXPART,
which is the first comparison sceptics will make; calibrated parameters belong
with the error model that gives them meaning (v2). Validating the fast model
rather than a slow one that is rewritten afterwards is the only order that
does the validation once.

**Rejected alternatives.** Shipping calibrated defaults in v1 (not comparable,
and calibration depends on ETEX which is late in the schedule). Validating
first and optimising after (two validations).

**Status.** Adopted 2026-09-16. Docs: dev/roadmap.md.
