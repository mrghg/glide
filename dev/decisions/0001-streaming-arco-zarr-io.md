# 0001 — Stream ARCO ERA5 from Zarr, not whole NetCDF/GRIB files

**Context.** Established LPDMs (NAME, FLEXPART) read monolithic NetCDF/GRIB met
files. That I/O is a primary scalability bottleneck: a regional, time-bounded run
still pages through large files, and staging the archive is a copy step.

**Decision.** Stream analysis-ready, cloud-optimised (ARCO) ERA5 directly from a
**Zarr** store, fetching only the chunks a run actually touches. Public ARCO
buckets open anonymously (`token="anon"`); local monthly `EUROPE_YYYYMM.zarr`
stores are supported and stitched along time.

**Rationale.** The chunked layout means a regional, time-bounded run reads a small
fraction of the archive. The met store becomes a queryable thing you sip from, not
a file you copy — the same code runs against a laptop-sized cube or the full cloud
archive, which is what makes the model scalable and portable.

**Rejected alternatives.**
- Pre-staging NetCDF/GRIB — the bottleneck we set out to remove.
- Native model-level data — see [0003](0003-terrain-following-agl-coordinate.md);
  pressure levels were chosen for ARCO availability, at the cost of the
  terrain-coordinate work.

**Status.** Foundational, in force. Reader: `src/lpdm/met_reader.py`. See
[docs/architecture.md](../../docs/architecture.md). Met I/O later became the
dominant wall cost at representative scale; addressed by per-hour caching
([0008](0008-multi-site-shared-met-batching.md)).
