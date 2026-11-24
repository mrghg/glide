# 0023 — Deep convection in each meteorology column, batched over columns

**Context.** The reduced Emanuel scheme ([0007](0007-reduced-emanuel-convection.md))
lifted one parcel in the bounding-box-mean column of the meteorology, so
convection fired everywhere in the domain or nowhere. The Tier 2 comparisons run
deep convection in FLEXPART, NAME and GLIDE over continental domains, where that
makes GLIDE's convection meaningless: on the EUROPE reference domain in January
the mean column never fires, while a few hundred tropical and warm-sector columns
(of ~151,000) do in every hour. Task A02 (`dev/agent/QUEUE.md`).

**Decision.**
- **Every (lon, lat) column of the meteorology window gets its own parcel lift,
  trigger, closure and mass-flux matrix**, from its own temperature, humidity,
  height above ground and pressure on the terrain-following levels. The reader
  now carries the per-column pressure it already computed for the level mean
  (`HourlyMetTensors.pressure_hpa`), so a mountain column lifts on its own
  pressures.
- **Vectorised over columns**: profiles are [columns, levels] tensors and the
  batched functions (`compute_lcl_lnb_cape_columns`,
  `cloud_base_mass_flux_columns`, `compute_mass_flux_matrices`) do the same
  element-by-element arithmetic as the single-column ones, with no Python loop
  over columns or levels. Matrices ([F, Z, Z]) are built only for the F columns
  that fire.
- **A particle belongs to the column of the nearest meteorology grid point**, and
  its host layer is that column's level nearest its height. It draws its
  destination from its own column's backward transition matrix (FLEXPART
  `redist`, `ldirect=-1`), unchanged in form.
- **Interval**: once per meteorology window, scaled by the window's width, which
  the reader sets to the store's time step (`reader.met_interval_seconds`). This
  was already so (September 2026); A02 adds the reader-level acceptance test.
- **The single-column functions stay** as the documented reference (decision
  [0005](0005-swappable-scheme-interfaces.md): physics stays interrogable). Tests
  check that on horizontally uniform meteorology every column's matrix equals the
  old scheme's bit for bit, and that on a box whose columns differ each column
  agrees with the single-column functions.

**Rationale.** Column-by-column is what the full scheme and FLEXPART do; it is the
only form in which "convection on" means the same thing in all three Tier 2
models. Batching keeps it cheap: the lift is ~151,000 x 40 levels x 30 bisection
steps per event, and the whole event took 78 ms on the GH200 against 63 ms for the
bbox-mean scheme (about 1% of wall, task A02 benchmark); it runs once per met
interval outside the captured step graph
([0004](0004-cuda-graph-static-step-path.md) is untouched). Nearest-column
assignment needs no new interpolation and keeps the matrix exactly non-divergent
for each particle.

**Rejected alternatives.**
- A Python loop over columns: ~10⁵ iterations per event, each a handful of kernel
  launches and host synchronisations.
- Interpolating the matrices between columns for each particle: a blend of
  non-divergent matrices is non-divergent, but it would blur the trigger edge and
  cost four gathers of [Z, Z] per particle.
- Computing the per-particle transition rows directly from the per-column
  entrainment and detrainment vectors without forming [F, Z, Z]: less memory, but
  it hides the matrix the tests and docs are written around. Memory is bounded by
  the firing columns (worst case every column of the EUROPE domain, ~1 GB in
  float32 on the GPU).
- Carrying pressure by hydrostatic integration from surface pressure inside the
  scheme: duplicates what the reader already computes (decision 0010).

**Status.** In force from task A02. Changes footprints wherever convection fires
under the particles; output with `convection.scheme: none` is bit-identical.
See [docs/convection.md](../../docs/convection.md) §1, §7.
