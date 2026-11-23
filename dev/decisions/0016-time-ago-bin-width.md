# 0016 — Time-ago bins of a configured width, covering the run

**Context.** Footprints can be resolved by age: bin *k* of the `time_ago` axis
holds residence at ages, counted back from the release window's end, in
[k, k+1) bin widths. The runtime indexed bins hourly, but the output labels
said each was `simulation.length_seconds / n_time_bins` wide. With more than one
bin but fewer bins than hours, the labels were wrong and older residence was
dropped without a word (physics specification review of 22 Aug 2026, finding
PR-F1). PR #41 (23 Nov 2026) made the labels hourly, and added bins for the
release window: every particle is now followed for the full length from its own
release, so ages reach the length plus the window.

**Decision.**
- **The width is a setting,** `output_grid.time_bin_seconds` (default 3,600 s).
  The runtime's index, the axis length and the labels all use it, so they
  cannot disagree. It is fixed for a run, a constant to the compiled step
  (decision 0004).
- **`n_time_bins` counts bins of travel;** the axis adds
  ceil(longest release window / width) bins, so a particle released at the
  start of its window keeps all its travel on the axis.
- **The bins must cover the run.** With more than one bin,
  `n_time_bins × time_bin_seconds` must be at least the simulation length.
  A configuration that falls short is refused at load, unless it sets
  `output_grid.drop_older_ages: true`; then older residence is dropped, as
  asked. Dropping is never silent.
- **One bin is the time-integrated footprint,** whatever its width: every age
  goes into it, and its label runs from 0 to the length plus the longest
  window.

**Rationale.** Coarser bins (daily bins on a multi-week run) keep the footprint
store small, which the storage work invites; they need a width that is not an
hour. Deriving the index and the labels from one value removes the original
inconsistency. Refusing a short axis, rather than dropping residence, makes the
loss a visible choice: a footprint that silently lacks its oldest days looks
like a complete one.

**Rejected alternatives.**
- *Width = length / n_time_bins, as the labels said.* Ties the width to the
  run length, so the same bins mean different ages in runs of different
  lengths, and it cannot add the release-window bins cleanly.
- *Clamp old ages into the last bin.* Keeps the total but makes the last bin's
  label false.
- *Drop silently (the old behaviour).* The defect this decision removes.

**Status.** Adopted 2026-11-23 (task A00). Shipped configurations use one bin
or hourly bins covering the run, so their output is unchanged. Docs:
[docs/physics.md](../../docs/physics.md) §7; README "Outputs".
