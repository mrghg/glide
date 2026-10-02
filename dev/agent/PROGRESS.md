# Shared state

_Maintained by the human operator. Agents read this file and never edit it, so
pull requests never conflict on it. Per-task records are in
[progress/](progress/), one file per task; how task status is derived is in
[INSTRUCTIONS.md](INSTRUCTIONS.md) §1._

## Current state

- Physics tag: none yet (pre-freeze).
- Golden fixtures: none yet (task A0).
- Benchmark baseline: none yet (task A0).
- Notes: none.

## Resume requests

_Numbered and never reused. Format: `R<n> — <ID>: <instruction, or the answer
to a BLOCKED question>`. The next session on that task's queue acts on it and
records the entry ID under `handled:` in the task's progress file. Delete
entries once the task is merged._

_None._

## Data inventory

_Paths are relative to `$GLIDE_DATA`. Agents use only what is listed here and
never download data._

| Path | Domain | Period | Levels | Cadence | Size | Use |
| --- | --- | --- | --- | --- | --- | --- |
| _to fill_ | | | | | | smoke tests |
| _to fill_ | | | | | | reference multi-site benchmark |

## Reference data outside the repository

_Located by environment variables set in `env.local.sh`. Never copied into the
repository or the tests. Download recipes for the tracer data are in
`data/tracer-experiments.md`._

| Variable | Contents | Needed by |
| --- | --- | --- |
| `GLIDE_FLEXPART_SRC` | FLEXPART v11 source | A9 |
| `GLIDE_FLEXPART_OUTPUTS` | FLEXPART v11 on ERA5 particle and gridded output for the case library | A10 |
| `GLIDE_NAME_FOOTPRINTS` | NAME footprints for the same sites and dates | A10 (secondary) |
| `GLIDE_EDGAR` | EDGAR flux maps | A10 |
| `GLIDE_ETEX` | ETEX observations from JRC | B10, A11 |
| `GLIDE_DATEM` | CAPTEX and ANATEX observations, DATEM statistics code and HYSPLIT reference output | B10, A11 |
