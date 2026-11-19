# Shared state

_Maintained by the human operator. Agents read this file and never edit it, so
pull requests never conflict on it. Per-task records are in
[progress/](progress/), one file per task; how task status is derived is in
[INSTRUCTIONS.md](INSTRUCTIONS.md) §1._

## Current state

- Physics tag: none yet (pre-freeze).
- Pre-registration: not yet registered (task A8b, then OSF).
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
| `era5-model/EUR_19941023-19941028_ml.zarr` | EUR (30° W–40° E, 32–72° N) | 1994-10-23 00:00 – 1994-10-28 00:00 | model, 137 | 1 h | 13 GB | ETEX-1 (B10, A11) |
| `era5-model/EUR_19941114-19941119_ml.zarr` | EUR (30° W–40° E, 32–72° N) | 1994-11-14 00:00 – 1994-11-19 00:00 | model, 137 | 1 h | 13 GB | ETEX-2 (B10, A11) |
| `era5-model/NAM_19800708-19800712_ml.zarr` | NAM (120–55° W, 22–58° N) | 1980-07-08 00:00 – 1980-07-12 00:00 | model, 137 | 1 h | 8.3 GB | OKC80 (B12) |
| `era5-model/NAM_19830918-19830922_ml.zarr` | NAM (120–55° W, 22–58° N) | 1983-09-18 00:00 – 1983-09-22 00:00 | model, 137 | 1 h | 8.3 GB | CAPTEX-1 (B10, A11) |
| `era5-model/NAM_19830925-19830929_ml.zarr` | NAM (120–55° W, 22–58° N) | 1983-09-25 00:00 – 1983-09-29 00:00 | model, 137 | 1 h | 8.3 GB | CAPTEX-2 (B10, A11) |
| `era5-model/NAM_19831002-19831006_ml.zarr` | NAM (120–55° W, 22–58° N) | 1983-10-02 00:00 – 1983-10-06 00:00 | model, 137 | 1 h | 8.3 GB | CAPTEX-3 (B10, A11) |
| `era5-model/NAM_19831014-19831017_ml.zarr` | NAM (120–55° W, 22–58° N) | 1983-10-14 00:00 – 1983-10-17 00:00 | model, 137 | 1 h | 6.3 GB | CAPTEX-4 (B10, A11) |
| `era5-model/NAM_19831026-19831101_ml.zarr` | NAM (120–55° W, 22–58° N) | 1983-10-26 00:00 – 1983-11-01 00:00 | model, 137 | 1 h | 13 GB | CAPTEX-5 to 7 (B10, A11) |
| `era5-model/EUR_20181222-20190115_ml.zarr` | EUR (30° W–40° E, 32–72° N) | 2018-12-22 00:00 – 2019-01-15 00:00 | model, 137 | 1 h | 59 GB | Block EUR-Jan (A01 onward, A8a, A10); one heat-flux value filled, see `glide_filled_values` |
  | `era5-model/EUR_20190621-20190715_ml.zarr` | EUR (30° W–40° E, 32–72° N) | 2019-06-21 00:00 – 2019-07-15 00:00 | model, 137 | 1 h | 58 GB | Block EUR-Jul (A8a, A10) |
  | `era5-model/NAM_20181222-20190115_ml.zarr` | NAM (120–55° W, 22–58° N) | 2018-12-22 00:00 – 2019-01-15 00:00 | model, 137 | 1 h | 49 GB |Block NAM-Jan (A8a, A10) |
  | `era5-model/NAM_20190621-20190715_ml.zarr` | NAM (120–55° W, 22–58° N) | 2019-06-21 00:00 – 2019-07-15 00:00 | model, 137 | 1 h | 50 GB | Block NAM-Jul (A8a, A10) |
  | `era5-model/NAM_19870105-19870118_ml.zarr` | NAM (120–55° W, 22–58° N) | 1987-01-05 00:00 – 1987-01-18 00:00 | model, 137 | 1 h | 27 GB | ANATEX (B10, A11) |
| `era5-model/EUROPE_20231201-20231231_ml.zarr` | EUROPE (98° W–39.5° E, 10.6–79.2° N) | 2023-12-01 00:00 – 2023-12-31 23:00 | model, 137 | 1 h | 244 GB | Reference config, model levels (A01 onward); same grid and times as `EUROPE_202312.zarr` |
| `era5-model/EUROPE_20240101-20240131_ml.zarr` | EUROPE (98° W–39.5° E, 10.6–79.2° N) | 2024-01-01 00:00 – 2024-01-31 23:00 | model, 137 | 1 h | 245 GB | Reference config, model levels (A01 onward); same grid and times as `EUROPE_202401.zarr` |

## Reference data outside the repository

_Located by environment variables set in `env.local.sh`. Never copied into the
repository or the tests. Download recipes for the tracer data are in
`data/tracer-experiments.md`._

| Variable | Contents | Needed by |
| --- | --- | --- |
| `GLIDE_FLEXPART_SRC` | FLEXPART v11 source | A8a, A9 |
| `GLIDE_FLEXPART_OUTPUTS` | FLEXPART v11 on ERA5 particle and gridded output for the case library, repeat-seed runs, and run timings | A8a, A10, B11 |
| `GLIDE_NAME_FOOTPRINTS` | NAME on Met Office global analyses: footprints for every case-library release on the EUR and NAM grids, repeat-seed runs, and particle output where available | A8a (Tier 2 ceiling and NAME noise floor), A10 (second reference) |
| `GLIDE_EDGAR` | EDGAR flux maps | A10 |
| `GLIDE_ETEX` | ETEX observations from JRC | B10, A11 |
| `GLIDE_DATEM` | CAPTEX, ANATEX and ACURATE observations, DATEM's copy of ETEX-1, DATEM statistics code and HYSPLIT reference output | B10, A11 |
