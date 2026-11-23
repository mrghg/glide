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
| `era5-pressure/EUROPE_202312.zarr` | EUROPE (98° W–39.5° E, 10.6–79.2° N) | 2023-12-01 00:00 – 2023-12-31 23:00 | pressure, 37 | 1 h | 69 GB | Reference config on pressure levels: A01's before-and-after comparison only |
| `era5-pressure/EUROPE_202401.zarr` | EUROPE | 2024-01-01 00:00 – 2024-01-31 23:00 | pressure, 37 | 1 h | 69 GB | Reference config on pressure levels: A01's before-and-after comparison only |
| `era5-model/EUROPE_20231201-20231231_ml.zarr` | EUROPE (98° W–39.5° E, 10.6–79.2° N) | 2023-12-01 00:00 – 2023-12-31 23:00 | model, 137 | 1 h | 244 GB | GLIDE's own download; for B1's equivalence check only; delete after B1|
| `era5-model/EUROPE_20240101-20240131_ml.zarr` | EUROPE (98° W–39.5° E, 10.6–79.2° N) | 2024-01-01 00:00 – 2024-01-31 23:00 | model, 137 | 1 h | 245 GB | GLIDE's own download; for B1's equivalence check only; delete after B1 |
| `um-mk10/EUR_20181222-20190115_UMG-Mk10.zarr` | EUR, UM grid (429 × 499) | 2018-12-22 00:00 – 2019-01-15 00:00 | UM theta levels, 59 | 3 h | 19 GB | Block EUR-Jan on Met Office meteorology (Tier 2); met-bazarr v0.2.0, validation input |
| `um-mk10/EUR_20190621-20190715_UMG-Mk10.zarr` | EUR, UM grid (429 × 499) | 2019-06-21 00:00 – 2019-07-15 00:00 | UM theta levels, 59 | 3 h | 19 GB | Block EUR-Jul on Met Office meteorology (Tier 2); met-bazarr v0.2.0, validation input |
| `um-mk10/NAM_20181222-20190115_UMG-Mk10.zarr` | NAM, UM grid (386 × 464) | 2018-12-22 00:00 – 2019-01-15 00:00 | UM theta levels, 59 | 3 h | 16 GB | Block NAM-Jan on Met Office meteorology (Tier 2); met-bazarr v0.2.0, validation input |
| `um-mk10/NAM_20190621-20190715_UMG-Mk10.zarr` | NAM, UM grid (386 × 464) | 2019-06-21 00:00 – 2019-07-15 00:00 | UM theta levels, 59 | 3 h | 17 GB | Block NAM-Jul on Met Office meteorology (Tier 2); met-bazarr v0.2.0, validation input |
| `era5-arco/EUR_20181222-20190115_ERA5-ARCO.zarr` | EUR (30° W–40° E, 32–72° N) | 2018-12-22 00:00 – 2019-01-15 00:00 | model, 137 | 1 h | 51 GB | Block EUR-Jan; validation input (met-bazarr v0.3.0); one heat-flux value filled |
| `era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr` | EUR | 2019-06-21 00:00 – 2019-07-15 00:00 | model, 137 | 1 h | 50 GB | Block EUR-Jul; validation input (v0.3.0) |
| `era5-arco/NAM_20181222-20190115_ERA5-ARCO.zarr` | NAM (120–55° W, 22–58° N) | 2018-12-22 00:00 – 2019-01-15 00:00 | model, 137 | 1 h | 43 GB | Block NAM-Jan; validation input (v0.3.0) |
| `era5-arco/NAM_20190621-20190715_ERA5-ARCO.zarr` | NAM | 2019-06-21 00:00 – 2019-07-15 00:00 | model, 137 | 1 h | 43 GB | Block NAM-Jul; validation input (v0.3.0) |
| `era5-arco/EUR_19941023-19941028_ERA5-ARCO.zarr` | EUR | 1994-10-23 00:00 – 1994-10-28 00:00 | model, 137 | 1 h | 11 GB | ETEX-1; validation input (v0.3.0) |
| `era5-arco/EUR_19941114-19941119_ERA5-ARCO.zarr` | EUR | 1994-11-14 00:00 – 1994-11-19 00:00 | model, 137 | 1 h | 11 GB | ETEX-2; validation input (v0.3.0) |
| `era5-arco/NAM_19830918-19830922_ERA5-ARCO.zarr` | NAM | 1983-09-18 00:00 – 1983-09-22 00:00 | model, 137 | 1 h | 7.2 GB | CAPTEX-1; validation input (v0.3.0) |
| `era5-arco/NAM_19830925-19830929_ERA5-ARCO.zarr` | NAM | 1983-09-25 00:00 – 1983-09-29 00:00 | model, 137 | 1 h | 7.2 GB | CAPTEX-2; validation input (v0.3.0) |
| `era5-arco/NAM_19831002-19831006_ERA5-ARCO.zarr` | NAM | 1983-10-02 00:00 – 1983-10-06 00:00 | model, 137 | 1 h | 7.2 GB | CAPTEX-3; validation input (v0.3.0) |
| `era5-arco/NAM_19831014-19831017_ERA5-ARCO.zarr` | NAM | 1983-10-14 00:00 – 1983-10-17 00:00 | model, 137 | 1 h | 5.5 GB | CAPTEX-4; validation input (v0.3.0) |
| `era5-arco/NAM_19831026-19831101_ERA5-ARCO.zarr` | NAM | 1983-10-26 00:00 – 1983-11-01 00:00 | model, 137 | 1 h | 11 GB | CAPTEX-5 to 7; validation input (v0.3.0) |
| `era5-arco/NAM_19870105-19870118_ERA5-ARCO.zarr` | NAM | 1987-01-05 00:00 – 1987-01-18 00:00 | model, 137 | 1 h | 24 GB | ANATEX; validation input (v0.3.0) |
| `era5-arco/NAM_19800708-19800712_ERA5-ARCO.zarr` | NAM (full box) | 1980-07-08 00:00 – 1980-07-12 00:00 | model, 137 | 1 h | 7.2 GB | OKC80 (B12); validation input (v0.3.0) |
| `era5-arco/EUROPE_20231201-20231231_ERA5-ARCO.zarr` | EUROPE (98° W–39.5° E, 10.6–79.2° N) | 2023-12-01 00:00 – 2023-12-31 23:00 | model, 137 | 1 h | 214 GB | Reference config, model levels: A01 onward, benchmarks and GPU smoke tests.|
| `era5-arco/EUROPE_20240101-20240131_ERA5-ARCO.zarr` | EUROPE | 2024-01-01 00:00 – 2024-01-31 23:00 | model, 137 | 1 h | 214 GB | Reference config, model levels: A01 onward, benchmarks and GPU smoke tests. |
| `era5-arco/EUR_20190108-20190108_ERA5-ARCO.zarr` | EUR | 2019-01-08 00:00 – 23:00 | model, 137 | 1 h | 2.2 GB | met-bazarr build check only: inside the validation period, not for GLIDE development runs. |

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
