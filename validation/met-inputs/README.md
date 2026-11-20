# Meteorology inputs for GLIDE's validation

The recipes, job scripts and checks that build GLIDE's validation meteorology
with [met-bazarr](https://github.com/mrghg/met-bazarr). met-bazarr is the
generic tool; what to build, where and how for GLIDE lives here, next to the
validation plan that defines it ([docs/validation-plan.md](../../docs/validation-plan.md)
§4). Moved here from met-bazarr on 2026-11-20.

| Folder | Contents |
| --- | --- |
| `era5-arco/recipes/` | ERA5 from the ARCO copy on Google Cloud (source `era5_arco`): the 13 windows of validation-plan §4 (case-library blocks EUR/NAM × Jan/Jul 2019; ETEX-1, -2; CAPTEX-1 to 7; ANATEX; OKC80 over the full NAM box), EUROPE Dec 2023 and Jan 2024 for task A01, and a test day |
| `um/recipes/` | The Met Office global UM analyses NAME runs on (series `UMG_Mk10PT`, source `um_global`): the four case-library blocks and a test day |
| `jobs/isambard/build_era5.sh` | Builds ERA5 stores on Isambard AI, filling each store's times with several processes at once |
| `jobs/lotus/build_um_store.sbatch` | Builds one UM store on JASMIN's LOTUS2 |
| `checks/check_era5_equivalence.py` | Compares an ERA5 store with one from GLIDE's own downloader (`scripts/download_era5.sh`) |
| `checks/check_um_store.py` | Checks a UM store against its source files at chosen times |

## Validation inputs come from a met-bazarr release

Validation inputs are built from a **tagged met-bazarr release**, with
`--validation-input`, and the pre-registration pins that release (task A8b).
The job scripts run met-bazarr from its own checkout, `$METBAZARR_REPO`
(default `~/code/met-bazarr`), and print the tag they used:

```bash
git -C ~/code/met-bazarr fetch --tags && git -C ~/code/met-bazarr checkout v0.3.0
```

Releases used: the UM stores from **v0.2.0** (19 Nov 2026); the ERA5 stores
from **v0.3.0** (rebuilt 20 Nov 2026, after the v0.2.0 build stalled; no values
differ between the two releases).

The recipes here and the met-bazarr release are a pair: a recipe uses
met-bazarr's recipe keys and source types, so check that a recipe still loads
(`metbazarr build <recipe> --step plan`) after moving to a newer release.

Stores are named `<DOMAIN>_<first date>-<last date>_<SOURCE>.zarr`
(`_ERA5-ARCO`, `_UMG-Mk10`; ERA5 straight from ECMWF will be `_ERA5-ECMWF`).
Recipes write to `$METBAZARR_OUT` (ERA5) or JASMIN scratch (UM).

## ERA5 from ARCO on Isambard AI

```bash
(cd ~/code/met-bazarr && uv sync --extra arco)    # gcsfs, for the public gs:// stores
export METBAZARR_OUT=/projects/<project>/glide-data/era5-arco
R=validation/met-inputs/era5-arco/recipes
~/code/met-bazarr/.venv/bin/metbazarr build $R/EUR-day-20190108.yaml --step plan
FINISH_FLAGS=--validation-input sbatch validation/met-inputs/jobs/isambard/build_era5.sh $R/*.yaml
```

Run from the GLIDE root. Only a person submits jobs, and on Isambard AI
downloads must not run on the login node. Each 24-day window takes about an
hour with the default 8 fill processes. EUR-Jan meets one missing heat-flux
value in ARCO (2018-12-28 21:00), which the fill rule fills and records.

## Met Office UM on JASMIN

Reading the Met Office's WGDOS-packed PP files needs `iris` **and** `mo_pack`,
and `mo_pack` is on conda-forge only. Build met-bazarr's environment on
JASMIN's `jaspy` Python, which has `mo_pack`:

```bash
cd ~/code/met-bazarr
JP=/apps/jasmin/jaspy/miniforge_envs/jaspy3.12/mf3-25.3.0-3/envs/jasml3.12-mf3-25.3.0-3-v20261105/bin/python
uv venv --python "$JP" --system-site-packages .venv-jaspy
UV_PROJECT_ENVIRONMENT=.venv-jaspy uv sync --python "$JP" --extra um
```

`--python` matters: without it `uv sync` follows `.python-version` and replaces
the environment with one that cannot see `mo_pack`. Reading the NAME met archive
needs membership of the `gws_name` group. Then, from the GLIDE root:

```bash
R=validation/met-inputs/um/recipes
~/code/met-bazarr/.venv-jaspy/bin/metbazarr build $R/EUR-day-20190108.yaml --step plan
sbatch --account=<account> validation/met-inputs/jobs/lotus/build_um_store.sbatch $R/EUR-Jan.yaml --validation-input
~/code/met-bazarr/.venv-jaspy/bin/python validation/met-inputs/checks/check_um_store.py $R/EUR-day-20190108.yaml --times 0 4
```

Met Office data, and anything derived from it, never goes into a repository, a
test or an issue. Copy the finished stores to `$GLIDE_DATA/um-mk10/` on Isambard
and check them against their sha256 lists.

## Where the decisions are

- The conversions: met-bazarr decisions 0003 (UM), 0006 (ERA5 from ARCO), 0007
  (UM stresses and hill drag).
- The format they write: [docs/met_schema.md](../../docs/met_schema.md) (format
  2.0), GLIDE decisions 0019 and 0020.
- What is built and when: [docs/validation-plan.md](../../docs/validation-plan.md)
  §4 and §8; the data inventory in `dev/agent/PROGRESS.md`.
