#!/bin/bash
# Decision 6 study: column statistics of the two triggers, January (reference
# config) and July 2019 samples of task CONV, on 23 / 40 / 80 levels (CPU).
# Outputs: outputs/agents/CONV/trigger/columns/ in the main checkout.
set -e
W=$(cd "$(dirname "$0")/../../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
cd "$MAIN"
O=outputs/agents/CONV/trigger/columns
mkdir -p $O
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 PYTHONPATH=$W/src
JAN="2023-12-28T00 2023-12-28T15 2023-12-29T06 2023-12-29T21 2023-12-30T12 2023-12-31T03 2023-12-31T18 2024-01-01T09 2024-01-02T00"
JUL="2019-06-25T12 2019-06-28T15 2019-07-01T12 2019-07-03T18 2019-07-06T15 2019-07-08T12 2019-07-10T15 2019-07-12T00"
SUMMER="--zarr-store data/era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr --met-domain -30 40 32 72"
S=$W/scripts/validation/conv/trigger
.venv/bin/python $S/trigger_columns.py $W/configs/ab_multisite_perf.yaml $O/jul.npz --times $JUL --nlev 40 23 80 $SUMMER --check outputs/agents/CONV/grid/jul_lcl1.npz
.venv/bin/python $S/trigger_columns.py $W/configs/ab_multisite_perf.yaml $O/jan.npz --times $JAN --nlev 40 23 80 --check outputs/agents/CONV/grid/jan_lcl1.npz
echo DONE
