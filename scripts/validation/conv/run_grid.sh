#!/bin/bash
# Task CONV: closure variants vs vertical grid and cadence; January (reference
# config) and July 2019. Run from anywhere; outputs under the main checkout's
# outputs/agents/CONV/grid.
# usage: run_grid.sh <tag> [extra args, e.g. --trigger-dz 150]
set -e
W=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
cd "$MAIN"
O=outputs/agents/CONV/grid
mkdir -p $O
TAG=$1
shift
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=$W/src
JAN="2023-12-28T00 2023-12-28T15 2023-12-29T06 2023-12-29T21 2023-12-30T12 2023-12-31T03 2023-12-31T18 2024-01-01T09 2024-01-02T00"
JUL="2019-06-25T12 2019-06-28T15 2019-07-01T12 2019-07-03T18 2019-07-06T15 2019-07-08T12 2019-07-10T15 2019-07-12T00"
SUMMER="--zarr-store data/era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr --met-domain -30 40 32 72"
S=$W/scripts/validation/conv
.venv/bin/python $S/closure_grid_cadence.py $W/configs/ab_multisite_perf.yaml $O/jan_$TAG.npz --times $JAN --nlev 23 40 80 "$@"
.venv/bin/python $S/closure_grid_cadence.py $W/configs/ab_multisite_perf.yaml $O/jul_$TAG.npz --times $JUL --nlev 23 40 80 $SUMMER "$@"
.venv/bin/python $S/summarize_grid_cadence.py $O/jan_$TAG.npz > $O/jan_$TAG.txt
.venv/bin/python $S/summarize_grid_cadence.py $O/jul_$TAG.npz > $O/jul_$TAG.txt
echo DONE
