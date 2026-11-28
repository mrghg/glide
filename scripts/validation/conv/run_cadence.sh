#!/bin/bash
# Task CONV: hourly vs 3-hourly convection on consecutive real hours (40 levels), January and July 2019.
set -e
W=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
cd "$MAIN"
O=$MAIN/outputs/agents/CONV/cadence
mkdir -p $O
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=$W/src
S=$W/scripts/validation/conv/cadence_consecutive.py
.venv/bin/python $S $W/configs/ab_multisite_perf.yaml --starts 2023-12-29T06 2023-12-30T12 2023-12-31T03 2024-01-01T09 > $O/jan.log 2>&1
.venv/bin/python $S $W/configs/ab_multisite_perf.yaml --starts 2019-07-01T12 2019-07-06T15 2019-07-08T12 \
    --zarr-store data/era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr --met-domain -30 40 32 72 > $O/jul.log 2>&1
echo DONE
