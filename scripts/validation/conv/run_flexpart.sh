#!/bin/bash
# Task CONV: FLEXPART 11 CONVECT (extracted from a local checkout) on the same
# January and July 2019 hours as run_grid.sh, on the stores' native L137 levels.
# usage: run_flexpart.sh [flexpart src dir]
set -e
W=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
FP=${1:-/home/u6za/chxmr.u6za/code/flexpart/src}
cd "$MAIN"
O=$MAIN/outputs/agents/CONV/flexpart
B=$MAIN/outputs/agents/CONV/flexpart_build
mkdir -p $O
$W/scripts/validation/conv/flexpart_driver/build.sh $FP $B
(cd $FP/.. && git log -1 --format='flexpart commit %h %cd') > $O/flexpart_version.txt
S=$W/scripts/validation/conv/flexpart_convect.py
export OMP_NUM_THREADS=8
EUROPE_BOX="-98 39.5 10.75 79"
.venv/bin/python $S $B/fp_convect $O/jan_dec.npz --zarr-store data/era5-arco/EUROPE_20231201-20231231_ERA5-ARCO.zarr \
    --box $EUROPE_BOX --times 2023-12-28T00 2023-12-28T15 2023-12-29T06 2023-12-29T21 2023-12-30T12 2023-12-31T03 2023-12-31T18
.venv/bin/python $S $B/fp_convect $O/jan_jan.npz --zarr-store data/era5-arco/EUROPE_20240101-20240131_ERA5-ARCO.zarr \
    --box $EUROPE_BOX --times 2024-01-01T09 2024-01-02T00
.venv/bin/python $S $B/fp_convect $O/jul.npz --zarr-store data/era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr \
    --box -30 40 32 72 --times 2019-06-25T12 2019-06-28T15 2019-07-01T12 2019-07-03T18 2019-07-06T15 2019-07-08T12 2019-07-10T15 2019-07-12T00
echo DONE
