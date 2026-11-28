#!/bin/bash
# Task CONV: build the FLEXPART CONVECT driver. Extracts CONVECT and TLIFT from a
# local FLEXPART checkout's conv_mod.f90 (GPL; not copied into GLIDE) and compiles
# them with qvsat_mod.f90 and driver.f90.
# usage: build.sh <flexpart src dir> <build dir>
set -e
FP=$1
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
awk '/^!\*\*\*\*  *SUBROUTINE CONVECT/{on=1} on{print} /^END SUBROUTINE TLIFT/{exit}' \
    "$FP/conv_mod.f90" > "$B/fp_convect.inc"
cp "$FP/qvsat_mod.f90" "$B/"
cp "$HERE/driver.f90" "$B/"
cd "$B"
gfortran -O2 -ffree-line-length-none -c qvsat_mod.f90
gfortran -O2 -ffree-line-length-none -c driver.f90
gfortran -O2 -o fp_convect qvsat_mod.o driver.o
echo "built $B/fp_convect from $FP (lines: $(wc -l < fp_convect.inc))"
