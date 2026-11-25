#!/bin/bash
# CPU smoke run (configs/smoke_mhd_single_release.yaml variants) from a source
# checkout or snapshot, with its own src/ first on PYTHONPATH.
# Usage: run_smoke_cpu.sh <source checkout> <config yaml> <output dir>
set -u
SRC=$1
CFG=$2
OUT=$3
cd "$SRC" || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$SRC/src \
	/usr/bin/time -v "${GLIDE_PYTHON:-$SRC/.venv/bin/python}" -m lpdm.main \
	--config "$CFG" --device cpu --output-uri "$OUT" >"$OUT.log" 2>&1
echo "exit $?" >>"$OUT.log"
