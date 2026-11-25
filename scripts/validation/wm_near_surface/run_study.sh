#!/bin/bash
# The accuracy, basal-clamp and adaptivity study behind decision 0027.
# Usage: run_study.sh <output dir> [old-source dir] [jobs]
# Writes one JSON line per run to <output dir>/<group>.jsonl (fixcount_study.py),
# running <jobs> runs at a time (2 threads each). <old-source dir> is a checkout
# of the code before decision 0027 (for the old-code rows); omit to skip them.
set -u
OUT=$1
OLD=${2:-}
JOBS=${3:-7}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PY=${GLIDE_PYTHON:-$REPO/.venv/bin/python}
STUDY="$HERE/fixcount_study.py"
SEEDS="1 2 3 4 5"
REGIMES="stable neutral convective"
P=20000
mkdir -p "$OUT"

{
	for r in $REGIMES; do for s in $SEEDS; do
		# Closed column (lid every sub-step), fixed count, and the fine-step references.
		for n in 4 8 16; do
			echo "$REPO/src --regime $r --harness closed --dt 60 --n $n --seed $s --out $OUT/closed.jsonl"
		done
		for dt in 3.75 1.875; do
			echo "$REPO/src --regime $r --harness closed --dt $dt --n 1 --seed $s --out $OUT/closed.jsonl"
		done
		# Open top: particles cross h into free-tropospheric coefficients.
		for n in 4 8 16; do
			echo "$REPO/src --regime $r --harness open --dt 60 --n $n --seed $s --out $OUT/open.jsonl"
		done
		for dt in 3.75 1.875; do
			echo "$REPO/src --regime $r --harness open --dt $dt --n 1 --seed $s --out $OUT/open.jsonl"
		done
		# Basal clamp: zero dsigma_w^2/dz strictly below z_ubl, and the stencil as it was.
		for v in clamp0 fixed_clampoff; do
			echo "$REPO/src --regime $r --harness closed --dt 60 --n 8 --variant $v --seed $s --out $OUT/clamp.jsonl"
			echo "$REPO/src --regime $r --harness closed --dt 1.875 --n 1 --variant $v --seed $s --out $OUT/clamp.jsonl"
		done
		# Adaptivity: k_i = max(n, ceil(dt / (0.125 T_Lw,i))), refinement only.
		for n in 4 8; do
			echo "$REPO/src --regime $r --harness closed --dt 60 --n $n --variant adaptive --adapt-c 0.125 --seed $s --out $OUT/adaptive.jsonl"
		done
		# The code before decision 0027, shipped cap of 6.
		if [ -n "$OLD" ]; then
			echo "$OLD/src --regime $r --harness closed --dt 60 --variant old --max-substeps 6 --seed $s --out $OUT/old.jsonl"
		fi
	done; done
	# Convergence of the reference itself: a still finer outer step.
	for r in $REGIMES; do for s in 1 2 3; do
		echo "$REPO/src --regime $r --harness closed --dt 0.9375 --n 1 --seed $s --out $OUT/closed.jsonl"
		echo "$REPO/src --regime $r --harness open --dt 0.9375 --n 1 --seed $s --out $OUT/open.jsonl"
	done; done
} | xargs -P "$JOBS" -L 1 sh -c 'src=$0; PYTHONPATH=$src '"$PY $STUDY"' --particles '"$P"' --threads 2 "$@"' \
	>>"$OUT/run_study.log" 2>&1
echo "done" >>"$OUT/run_study.log"
