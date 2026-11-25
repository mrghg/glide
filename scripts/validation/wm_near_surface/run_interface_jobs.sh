#!/bin/bash
# Run interface_study.py jobs from a job file, P at a time (CPU only).
# Each job line: <src root> <interface_study.py args...>; <src root>/src goes
# first on PYTHONPATH (this branch, PR #69's head or main), and the study
# script is taken from $SDIR (default: the directory of this file).
# Usage: run_interface_jobs.sh <jobfile> <P>
JOBS=$1
P=${2:-10}
export PY=${GLIDE_PYTHON:-.venv/bin/python}
export SDIR=${SDIR:-$(cd "$(dirname "$0")" && pwd)}
run_one() {
	src=$1
	shift
	CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
		PYTHONPATH=$src/src:$SDIR "$PY" "$SDIR/interface_study.py" "$@"
}
export -f run_one
xargs -P "$P" -L 1 bash -c 'run_one "$@"' _ <"$JOBS"
echo JOBS_DONE
