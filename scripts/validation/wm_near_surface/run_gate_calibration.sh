#!/bin/bash
# Run bl_top_gate_calibration.py jobs, P at a time (CPU only).
# Job line: <src dir (holds lpdm/)> <label> <n_substeps or -> <regime> <seed>
# The calibration script and tests/test_bl_top_well_mixed.py are taken from
# $G (default: the job file's directory), results go to $G/gate.jsonl.
# Usage: run_gate_calibration.sh <jobfile> [P]
JOBS=$1
export G=${G:-$(cd "$(dirname "$JOBS")" && pwd)}
export PY=${GLIDE_PYTHON:-.venv/bin/python}
one() {
	src=$1
	label=$2
	n=$3
	r=$4
	s=$5
	nn=()
	[ "$n" != "-" ] && nn=(--n "$n")
	CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$src:$G "$PY" "$G/bl_top_gate_calibration.py" \
		--regime "$r" --seed "$s" "${nn[@]}" --label "$label" --out "$G/gate.jsonl"
}
export -f one
xargs -P "${2:-6}" -L 1 bash -c 'one "$@"' _ <"$JOBS"
echo JOBS_DONE
