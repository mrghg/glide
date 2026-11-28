#!/bin/bash
# Decision 0033: CPU before (main) / after (this branch) footprint runs, seeds 42 and
# 43, two runs at a time (3 threads each), plus the legacy options on this branch
# for the bit-identity check against main. Outputs: outputs/agents/CONV/pr/{before,after}/.
# usage: MAIN_SRC=<main's src> run_before_after.sh
set -e
H=$(cd "$(dirname "$0")" && pwd)
: "${MAIN_SRC:?set MAIN_SRC to the src directory of main}"
R=$H/run_footprints.sh
SRC=$MAIN_SRC OUT=pr/before $R smoke default &
OUT=pr/after $R smoke default &
wait
SRC=$MAIN_SRC OUT=pr/before $R smoke default_s43 &
OUT=pr/after $R smoke default_s43 &
wait
OUT=pr/after $R smoke legacy &
wait
SRC=$MAIN_SRC OUT=pr/before $R jfj default &
OUT=pr/after $R jfj default &
wait
SRC=$MAIN_SRC OUT=pr/before $R jfj default_s43 &
OUT=pr/after $R jfj default_s43 &
wait
echo ALL-DONE
