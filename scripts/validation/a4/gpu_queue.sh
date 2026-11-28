#!/bin/bash
# Task A4 GPU A/B queue on configs/ab_multisite_perf.yaml: each run takes the
# shared GPU lock in turn (one GPU job at a time, ~20 min and ~54 GB host each).
#
#   gpu_queue.sh <out dir> <main checkout> <A4 checkout> <run>...
#
# <run> is one of the cases below, with an
# optional suffix after a dot for repeats (e.g. main-s42.r2). The A4 checkout's
# configs/ holds ab_multisite_perf_{bf16,seed43}.yaml (copies of the shipped
# config with one line changed each); the main checkout's holds _seed43.
set -u
OUT=$1; MAIN=$2; A4=$3; shift 3
LOCK=/home/u6za/chxmr.u6za/code/glide/outputs/agents/gpu.lock
BENCH=$(dirname "$(readlink -f "$0")")/run_bench.sh
for run in "$@"; do
  base=${run%%.*}; extra=""
  case $base in
    main-s42) dir=$MAIN; cfg=configs/ab_multisite_perf.yaml ;;
    main-s43) dir=$MAIN; cfg=configs/ab_multisite_perf_seed43.yaml ;;
    a4-fp32-s42) dir=$A4; cfg=configs/ab_multisite_perf.yaml ;;
    a4-bf16-s42) dir=$A4; cfg=configs/ab_multisite_perf_bf16.yaml ;;
    a4-fp32-s43) dir=$A4; cfg=configs/ab_multisite_perf_seed43.yaml ;;
    # 12 h, 12-release cut (configs/ab_cut12h*.yaml), eager (GLIDE_COMPILE=0): the
    # same random stream in every variant, so differences are the change itself.
    cut-main-eager) dir=$MAIN; cfg=configs/ab_cut12h.yaml; extra=GLIDE_COMPILE=0 ;;
    cut-a4-degrees-eager) dir=$A4; cfg=configs/ab_cut12h.yaml
      extra="GLIDE_COMPILE=0 GLIDE_POSITION_FRAME=degrees" ;;
    cut-a4-fp32-eager) dir=$A4; cfg=configs/ab_cut12h.yaml; extra=GLIDE_COMPILE=0 ;;
    cut-a4-bf16-eager) dir=$A4; cfg=configs/ab_cut12h_bf16.yaml; extra=GLIDE_COMPILE=0 ;;
    *) echo "unknown run $run"; continue ;;
  esac
  name=ref-$run
  [ -e "$OUT/$name" ] && { echo "skip $name (exists)"; continue; }
  # A4_HOLD_LOCK=1: the caller already holds the lock for the whole group (runs
  # compared for timing go back to back, under the same conditions).
  if [ "${A4_HOLD_LOCK:-0}" = 1 ]; then
    # shellcheck disable=SC2086
    "$BENCH" "$dir" "$OUT" "$name" A4_CONFIG=$cfg $extra
  else
    # shellcheck disable=SC2086
    flock "$LOCK" "$BENCH" "$dir" "$OUT" "$name" A4_CONFIG=$cfg $extra
  fi
done
