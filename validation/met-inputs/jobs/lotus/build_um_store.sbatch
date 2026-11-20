#!/bin/bash
# Build one met-bazarr store on JASMIN's LOTUS2: create, fill every time, then
# check and validate. One CPU; a 193-time UM window takes ~25 min and ~3 GB.
#
# Lives in GLIDE (validation/met-inputs/); runs met-bazarr from its own checkout.
# Submit from the GLIDE repository root on a sci server:
#   export METBAZARR_REPO=~/code/met-bazarr     # a checkout at the pinned tag
#   sbatch --account=<account> validation/met-inputs/jobs/lotus/build_um_store.sbatch \
#       validation/met-inputs/um/recipes/EUR-Jan.yaml [extra finish flags]
# e.g. add --validation-input for a store built from a tagged release.
# An existing store is kept unless OVERWRITE=1 is exported (sbatch --export=ALL,OVERWRITE=1).
#
# Needs met-bazarr's jaspy-based environment .venv-jaspy in $METBAZARR_REPO
# (validation/met-inputs/README.md, "Met Office UM on JASMIN"). The job exits
# non-zero if any step fails or validation fails.
#SBATCH --job-name=metbazarr
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

recipe="${1:?usage: sbatch build_store.sbatch <recipe.yaml> [finish flags]}"
shift
cd "${SLURM_SUBMIT_DIR:-.}"
METBAZARR_REPO="${METBAZARR_REPO:-${HOME}/code/met-bazarr}"
MB="${METBAZARR_REPO}/.venv-jaspy/bin/metbazarr"

echo "host $(hostname), glide $(git rev-parse --short HEAD), met-bazarr $(git -C "${METBAZARR_REPO}" describe --tags --always --dirty), recipe ${recipe}, start $(date -u +%FT%TZ)"
"${MB}" build "${recipe}" --step create ${OVERWRITE:+--overwrite}
"${MB}" build "${recipe}" --step fill
"${MB}" build "${recipe}" --step finish "$@"
echo "done $(date -u +%FT%TZ)"
