#!/usr/bin/env bash
set -euo pipefail

# One-command local/bootstrap setup for GLIDE across machines.
# Usage:
#   ./scripts/setup.sh                          # core install only
#   ./scripts/setup.sh --with-viz               # core + notebook/plotting stack
#   ./scripts/setup.sh --with-viz --run-tests   # full + run physics tests
#   ./scripts/setup.sh --python 3.11 --run-tests
#
# Dependencies are defined in pyproject.toml; this script just glues uv / pip
# around the right extras. The `[viz]` extras pull in cartopy (via geoviews),
# which is C-extension heavy and may need `python3-dev` + `libgeos-dev` on the
# system if no binary wheel is available for your (cpython × OS × arch).

PYTHON_BIN="python3.11"
RUN_TESTS="false"
WITH_VIZ="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --run-tests)
      RUN_TESTS="true"
      shift
      ;;
    --with-viz)
      WITH_VIZ="true"
      shift
      ;;
    -h|--help)
      echo "Usage: ./scripts/setup.sh [--python <python-bin>] [--with-viz] [--run-tests]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./scripts/setup.sh [--python <python-bin>] [--with-viz] [--run-tests]"
      exit 1
      ;;
  esac
done

if [[ ! -f pyproject.toml ]]; then
  echo "ERROR: Run this script from the repository root (pyproject.toml not found)."
  exit 1
fi

# Pick the install spec. `--run-tests` implies the `dev` extras (pytest).
EXTRAS=""
if [[ "$WITH_VIZ" == "true" && "$RUN_TESTS" == "true" ]]; then
  EXTRAS="[viz,dev]"
elif [[ "$WITH_VIZ" == "true" ]]; then
  EXTRAS="[viz]"
elif [[ "$RUN_TESTS" == "true" ]]; then
  EXTRAS="[dev]"
fi
INSTALL_SPEC="-e .${EXTRAS}"

if command -v uv >/dev/null 2>&1; then
  echo "==> Using uv for environment setup"
  uv venv --python "$PYTHON_BIN"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install $INSTALL_SPEC
else
  echo "==> uv not found, falling back to stdlib venv + pip"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable '$PYTHON_BIN' not found."
    echo "Install Python 3.11+ or rerun with --python <path-or-command>."
    exit 1
  fi
  "$PYTHON_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install $INSTALL_SPEC
fi

echo "==> Environment ready: $(pwd)/.venv  (extras: ${EXTRAS:-none})"

if [[ "$RUN_TESTS" == "true" ]]; then
  echo "==> Running physics tests"
  PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_physics.py
fi

echo "==> Done"
echo "Next: source .venv/bin/activate"
