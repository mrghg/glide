#!/usr/bin/env bash
set -euo pipefail

# One-command local/bootstrap setup for GLIDE across machines.
# Usage:
#   ./scripts/setup.sh
#   ./scripts/setup.sh --run-tests
#   ./scripts/setup.sh --python 3.11 --run-tests

PYTHON_BIN="python3.11"
RUN_TESTS="false"

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
    -h|--help)
      echo "Usage: ./scripts/setup.sh [--python <python-bin>] [--run-tests]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./scripts/setup.sh [--python <python-bin>] [--run-tests]"
      exit 1
      ;;
  esac
done

if [[ ! -f requirements.txt ]]; then
  echo "ERROR: Run this script from the repository root (requirements.txt not found)."
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  echo "==> Using uv for environment setup"
  uv venv --python "$PYTHON_BIN"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
  uv pip install -e .
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
  pip install -r requirements.txt
  pip install -e .
fi

echo "==> Environment ready: $(pwd)/.venv"

if [[ "$RUN_TESTS" == "true" ]]; then
  echo "==> Running physics tests"
  PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_physics.py
fi

echo "==> Done"
echo "Next: source .venv/bin/activate"
