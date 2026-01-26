#!/bin/bash
# setup_dev_env.sh: Create Python venv and install dependencies for ensemble pipeline
# Usage: bash scripts/setup_dev_env.sh [--no-system-deps]

set -e

if [[ "$1" != "--no-system-deps" ]]; then
  echo "Installing system dependencies (macOS: brew, Ubuntu: apt-get)..."
  if [[ "$(uname)" == "Darwin" ]]; then
    brew install libomp || true
  else
    sudo apt-get update && sudo apt-get install -y libomp-dev || true
  fi
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

python3 -m pip install --upgrade pip
if [[ -f "tests/requirements.txt" ]]; then
  pip install -r tests/requirements.txt
else
  echo "Warning: tests/requirements.txt not found. Installing core packages."
  pip install torch lightgbm joblib pandas numpy scikit-learn
fi

echo "Environment setup complete."#!/usr/bin/env bash
# scripts/setup_dev_env.sh
# Create a Python virtual environment and install test/dev dependencies.
# Usage: ./scripts/setup_dev_env.sh [--venv-dir .venv] [--no-system-deps]
set -euo pipefail

VENV_DIR=.venv
NO_SYSTEM_DEPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --no-system-deps) NO_SYSTEM_DEPS=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [[ ! -f tests/requirements.txt ]]; then
  echo "Missing tests/requirements.txt — did you pull the latest repo?"
  echo "You can still install individual packages like: pip install joblib pandas numpy"
  exit 1
fi

# On Debian/Ubuntu install libomp-dev for LightGBM if not opted out
if [[ "$NO_SYSTEM_DEPS" -eq 0 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "Installing libomp-dev via apt-get (may require sudo)..."
    sudo apt-get update && sudo apt-get install -y libomp-dev || true
  elif [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v brew >/dev/null 2>&1; then
      echo "Homebrew not found. Install libomp manually with: brew install libomp"
    else
      echo "Installing libomp via brew..."
      brew install libomp || true
    fi
  fi
fi

# Install Python deps (use CPU PyTorch wheel index)
python -m pip install -r tests/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Check for NumPy 2.x incompatibilities with compiled extensions
python - <<'PY'
import sys
import pkgutil
try:
    import numpy as np
    v = tuple(int(x) for x in np.__version__.split('.')[:2])
    if v[0] >= 2:
        print('NumPy >= 2 detected. Some compiled modules may be incompatible.')
        print("If you hit import errors, run: python -m pip install 'numpy<2' --force-reinstall")
except Exception as e:
    print('NumPy import check failed:', e)
    sys.exit(0)
PY

echo "Development environment ready. Activate with: source $VENV_DIR/bin/activate"

echo "Run smoke tests: python3 scripts/run_smoke_tests.py"
