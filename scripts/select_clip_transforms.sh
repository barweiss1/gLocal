#!/bin/bash
#SBATCH --job-name=clip_transform_select
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=high
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail

if [[ -n "${GLOCAL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$GLOCAL_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # Keep the explicit interpreter.
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

PROBING_BASE="${PROBING_BASE:-$REPO_ROOT/clip-transform-training}"
case "$PROBING_BASE" in
  /path/to/*|/path/to)
    echo "PROBING_BASE is still a placeholder: $PROBING_BASE" >&2
    exit 1
    ;;
esac

"$PYTHON_BIN" scripts/clip_transform_sweep.py select \
  --probing-base "$PROBING_BASE"
