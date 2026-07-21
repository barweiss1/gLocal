#!/bin/bash
#SBATCH --job-name=prepare_things_clip
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${THINGS_DATA_ROOT:?Set THINGS_DATA_ROOT to the desired THINGS directory}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # Keep the explicitly supplied interpreter.
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="python3"
fi

if ! command -v "$PYTHON_BIN" >/dev/null; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  echo "Submit with PYTHON_BIN set to your conda environment's Python." >&2
  exit 1
fi

DEVICE="${DEVICE:-cuda}"
PREP_BATCH_SIZE="${PREP_BATCH_SIZE:-32}"

ARGS=(
  --data-root "$THINGS_DATA_ROOT"
  --device "$DEVICE"
  --batch-size "$PREP_BATCH_SIZE"
)

if [[ -n "${THINGS_FEATURES:-}" ]]; then
  ARGS+=(--features "$THINGS_FEATURES")
fi

if [[ "${USE_EXISTING_THINGS_IMAGES:-0}" == "1" ]]; then
  ARGS+=(--use-existing-images)
fi

echo "Preparing THINGS data in: $THINGS_DATA_ROOT"
echo "Python: $PYTHON_BIN"
echo "Device: $DEVICE"

"$PYTHON_BIN" scripts/prepare_things_clip.py "${ARGS[@]}"
