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

if [[ -n "${GLOCAL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$GLOCAL_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ ! -f "$REPO_ROOT/scripts/prepare_things_clip.py" ]]; then
  echo "gLocal repository not found at: $REPO_ROOT" >&2
  echo "Submit from the repository root or set GLOCAL_REPO_ROOT explicitly." >&2
  exit 1
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

: "${THINGS_DATA_ROOT:?Set THINGS_DATA_ROOT to the desired THINGS directory}"

case "$THINGS_DATA_ROOT" in
  /path/to/*|/path/to)
    echo "THINGS_DATA_ROOT is still a documentation placeholder: $THINGS_DATA_ROOT" >&2
    echo "Set it to a writable path, for example:" >&2
    echo "  export THINGS_DATA_ROOT=/raid/home/barweiss/datasets/things-clip-training" >&2
    exit 1
    ;;
esac

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

if [[ -n "${PREP_FEATURES_OUTPUT:-}" ]]; then
  case "$PREP_FEATURES_OUTPUT" in
    /path/to/*|/path/to)
      echo "PREP_FEATURES_OUTPUT is still a placeholder: $PREP_FEATURES_OUTPUT" >&2
      exit 1
      ;;
  esac
  ARGS+=(--features "$PREP_FEATURES_OUTPUT")
fi

if [[ "${USE_EXISTING_THINGS_IMAGES:-0}" == "1" ]]; then
  ARGS+=(--use-existing-images)
fi

echo "Preparing THINGS data in: $THINGS_DATA_ROOT"
echo "Python: $PYTHON_BIN"
echo "Device: $DEVICE"

"$PYTHON_BIN" scripts/prepare_things_clip.py "${ARGS[@]}"
