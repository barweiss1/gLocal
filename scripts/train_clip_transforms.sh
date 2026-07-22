#!/bin/bash
#SBATCH --job-name=clip_transform
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-31%4
#SBATCH --time=2-00:00:00

set -euo pipefail

if [[ -n "${GLOCAL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$GLOCAL_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ ! -f "$REPO_ROOT/scripts/clip_transform_sweep.py" ]]; then
  echo "gLocal repository not found at: $REPO_ROOT" >&2
  echo "Submit from the repository root or set GLOCAL_REPO_ROOT explicitly." >&2
  exit 1
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

: "${THINGS_DATA_ROOT:?Set THINGS_DATA_ROOT to the prepared THINGS directory}"

case "$THINGS_DATA_ROOT" in
  /path/to/*|/path/to)
    echo "THINGS_DATA_ROOT is still a documentation placeholder: $THINGS_DATA_ROOT" >&2
    echo "Set it to the writable path used by the preparation job." >&2
    exit 1
    ;;
esac

case "${THINGS_FEATURES:-}" in
  /path/to/*|/path/to)
    echo "Ignoring placeholder THINGS_FEATURES: $THINGS_FEATURES" >&2
    unset THINGS_FEATURES
    ;;
esac

THINGS_FEATURES="${THINGS_FEATURES:-$THINGS_DATA_ROOT/features.pkl}"

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

for required in \
  "$THINGS_DATA_ROOT/triplets/train_90.npy" \
  "$THINGS_DATA_ROOT/triplets/test_10.npy" \
  "$THINGS_FEATURES"; do
  if [[ ! -f "$required" ]]; then
    echo "Required training input not found: $required" >&2
    echo "Prepare it first with scripts/prepare_things_clip.py (see the runbook)." >&2
    exit 1
  fi
done

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= 32 )); then
  echo "Expected SLURM_ARRAY_TASK_ID in [0, 31], got $TASK_ID" >&2
  exit 1
fi

PROBING_BASE="${PROBING_BASE:-$REPO_ROOT/clip-transform-training}"
case "$PROBING_BASE" in
  /path/to/*|/path/to)
    echo "PROBING_BASE is still a placeholder: $PROBING_BASE" >&2
    echo "Set it to a writable output directory before submitting." >&2
    exit 1
    ;;
esac
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
BURNIN="${BURNIN:-15}"
PATIENCE="${PATIENCE:-10}"
SIGMA="${SIGMA:-0.001}"

SCRATCH_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

"$PYTHON_BIN" scripts/clip_transform_sweep.py run \
  --task-id "$TASK_ID" \
  --repo-root "$REPO_ROOT" \
  --data-root "$THINGS_DATA_ROOT" \
  --features "$THINGS_FEATURES" \
  --probing-base "$PROBING_BASE" \
  --scratch-root "$SCRATCH_ROOT" \
  --device gpu \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --burnin "$BURNIN" \
  --patience "$PATIENCE" \
  --sigma "$SIGMA"
