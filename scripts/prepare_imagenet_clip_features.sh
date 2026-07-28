#!/bin/bash
#SBATCH --job-name=imagenet_clip_features
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT="${GLOCAL_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

: "${GLOCAL_SWEEP_CONFIG:?Set GLOCAL_SWEEP_CONFIG}"
: "${IMAGENET_ROOT:?Set IMAGENET_ROOT}"
: "${IMAGENET_FEATURES_BASE:?Set IMAGENET_FEATURES_BASE}"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-128}"
FEATURE_WORKERS="${FEATURE_WORKERS:-2}"

"$PYTHON_BIN" scripts/precompute_imagenet_clip_features.py run \
  --config "$GLOCAL_SWEEP_CONFIG" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --repo-root "$REPO_ROOT" \
  --imagenet-root "$IMAGENET_ROOT" \
  --output-root "$IMAGENET_FEATURES_BASE" \
  --batch-size "$FEATURE_BATCH_SIZE" \
  --workers "$FEATURE_WORKERS" \
  --device cuda
