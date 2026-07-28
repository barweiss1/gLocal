#!/bin/bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: scripts/submit_imagenet_clip_features.sh CONFIG [sbatch options...]" >&2
  exit 2
fi

CONFIG_INPUT="$1"
shift
REPO_ROOT="${GLOCAL_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$CONFIG_INPUT" = /* ]]; then
  GLOCAL_SWEEP_CONFIG="$CONFIG_INPUT"
else
  GLOCAL_SWEEP_CONFIG="$(cd "$(dirname "$CONFIG_INPUT")" && pwd)/$(basename "$CONFIG_INPUT")"
fi

TASK_COUNT="$(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/precompute_imagenet_clip_features.py" count \
    --config "$GLOCAL_SWEEP_CONFIG"
)"
LAST_TASK=$((TASK_COUNT - 1))
ARRAY_SPEC="${IMAGENET_FEATURE_ARRAY_OVERRIDE:-0-${LAST_TASK}%1}"

export GLOCAL_REPO_ROOT="$REPO_ROOT"
export GLOCAL_SWEEP_CONFIG
export PYTHON_BIN

echo "Submitting $TASK_COUNT ImageNet feature tasks as array $ARRAY_SPEC" >&2
sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL \
  "$@" \
  "$REPO_ROOT/scripts/prepare_imagenet_clip_features.sh"
