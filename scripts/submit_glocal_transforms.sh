#!/bin/bash
# Submit the configured gLocal sweep with a dynamically sized SLURM array.

set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: scripts/submit_glocal_transforms.sh CONFIG [sbatch options...]" >&2
  exit 2
fi

CONFIG_INPUT="$1"
shift

if [[ -n "${GLOCAL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$GLOCAL_REPO_ROOT"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

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

if [[ "$CONFIG_INPUT" = /* ]]; then
  GLOCAL_SWEEP_CONFIG="$CONFIG_INPUT"
else
  GLOCAL_SWEEP_CONFIG="$(cd "$(dirname "$CONFIG_INPUT")" && pwd)/$(basename "$CONFIG_INPUT")"
fi
if [[ ! -f "$GLOCAL_SWEEP_CONFIG" ]]; then
  echo "Sweep configuration not found: $GLOCAL_SWEEP_CONFIG" >&2
  exit 1
fi

TASK_COUNT="$(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/glocal_transform_sweep.py" count \
    --config "$GLOCAL_SWEEP_CONFIG"
)"
LAST_TASK=$((TASK_COUNT - 1))
MAX_PARALLEL="${MAX_PARALLEL:-4}"
if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be at least 1, got $MAX_PARALLEL" >&2
  exit 1
fi
ARRAY_SPEC="${GLOCAL_ARRAY_OVERRIDE:-0-${LAST_TASK}%${MAX_PARALLEL}}"

export GLOCAL_REPO_ROOT
export GLOCAL_SWEEP_CONFIG
export PYTHON_BIN

echo "Submitting $TASK_COUNT gLocal tasks as array $ARRAY_SPEC" >&2
sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL \
  "$@" \
  "$REPO_ROOT/scripts/train_glocal_transforms.sh"
