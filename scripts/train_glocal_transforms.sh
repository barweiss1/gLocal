#!/bin/bash
#SBATCH --job-name=glocal_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

if [[ -n "${GLOCAL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$GLOCAL_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ ! -f "$REPO_ROOT/scripts/glocal_transform_sweep.py" ]]; then
  echo "gLocal repository not found at: $REPO_ROOT" >&2
  echo "Submit from the repository root or set GLOCAL_REPO_ROOT." >&2
  exit 1
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

: "${GLOCAL_SWEEP_CONFIG:?Set GLOCAL_SWEEP_CONFIG to the sweep JSON file}"
: "${THINGS_DATA_ROOT:?Set THINGS_DATA_ROOT to the prepared THINGS directory}"
: "${IMAGENET_ROOT:?Set IMAGENET_ROOT to the ImageNet directory}"

for value_name in THINGS_DATA_ROOT IMAGENET_ROOT; do
  value="${!value_name}"
  case "$value" in
    /path/to/*|/path/to)
      echo "$value_name is still a documentation placeholder: $value" >&2
      exit 1
      ;;
  esac
done

case "${MODEL_DICT_PATH:-}" in
  /path/to/*|/path/to|/actual/path/to/*|/actual/path/to)
    echo "Ignoring placeholder MODEL_DICT_PATH: $MODEL_DICT_PATH" >&2
    unset MODEL_DICT_PATH
    ;;
esac

THINGS_FEATURES="${THINGS_FEATURES:-$THINGS_DATA_ROOT/features.pkl}"
PROBING_BASE="${PROBING_BASE:-$REPO_ROOT/glocal-transform-training}"

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

if [[ ! -f "$GLOCAL_SWEEP_CONFIG" ]]; then
  echo "Sweep configuration not found: $GLOCAL_SWEEP_CONFIG" >&2
  exit 1
fi

TASK_COUNT="$(
  "$PYTHON_BIN" scripts/glocal_transform_sweep.py count \
    --config "$GLOCAL_SWEEP_CONFIG"
)"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
  echo "Expected SLURM_ARRAY_TASK_ID in [0, $((TASK_COUNT - 1))], got $TASK_ID" >&2
  exit 1
fi

BURNIN="${BURNIN:-20}"
PATIENCE="${PATIENCE:-20}"
if (( PATIENCE < BURNIN )); then
  echo "PATIENCE ($PATIENCE) must be greater than or equal to BURNIN ($BURNIN)." >&2
  echo "Lightning 1.8 can repeat validation indefinitely before min_epochs." >&2
  exit 1
fi

NUM_PROCESSES="${NUM_PROCESSES:-4}"
N_OBJECTS="${N_OBJECTS:-1854}"
SCRATCH_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

RUN_ARGS=(
  --config "$GLOCAL_SWEEP_CONFIG"
  --task-id "$TASK_ID"
  --repo-root "$REPO_ROOT"
  --data-root "$THINGS_DATA_ROOT"
  --features "$THINGS_FEATURES"
  --imagenet-root "$IMAGENET_ROOT"
  --probing-base "$PROBING_BASE"
  --scratch-root "$SCRATCH_ROOT"
  --device gpu
  --num-processes "$NUM_PROCESSES"
  --n-objects "$N_OBJECTS"
  --burnin "$BURNIN"
  --patience "$PATIENCE"
)
if [[ -n "${MODEL_DICT_PATH:-}" ]]; then
  RUN_ARGS+=(--model-dict "$MODEL_DICT_PATH")
fi

"$PYTHON_BIN" scripts/glocal_transform_sweep.py run "${RUN_ARGS[@]}"
