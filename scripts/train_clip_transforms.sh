#!/bin/bash
#SBATCH --job-name=clip_transform
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-7%4
#SBATCH --time=2-00:00:00

set -euo pipefail

if [[ -n "${GLOCAL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$GLOCAL_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ ! -f "$REPO_ROOT/main_global_probing.py" ]]; then
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

MODELS=(
  "clip_RN50"
  "clip_ViT-L/14"
  "OpenCLIP_ViT-L-14_laion400m_e32"
  "OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
)
SLUGS=(
  "clip_RN50"
  "clip_ViT-L-14"
  "OpenCLIP_ViT-L-14_laion400m_e32"
  "OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
)
KINDS=("naive" "global")
REGULARIZATIONS=("l2" "eye")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= 8 )); then
  echo "Expected SLURM_ARRAY_TASK_ID in [0, 7], got $TASK_ID" >&2
  exit 1
fi

MODEL_INDEX=$((TASK_ID % 4))
KIND_INDEX=$((TASK_ID / 4))
MODEL="${MODELS[$MODEL_INDEX]}"
MODEL_SLUG="${SLUGS[$MODEL_INDEX]}"
KIND="${KINDS[$KIND_INDEX]}"
REGULARIZATION="${REGULARIZATIONS[$KIND_INDEX]}"

PROBING_BASE="${PROBING_BASE:-$REPO_ROOT/clip-transform-training}"
case "$PROBING_BASE" in
  /path/to/*|/path/to)
    echo "PROBING_BASE is still a placeholder: $PROBING_BASE" >&2
    echo "Set it to a writable output directory before submitting." >&2
    exit 1
    ;;
esac
MODULE="${MODULE:-penultimate}"
N_FOLDS="${N_FOLDS:-3}"
LMBDA="${LMBDA:-0.001}"
OPTIM="${OPTIM:-Adam}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
BURNIN="${BURNIN:-15}"
PATIENCE="${PATIENCE:-15}"
SIGMA="${SIGMA:-0.001}"

if (( PATIENCE < BURNIN )); then
  echo "PATIENCE ($PATIENCE) must be at least BURNIN ($BURNIN)." >&2
  echo "With Lightning 1.8, an earlier stop signal can repeatedly trigger validation while min_epochs blocks exit." >&2
  exit 1
fi

OPTIM_LOWER="${OPTIM,,}"
TASK_ROOT="$PROBING_BASE/$KIND/$MODEL_SLUG"
FEATURE_DIR="$TASK_ROOT/embeddings"
FEATURE_PATH="$FEATURE_DIR/features.pkl"
OUTPUT_PATH="$TASK_ROOT/results/custom/$MODEL/$MODULE/$N_FOLDS/$LMBDA/$OPTIM_LOWER/$LEARNING_RATE/transform.npz"

if [[ -f "$OUTPUT_PATH" ]]; then
  echo "Transform already exists; skipping: $OUTPUT_PATH"
  exit 0
fi

mkdir -p "$FEATURE_DIR" "$TASK_ROOT/checkpoints"
if [[ ! -e "$FEATURE_PATH" ]]; then
  ln -s "$THINGS_FEATURES" "$FEATURE_PATH"
fi

echo "Training $KIND transform"
echo "Model: $MODEL"
echo "Regularization: $REGULARIZATION"
echo "Output: $OUTPUT_PATH"

"$PYTHON_BIN" main_global_probing.py \
  --data_root "$THINGS_DATA_ROOT" \
  --probing_root "$TASK_ROOT" \
  --log_dir "$TASK_ROOT/checkpoints" \
  --model "$MODEL" \
  --source custom \
  --module "$MODULE" \
  --n_folds "$N_FOLDS" \
  --optim "$OPTIM" \
  --learning_rate "$LEARNING_RATE" \
  --regularization "$REGULARIZATION" \
  --lmbda "$LMBDA" \
  --sigma "$SIGMA" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --burnin "$BURNIN" \
  --patience "$PATIENCE" \
  --device gpu \
  --use_bias

if [[ ! -f "$OUTPUT_PATH" ]]; then
  echo "Training completed without producing the expected transform: $OUTPUT_PATH" >&2
  exit 1
fi
echo "Created: $OUTPUT_PATH"
