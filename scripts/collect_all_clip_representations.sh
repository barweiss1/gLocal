#!/bin/bash
#SBATCH --job-name=clip_features_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
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

if [[ ! -f "$REPO_ROOT/scripts/collect_representations.py" ]]; then
  echo "gLocal repository not found at: $REPO_ROOT" >&2
  echo "Submit from the repository root or set GLOCAL_REPO_ROOT explicitly." >&2
  exit 1
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
MODULE="${MODULE:-penultimate}"
N_FOLDS="${N_FOLDS:-3}"
LMBDA="${LMBDA:-0.001}"
OPTIM="${OPTIM:-Adam}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
OPTIM_LOWER="${OPTIM,,}"

transform_path() {
  local kind="$1"
  local slug="$2"
  local model="$3"
  echo "$PROBING_BASE/$kind/$slug/results/custom/$model/$MODULE/$N_FOLDS/$LMBDA/$OPTIM_LOWER/$LEARNING_RATE/transform.npz"
}

export CLIP_RN50_NAIVE_TRANSFORM
CLIP_RN50_NAIVE_TRANSFORM="$(transform_path naive clip_RN50 clip_RN50)"
export CLIP_RN50_GLOBAL_TRANSFORM
CLIP_RN50_GLOBAL_TRANSFORM="$(transform_path global clip_RN50 clip_RN50)"

export CLIP_VIT_L14_NAIVE_TRANSFORM
CLIP_VIT_L14_NAIVE_TRANSFORM="$(transform_path naive clip_ViT-L-14 'clip_ViT-L/14')"
export CLIP_VIT_L14_GLOBAL_TRANSFORM
CLIP_VIT_L14_GLOBAL_TRANSFORM="$(transform_path global clip_ViT-L-14 'clip_ViT-L/14')"

export OPENCLIP_LAION400M_NAIVE_TRANSFORM
OPENCLIP_LAION400M_NAIVE_TRANSFORM="$(transform_path naive OpenCLIP_ViT-L-14_laion400m_e32 OpenCLIP_ViT-L-14_laion400m_e32)"
export OPENCLIP_LAION400M_GLOBAL_TRANSFORM
OPENCLIP_LAION400M_GLOBAL_TRANSFORM="$(transform_path global OpenCLIP_ViT-L-14_laion400m_e32 OpenCLIP_ViT-L-14_laion400m_e32)"

export OPENCLIP_LAION2B_NAIVE_TRANSFORM
OPENCLIP_LAION2B_NAIVE_TRANSFORM="$(transform_path naive OpenCLIP_ViT-L-14_laion2b_s32b_b82k OpenCLIP_ViT-L-14_laion2b_s32b_b82k)"
export OPENCLIP_LAION2B_GLOBAL_TRANSFORM
OPENCLIP_LAION2B_GLOBAL_TRANSFORM="$(transform_path global OpenCLIP_ViT-L-14_laion2b_s32b_b82k OpenCLIP_ViT-L-14_laion2b_s32b_b82k)"

TRANSFORM_PATHS=(
  "$CLIP_RN50_NAIVE_TRANSFORM"
  "$CLIP_RN50_GLOBAL_TRANSFORM"
  "$CLIP_VIT_L14_NAIVE_TRANSFORM"
  "$CLIP_VIT_L14_GLOBAL_TRANSFORM"
  "$OPENCLIP_LAION400M_NAIVE_TRANSFORM"
  "$OPENCLIP_LAION400M_GLOBAL_TRANSFORM"
  "$OPENCLIP_LAION2B_NAIVE_TRANSFORM"
  "$OPENCLIP_LAION2B_GLOBAL_TRANSFORM"
)
for path in "${TRANSFORM_PATHS[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Trained transform not found: $path" >&2
    echo "Run scripts/train_clip_transforms.sh first." >&2
    exit 1
  fi
done

CONFIG_PATH="${CONFIG_PATH:-scripts/representation_export.clip-cifar.json}"
echo "Collecting with config: $CONFIG_PATH"
"$PYTHON_BIN" scripts/collect_representations.py \
  --config "$CONFIG_PATH" \
  "$@"
