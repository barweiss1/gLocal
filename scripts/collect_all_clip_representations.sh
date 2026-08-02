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
case "$PROBING_BASE" in
  /path/to/*|/path/to)
    echo "PROBING_BASE is still a placeholder: $PROBING_BASE" >&2
    echo "Set it to the transform-training output directory." >&2
    exit 1
    ;;
esac
export PROBING_BASE
CONFIG_PATH="${CONFIG_PATH:-scripts/representation_export.clip-cifar.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Representation config not found: $CONFIG_PATH" >&2
  exit 1
fi
VALIDATION_MODE="$(
  "$PYTHON_BIN" -c \
    'import json, sys; print(json.load(open(sys.argv[1])).get("transform_validation", "config"))' \
    "$CONFIG_PATH"
)"

transform_path() {
  local kind="$1"
  local slug="$2"
  echo "$PROBING_BASE/selected/$kind/$slug/transform.npz"
}

export CLIP_RN50_NAIVE_TRANSFORM
CLIP_RN50_NAIVE_TRANSFORM="$(transform_path naive clip_RN50)"
export CLIP_RN50_GLOBAL_TRANSFORM
CLIP_RN50_GLOBAL_TRANSFORM="$(transform_path global clip_RN50)"

export CLIP_VIT_L14_NAIVE_TRANSFORM
CLIP_VIT_L14_NAIVE_TRANSFORM="$(transform_path naive clip_ViT-L-14)"
export CLIP_VIT_L14_GLOBAL_TRANSFORM
CLIP_VIT_L14_GLOBAL_TRANSFORM="$(transform_path global clip_ViT-L-14)"

export OPENCLIP_LAION400M_NAIVE_TRANSFORM
OPENCLIP_LAION400M_NAIVE_TRANSFORM="$(transform_path naive OpenCLIP_ViT-L-14_laion400m_e32)"
export OPENCLIP_LAION400M_GLOBAL_TRANSFORM
OPENCLIP_LAION400M_GLOBAL_TRANSFORM="$(transform_path global OpenCLIP_ViT-L-14_laion400m_e32)"

export OPENCLIP_LAION2B_NAIVE_TRANSFORM
OPENCLIP_LAION2B_NAIVE_TRANSFORM="$(transform_path naive OpenCLIP_ViT-L-14_laion2b_s32b_b82k)"
export OPENCLIP_LAION2B_GLOBAL_TRANSFORM
OPENCLIP_LAION2B_GLOBAL_TRANSFORM="$(transform_path global OpenCLIP_ViT-L-14_laion2b_s32b_b82k)"

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

case "$VALIDATION_MODE" in
  selected)
    for path in "${TRANSFORM_PATHS[@]}"; do
      if [[ ! -f "$path" ]]; then
        echo "Selected transform not found: $path" >&2
        echo "Run scripts/select_clip_transforms.sh first." >&2
        exit 1
      fi
    done
    "$PYTHON_BIN" scripts/clip_transform_sweep.py validate-selected \
      --probing-base "$PROBING_BASE"
    ;;
  param_sweep)
    "$PYTHON_BIN" scripts/clip_transform_sweep.py validate-sweep \
      --probing-base "$PROBING_BASE"
    ;;
  all_sweeps)
    "$PYTHON_BIN" scripts/clip_transform_sweep.py validate-sweep \
      --probing-base "$PROBING_BASE"
    GLOCAL_SWEEP_CONFIG_PATH="$(
      "$PYTHON_BIN" -c \
        'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]).resolve(); print((p.parent / json.load(p.open())["glocal_sweep_config"]).resolve())' \
        "$CONFIG_PATH"
    )"
    "$PYTHON_BIN" scripts/glocal_transform_sweep.py validate \
      --config "$GLOCAL_SWEEP_CONFIG_PATH" \
      --probing-base "$PROBING_BASE"
    ;;
  config)
    ;;
  *)
    echo "Unsupported transform_validation mode in $CONFIG_PATH: $VALIDATION_MODE" >&2
    exit 1
    ;;
esac

echo "Collecting with config: $CONFIG_PATH"
"$PYTHON_BIN" scripts/collect_representations.py \
  --config "$CONFIG_PATH" \
  "$@"
