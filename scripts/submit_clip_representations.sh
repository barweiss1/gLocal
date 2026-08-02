#!/bin/bash
# Submit the combined CLIP representation export as four GPU jobs.

set -euo pipefail

if [[ -n "${GLOCAL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$GLOCAL_REPO_ROOT"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN=python3
fi
if ! command -v "$PYTHON_BIN" >/dev/null; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

PROBING_BASE="${PROBING_BASE:-$REPO_ROOT/clip-transform-training}"
CONFIG_PATH="${CONFIG_PATH:-scripts/representation_export.clip-all-sweeps.json}"
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Representation config not found: $CONFIG_PATH" >&2
  exit 1
fi

export GLOCAL_REPO_ROOT="$REPO_ROOT"
export PYTHON_BIN
export PROBING_BASE
export CONFIG_PATH

echo "Preparing shared CIFAR downloads and catalogs..." >&2
"$PYTHON_BIN" scripts/collect_representations.py \
  --config "$CONFIG_PATH" \
  --stage prepare-catalogs

submit() {
  local label="$1"
  shift
  local job_id
  job_id="$(
    sbatch --parsable \
      --export=ALL \
      "$@" \
      scripts/collect_all_clip_representations.sh \
      "${SUBMIT_ARGS[@]}"
  )"
  echo "$label: $job_id"
}

SBATCH_OPTIONS=("$@")

SUBMIT_ARGS=(--model clip_RN50)
submit "clip_RN50 (all CIFAR datasets)" "${SBATCH_OPTIONS[@]}"

OPENCLIP_MODEL=OpenCLIP_ViT-L-14_laion2b_s32b_b82k
for dataset in cifar10 cifar100 cifar100-coarse; do
  SUBMIT_ARGS=(--model "$OPENCLIP_MODEL" --dataset "$dataset")
  submit "$OPENCLIP_MODEL ($dataset)" "${SBATCH_OPTIONS[@]}"
done
