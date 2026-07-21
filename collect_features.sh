#!/bin/bash
#SBATCH --job-name=glocal_feats
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

# Submit from an activated Conda environment when possible. Otherwise, use the
# python3 available on the compute node. PYTHON_BIN can be set explicitly:
#   sbatch --export=ALL,PYTHON_BIN=/path/to/env/bin/python collect_features.sh
if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # Keep the explicit interpreter.
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="python3"
fi
CONFIG_PATH="${CONFIG_PATH:-scripts/representation_export.clip-small.json}"

echo "Checking GPU..."
nvidia-smi

echo "Using Python..."
if ! command -v "$PYTHON_BIN"; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to the absolute path of the prepared environment." >&2
  exit 1
fi
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c "import packaging, thingsvision; print('Dependencies: OK')"
echo "Config: $CONFIG_PATH"

"$PYTHON_BIN" scripts/collect_representations.py \
  --config "$CONFIG_PATH" \
  "$@"
