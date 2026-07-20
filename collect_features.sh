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

# Submit from an activated virtual or Conda environment. PYTHON_BIN can also be
# set explicitly: PYTHON_BIN=/path/to/env/bin/python sbatch collect_features.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Checking GPU..."
nvidia-smi

echo "Using Python..."
command -v "$PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c "import packaging, thingsvision; print('Dependencies: OK')"

"$PYTHON_BIN" scripts/collect_representations.py \
  --config scripts/representation_export.clip-small.json
