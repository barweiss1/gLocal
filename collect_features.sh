#!/bin/bash
#SBATCH --job-name=glocal_feats
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --time=2-00:00:00


# Create logs directory if it doesn't exist
mkdir -p logs

echo "Checking GPU..."
nvidia-smi


python3 scripts/collect_representations.py \
     --config scripts/representation_export.clip-small.json
     
