# gLocal transform parameter sweep

This wrapper trains one raw-ImageNet gLocal transform for every configured model
and every combination of:

```text
lambda = 0.1, 0.001
alpha  = 0.05, 0.1, 0.25, 0.5
tau    = 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0
```

Learning rate is fixed at `0.001`. Training uses SGD with momentum, scaled-identity
regularization, seed 42, no bias, and the first deterministic KFold partition used
by `main_glocal_probing.py`. The published Python file is imported unchanged.
The wrapper only corrects OpenCLIP dataset-name parsing in memory because the raw
runner assumes dataset identifiers contain no underscores.
For one-GPU SLURM tasks it also disables the published runner's forced DDP mode:
Lightning 1.8 cannot replace the sampler of the repository's custom zipped loader,
and DDP is unnecessary when each array task owns one GPU.
The repository's ThingsVision version may return extracted ImageNet features on
CPU even when the probe is on CUDA. The wrapper aligns that tensor with the
probe's transform-matrix device immediately before normalization.

## Inputs

The ImageNet root must use the repository's expected layout:

```text
IMAGENET_ROOT/
  train_set/<class>/<image>
  val_set/<class>/<image>
```

The THINGS root needs `triplets/train_90.npy` and `triplets/test_10.npy`.
`THINGS_FEATURES` defaults to `$THINGS_DATA_ROOT/features.pkl` and must contain
each configured model at `features["custom"][model]["penultimate"]`.
The CLIP configuration supplies the extractor module name (`visual`) directly, so
an external model dictionary is not required. If `MODEL_DICT_PATH` is supplied,
the wrapper validates that its module name agrees with the configuration.

## Configure and submit

Activate the prepared Conda environment and export the server paths:

```bash
conda activate glocal_env
mkdir -p logs

export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export THINGS_DATA_ROOT=/raid/home/barweiss/datasets/things-clip-training
export IMAGENET_ROOT=/path/to/imagenet/2012
export PROBING_BASE=/raid/home/barweiss/datasets/clip-transform-training

scripts/submit_glocal_transforms.sh scripts/glocal_sweep.clip.json
```

The example configuration contains four models and therefore submits 224 tasks.
Remove model entries to run a subset; each model contributes 56 tasks. Concurrent
jobs default to four and can be changed with `MAX_PARALLEL`.

The submit helper accepts additional `sbatch` options:

```bash
scripts/submit_glocal_transforms.sh \
  scripts/glocal_sweep.clip.json \
  --dependency=afterok:<job-id>
```

Inspect any task mapping without loading Torch or ImageNet:

```bash
"$PYTHON_BIN" scripts/glocal_transform_sweep.py describe \
  --config scripts/glocal_sweep.clip.json \
  --task-id 10
```

Task 10 in the example is the requested CLIP-RN50 pilot with lambda `0.1`, alpha
`0.1`, and tau `0.1`. Submit only that task with:

```bash
GLOCAL_ARRAY_OVERRIDE=10-10 \
  scripts/submit_glocal_transforms.sh scripts/glocal_sweep.clip.json
```

Both the shell worker and Python runner require `PATIENCE >= BURNIN`. The default
is 20 for both. This prevents Lightning 1.8 from repeatedly validating when early
stopping is signaled before `min_epochs`.

## Outputs and resume

Each successful task publishes:

```text
PROBING_BASE/selected/glocal/<model>/param_sweep/
  transform_lambda_<lambda>_alpha_<alpha>_tau_<tau>.npz
  result_lambda_<lambda>_alpha_<alpha>_tau_<tau>.json
```

Training and validation snapshots remain in temporary job storage. Canonical files
are written only after the returned transform and metrics pass validation. A rerun
skips only an NPZ/JSON pair whose configuration, inputs, repository revision,
dimensions, and checksum still match.

Validate the complete configured sweep after the array finishes:

```bash
"$PYTHON_BIN" scripts/glocal_transform_sweep.py validate \
  --config scripts/glocal_sweep.clip.json \
  --probing-base "$PROBING_BASE"
```

No top-level selected gLocal transform is created. All parameter combinations are
retained for the later representation-collection stage.
