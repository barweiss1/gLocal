# Efficient gLocal transform parameter sweep

The wrapper trains every configured model with:

```text
lambda = 0.1, 0.001
alpha  = 0.1, 0.25, 0.5, 0.75
tau    = 0.1, 0.25, 0.5, 1.0
```

Learning rate is fixed at `0.001`. Training uses SGD with momentum,
scaled-identity regularization, seed 42, no bias, and the first deterministic
three-way KFold partition. Published repository files remain unchanged.

The efficient repository path is used: ImageNet representations are extracted
once per model and reused across all 32 parameter jobs. This removes repeated
CLIP inference from every optimization step while retaining contrastive batch
size 1,024.

## Environment

```bash
conda activate glocal_env
mkdir -p logs

export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export THINGS_DATA_ROOT=/raid/home/barweiss/datasets/things-clip-training
export IMAGENET_ROOT=/raid/home/barweiss/datasets/imagenet-2012/glocal-layout
export IMAGENET_FEATURES_BASE=/raid/home/barweiss/datasets/imagenet-clip-features
export PROBING_BASE=/raid/home/barweiss/datasets/clip-transform-training
```

`IMAGENET_ROOT` is needed only during one-time feature extraction and must have:

```text
IMAGENET_ROOT/
  train_set/<class>/<image>
  val_set/<class>/<image>
```

The THINGS root needs `triplets/train_90.npy`, `triplets/test_10.npy`, and
`features.pkl`.

## 1. Extract ImageNet features once

Submit one task per model:

```bash
FEATURE_JOB="$(
  bash scripts/submit_imagenet_clip_features.sh \
    scripts/glocal_sweep.clip.json
)"
echo "$FEATURE_JOB"
```

Tasks run sequentially by default so this stage does not require extra
concurrent GPUs. Each model is stored under:

```text
IMAGENET_FEATURES_BASE/<model>/
  train/features.hdf5
  val/features.hdf5
  manifest.json
```

The default extraction batch is 128 with two loader workers. Override them when
needed:

```bash
FEATURE_BATCH_SIZE=64 FEATURE_WORKERS=1 \
  bash scripts/submit_imagenet_clip_features.sh \
    scripts/glocal_sweep.clip.json
```

Validate completed caches:

```bash
"$PYTHON_BIN" scripts/precompute_imagenet_clip_features.py validate \
  --config scripts/glocal_sweep.clip.json \
  --repo-root "$PWD" \
  --imagenet-root "$IMAGENET_ROOT" \
  --output-root "$IMAGENET_FEATURES_BASE"
```

## 2. Run the gLocal sweep

After feature extraction succeeds, submit a pilot for task 0. In the example
configuration this is CLIP-RN50 with lambda `0.1`, alpha `0.1`, and tau `0.1`:

```bash
GLOCAL_ARRAY_OVERRIDE=0-0 MAX_PARALLEL=1 \
  bash scripts/submit_glocal_transforms.sh \
    scripts/glocal_sweep.clip.json
```

Then submit the complete configured sweep:

```bash
bash scripts/submit_glocal_transforms.sh \
  scripts/glocal_sweep.clip.json
```

Alternatively, submit extraction and training together with a dependency:

```bash
FEATURE_JOB="$(
  bash scripts/submit_imagenet_clip_features.sh \
    scripts/glocal_sweep.clip.json
)"
bash scripts/submit_glocal_transforms.sh \
  scripts/glocal_sweep.clip.json \
  --dependency="afterok:$FEATURE_JOB"
```

Each model contributes 32 training tasks. `MAX_PARALLEL` controls concurrent
training jobs; it defaults to four. Both the shell worker and Python runner
limit training to 10 epochs and require `PATIENCE >= BURNIN`; both default to
10.

## Outputs and resume

Each successful task publishes:

```text
PROBING_BASE/selected/glocal/<model>/param_sweep/
  transform_lambda_<lambda>_alpha_<alpha>_tau_<tau>.npz
  result_lambda_<lambda>_alpha_<alpha>_tau_<tau>.json
```

Feature extraction and training both use temporary directories. A rerun skips
only validated canonical artifacts. Failed jobs do not publish partial
transforms.

Validate the configured transform sweep:

```bash
"$PYTHON_BIN" scripts/glocal_transform_sweep.py validate \
  --config scripts/glocal_sweep.clip.json \
  --probing-base "$PROBING_BASE"
```

No top-level selected gLocal transform is created because every parameter
combination is retained for representation collection.
