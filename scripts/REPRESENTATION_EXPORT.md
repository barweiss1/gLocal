# Representation export wrapper

The wrapper extracts deterministic feature batches without changing the repository's
existing model, probing, anomaly-detection, or few-shot code.

## Configure

Copy `representation_export.example.json` and set the dataset roots. Relative paths
are resolved from the config file and environment variables are expanded.

The checked-in `transforms/` files are the glocal transforms. Global and naive
transforms are selected from the CLIP lambda sweep described below.

The extended CLIP/CIFAR configuration supports per-model NPZ artifacts for both
regularization variants. `naive` uses `--regularization l2`; `global` uses
`--regularization eye`, which penalizes distance from a scaled identity matrix.
Legacy nested `naive_transforms.pkl` artifacts remain supported as well.

## Extract

```bash
python scripts/collect_representations.py --config representation-export.json
```

For the small CLIP/CIFAR-100 pilot, the included configuration downloads CIFAR-100
on the first run and exports only the test split with `none` and `glocal` features:

```bash
.venv/bin/python scripts/collect_representations.py \
  --config scripts/representation_export.clip-small.json
```

## Train and select CLIP transforms

The training script is a 32-task SLURM array: four CLIP models, naive and global
regularization, and four lambda values (`0.01`, `0.1`, `1.0`, and `10.0`), with at
most four tasks running concurrently. Learning rate is fixed at `0.001`; training
uses SGD with momentum, three folds, seed 42, and no bias. Naive uses L2
regularization and global uses distance from a scaled identity matrix.

Each task runs the published `main_global_probing.py` unchanged through a small
compatibility launcher. The launcher gives every Lightning Trainer a fresh copy
of the callback list, so early-stopping state cannot leak into the next fold.
Training occurs in temporary SLURM storage and is published only after the output
NPZ and CV result row pass validation.

First prepare THINGS once. The repository loader's `download=True` option only
downloads the concept table; it does not download the images or triplets. The
preparation wrapper downloads the official 90/10 behavioral splits and the 1,854
image THINGSplus CC0 subset, then uses `THINGSBehavior` to extract all four CLIP
feature matrices. It does not download the full 26,107-image THINGS archive.

Submit preparation to a GPU worker:

```bash
conda activate glocal_env
mkdir -p logs

export THINGS_DATA_ROOT=/raid/home/barweiss/datasets/things-clip-training

PREP_JOB=$(sbatch --parsable \
  --export="ALL,PYTHON_BIN=$CONDA_PREFIX/bin/python" \
  scripts/prepare_things_clip.sh)
```

Run `sbatch` from the gLocal repository root. SLURM executes a spool copy of the
shell script, so the wrappers use `SLURM_SUBMIT_DIR` to return to the repository.
If submitting from another directory, include
`GLOCAL_REPO_ROOT=/absolute/path/to/gLocal` in `--export`.

The CC0 archive is about 1.18 GB. If the expected 1,854 files are already present
as `$THINGS_DATA_ROOT/images/<uniqueID>.jpg`, add `--use-existing-images` to skip
that download. The preparation command resumes completed triplets, images, models,
and `features.pkl` entries.

```bash
conda activate glocal_env
mkdir -p logs

export PROBING_BASE=/raid/home/barweiss/datasets/clip-transform-training

TRAIN_JOB=$(sbatch --parsable \
  --dependency="afterok:$PREP_JOB" \
  --export="ALL,PYTHON_BIN=$CONDA_PREFIX/bin/python" \
  scripts/train_clip_transforms.sh)
```

This dependency launches transform training only after THINGS preparation has
completed successfully. To run preparation without downloading images that are
already installed, submit with
`--export="ALL,PYTHON_BIN=$CONDA_PREFIX/bin/python,USE_EXISTING_THINGS_IMAGES=1"`.

The training script defaults `THINGS_FEATURES` to `$THINGS_DATA_ROOT/features.pkl`.
You can set it explicitly when using an existing feature dictionary. It must
contain all four models under `features["custom"][model]["penultimate"]`.

By default, a task whose validated artifact already exists is skipped. Set
`OVERWRITE=1` in the job's `--export` list to force that task to retrain and
republish even though a valid transform is already present, for example
`--export="ALL,PYTHON_BIN=$CONDA_PREFIX/bin/python,OVERWRITE=1"`.

Training defaults to `BURNIN=15` and `PATIENCE=15`, and the wrappers require
patience to be greater than or equal to burn-in. In Lightning 1.8, an
EarlyStopping callback can set `trainer.should_stop` before `min_epochs` has been
met. The epoch loop then forces another validation because stopping was requested,
but it cannot exit because of `min_epochs`, causing validation to repeat without
advancing the epoch. Preventing an early-stop signal before burn-in avoids that
loop while leaving the published probing source unchanged.

If an older sweep submission is looping, cancel its affected array tasks and
resubmit the array. For example:

```bash
scancel <array-job-id_or_array-task-id>
```

Valid lambda artifacts that finished earlier are detected and skipped. Looping
tasks used temporary storage and did not publish a canonical artifact, so they
restart cleanly. Older Adam/lambda `0.001` outputs are also left untouched.

For a one-configuration pilot, submit array task 0 (CLIP-RN50, naive, lambda
`0.01`) and check that the log prints three fresh-Trainer messages:

```bash
sbatch --array=0-0 \
  --export="ALL,PYTHON_BIN=$CONDA_PREFIX/bin/python" \
  scripts/train_clip_transforms.sh
```

After all 32 tasks complete, select the best lambda for each model and transform:

```bash
SELECT_JOB=$(sbatch --parsable \
  --dependency="afterok:$TRAIN_JOB" \
  --export="ALL,PYTHON_BIN=$CONDA_PREFIX/bin/python" \
  scripts/select_clip_transforms.sh)
```

Selection minimizes mean CV cross-entropy, then breaks exact ties with higher CV
accuracy and lower lambda. All sweep transforms remain available as:

```text
PROBING_BASE/selected/<naive|global>/<model>/param_sweep/transform_lambda_<lambda>.npz
```

The winner is copied to
`PROBING_BASE/selected/<naive|global>/<model>/transform.npz`, with all candidate
metrics, hashes, hyperparameters, and package versions recorded in
`manifest.json`. Rerunning the array skips only artifacts whose NPZ and metadata
fully match the fixed sweep contract. Older Adam/lambda `0.001` outputs are left
untouched and ignored.

The preparation job intentionally ignores `THINGS_FEATURES` inherited from the
shell. To place its output somewhere other than `$THINGS_DATA_ROOT/features.pkl`,
set `PREP_FEATURES_OUTPUT` explicitly.

## Collect all CLIP representations

The collection job resolves the eight trained transforms from `PROBING_BASE` and
uses the checked-in gLocal transforms. It covers CIFAR-10, CIFAR-100 fine labels,
and CIFAR-100 coarse labels with `none`, `naive`, `global`, and `glocal`:

```bash
sbatch \
  --dependency="afterok:$SELECT_JOB" \
  --export="ALL,PYTHON_BIN=$CONDA_PREFIX/bin/python" \
  scripts/collect_all_clip_representations.sh
```

The dependency starts collection only after the 32 sweep tasks and selection job
succeed. Collection validates all eight top-level selected transforms and their
manifests before loading a model or dataset.

### Collect every naive/global lambda

To compare the complete parameter sweep instead of only the selected transforms,
use the parameter-sweep configuration:

```bash
export CONFIG_PATH=scripts/representation_export.clip-param-sweep.json

PARAM_FEATURE_JOB=$(sbatch --parsable \
  --export="ALL,PYTHON_BIN=$CONDA_PREFIX/bin/python,CONFIG_PATH=$CONFIG_PATH" \
  scripts/collect_all_clip_representations.sh)
```

This exports `none` and all four lambda values for both naive and global from the
same raw feature batches. It validates the 32 per-lambda artifacts directly and
does not require the top-level selected transforms or a selection job. The output
root is `features-export-param-sweep`:

```text
features-export-param-sweep/<model>/<dataset>/none/test-batch-000000.npz
features-export-param-sweep/<model>/<dataset>/naive-lambda-0.01/test-batch-000000.npz
features-export-param-sweep/<model>/<dataset>/naive-lambda-0.1/test-batch-000000.npz
features-export-param-sweep/<model>/<dataset>/naive-lambda-1.0/test-batch-000000.npz
features-export-param-sweep/<model>/<dataset>/naive-lambda-10.0/test-batch-000000.npz
features-export-param-sweep/<model>/<dataset>/global-lambda-0.01/test-batch-000000.npz
...
```

The command is resumable and the configuration can be narrowed with repeated
`--model`, `--dataset`, or `--transform` arguments. For example,
`--transform none --transform global-lambda-0.1` exports only those two variants.

### Collect naive, global, and gLocal parameter sweeps

The combined configuration covers the two models in `glocal_sweep.clip.json` and
exports 73 variants per model: `none`, four naive lambdas, four global lambdas,
and all 64 gLocal `(lambda, alpha, tau)` combinations:

```bash
conda activate glocal_env
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export PROBING_BASE=/raid/home/barweiss/datasets/clip-transform-training
export CONFIG_PATH=scripts/representation_export.clip-all-sweeps.json
mkdir -p logs

sbatch --export=ALL \
  scripts/collect_all_clip_representations.sh
```

To collect only CIFAR-100:

```bash
sbatch --export=ALL \
  scripts/collect_all_clip_representations.sh \
  --dataset cifar100
```

Outputs are written to `features-export-all-sweeps`. The collection wrapper
validates both the naive/global sweep and the configured gLocal sweep before
loading a model. CIFAR data downloads automatically on the first run.

To use four GPUs, run the four-way submit helper instead. It prepares the shared
CIFAR catalogs first, then submits one CLIP-RN50 job for all datasets and three
OpenCLIP jobs (one per dataset):

```bash
bash scripts/submit_clip_representations.sh
```

Optional `sbatch` flags are forwarded to all four jobs:

```bash
bash scripts/submit_clip_representations.sh \
  --partition=high \
  --time=2-00:00:00
```

Sharding is by model/dataset rather than transform, so each job still computes
raw model features only once and derives all 73 variants from them.

The CIFAR `-shift`, `-rvo`, and `cifar10vs100` variants are AD evaluation protocols
derived from these image sets, not additional representation-extraction datasets.

The script loads one model at a time. Dataset order defines fixed logical batches of
1,024 samples. Inference may use smaller batches, but each output file keeps the same
sample IDs across every model and transform:

```text
<features>/<model>/<dataset>/<none|global|glocal|naive>/<split>-batch-000000.npz
```

Each NPZ contains `features`, `labels`, `sample_ids`, `sample_indices`, model/layer
metadata, and `performance_json`. Performance initially has status `not-attached`.

### Resumability

`collect_representations.py extract` is resumable per output batch, with no
separate overwrite flag needed. Each batch is reused only if its `sample_ids`
and transform identity/hash match what the current configuration expects;
otherwise it is silently regenerated. A transform hash change (for example, a
retrained gLocal transform) therefore invalidates and regenerates only the
batches that depended on it, while everything else is left untouched.

## Add downstream results

Run the repository's existing AD and FS scripts normally. Put their desired summary
metrics into one small JSON file using this shape:

```json
{
  "resnet18": {
    "cifar100": {
      "none": {"ad": {"macro_auroc": 0.80}, "fs": {"accuracy": 0.65}},
      "global": {"ad": {"macro_auroc": 0.82}, "fs": {"accuracy": 0.67}},
      "glocal": {"ad": {"macro_auroc": 0.84}, "fs": {"accuracy": 0.69}}
    }
  }
}
```

Attach that summary to every corresponding feature file:

```bash
python scripts/collect_representations.py \
  --config representation-export.json \
  --stage attach-performance \
  --performance performance.json
```

Finally, verify that all files exist and have identical sample alignment:

```bash
python scripts/collect_representations.py \
  --config representation-export.json \
  --stage validate
```

Repeated `--model`, `--dataset`, and `--transform` options select a smaller subset
for pilot runs. For example, this compares raw and checked-in gLocal features and
does not require a global transform:

```bash
python scripts/collect_representations.py \
  --config representation-export.json \
  --model clip_RN50 \
  --dataset cifar100 \
  --transform none \
  --transform glocal
```
