#!/usr/bin/env python3
"""Extract reusable ImageNet features for the configured CLIP models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import glocal_transform_sweep as sweep

SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_WORKERS = 2
EXPECTED_TRAIN_IMAGES = 1_281_167
EXPECTED_VAL_IMAGES = 50_000


def model_root(output_root: Path, model: sweep.ModelSpec) -> Path:
    return output_root / model.slug


def manifest_path(output_root: Path, model: sweep.ModelSpec) -> Path:
    return model_root(output_root, model) / "manifest.json"


def hdf5_path(root: Path, split: str) -> Path:
    return root / split / "features.hdf5"


def inspect_hdf5(path: Path, expected_count: int) -> dict[str, Any]:
    """Validate one feature matrix without loading it fully into RAM."""
    sweep.require_file(path, "ImageNet feature file")
    try:
        with h5py.File(path, "r") as archive:
            keys = list(archive.keys())
            if len(keys) != 1 or not isinstance(archive[keys[0]], h5py.Dataset):
                raise sweep.SweepError(
                    f"Expected one root HDF5 dataset in {path}, found {keys}"
                )
            dataset = archive[keys[0]]
            if dataset.ndim != 2 or dataset.shape[0] != expected_count:
                raise sweep.SweepError(
                    f"Expected {expected_count} × D features in {path}, "
                    f"got {dataset.shape}"
                )
            if dataset.shape[1] < 1:
                raise sweep.SweepError(f"Feature dimension is empty: {path}")
            sample_indices = sorted({0, dataset.shape[0] // 2, dataset.shape[0] - 1})
            if not np.isfinite(dataset[sample_indices]).all():
                raise sweep.SweepError(f"Sampled non-finite features in {path}")
            return {
                "count": int(dataset.shape[0]),
                "feature_dim": int(dataset.shape[1]),
                "dtype": str(dataset.dtype),
                "key": keys[0],
                "bytes": path.stat().st_size,
            }
    except (OSError, ValueError) as error:
        raise sweep.SweepError(
            f"Cannot read ImageNet features {path}: {error}"
        ) from error


def expected_manifest(
    config: Path,
    repo_root: Path,
    imagenet_root: Path,
    model: sweep.ModelSpec,
    batch_size: int,
    workers: int,
    train_info: dict[str, Any],
    val_info: dict[str, Any],
    train_sha256: str,
    val_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "model": model.name,
            "model_slug": model.slug,
            "source": model.source,
            "module": model.module,
            "module_name": model.module_name,
            "batch_size": batch_size,
            "workers": workers,
            "resize_dim": 256,
            "crop_dim": 224,
            "format": "hdf5",
        },
        "features": {
            "train": {**train_info, "sha256": train_sha256},
            "val": {**val_info, "sha256": val_sha256},
        },
        "inputs": {
            "config": str(config),
            "config_sha256": sweep.sha256(config),
            "imagenet_root": str(imagenet_root),
            "published_entrypoint": str(
                repo_root / "main_imagenet_feature_extraction.py"
            ),
            "published_entrypoint_sha256": sweep.sha256(
                repo_root / "main_imagenet_feature_extraction.py"
            ),
        },
        "packages": sweep.package_versions(),
        "repo_revision": sweep.git_revision(repo_root),
    }


def validate_model(
    config: Path,
    repo_root: Path,
    imagenet_root: Path,
    output_root: Path,
    model: sweep.ModelSpec,
    batch_size: int,
    workers: int,
    train_count: int,
    val_count: int,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Validate whether a published cache is reusable by the current model.

    Extraction batch size, worker count, package versions, paths, and repository
    revision remain recorded as provenance, but do not change the meaning of an
    already published feature matrix. Compatibility therefore depends on the
    model/preprocessing identity, the published extractor, and the recorded file
    metadata and checksums.
    """
    del batch_size, workers
    root = model_root(output_root, model)
    train_file = hdf5_path(root, "train")
    val_file = hdf5_path(root, "val")
    train_info = inspect_hdf5(train_file, train_count)
    val_info = inspect_hdf5(val_file, val_count)
    if train_info["feature_dim"] != val_info["feature_dim"]:
        raise sweep.SweepError(
            f"Train/validation feature dimensions differ for {model.name}"
        )
    manifest = sweep.read_json(manifest_path(output_root, model))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise sweep.SweepError(f"Feature schema mismatch for {model.name}")
    configuration = manifest.get("configuration")
    expected_identity = {
        "model": model.name,
        "model_slug": model.slug,
        "source": model.source,
        "module": model.module,
        "module_name": model.module_name,
        "resize_dim": 256,
        "crop_dim": 224,
        "format": "hdf5",
    }
    if not isinstance(configuration, dict):
        raise sweep.SweepError(f"Feature configuration is missing for {model.name}")
    for key, expected_value in expected_identity.items():
        if configuration.get(key) != expected_value:
            raise sweep.SweepError(
                f"ImageNet feature {key} mismatch for {model.name}: "
                f"{configuration.get(key)!r} != {expected_value!r}"
            )

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise sweep.SweepError(f"Feature input metadata is missing for {model.name}")
    entrypoint = repo_root / "main_imagenet_feature_extraction.py"
    if inputs.get("published_entrypoint_sha256") != sweep.sha256(entrypoint):
        raise sweep.SweepError(
            f"Published ImageNet extractor mismatch for {model.name}"
        )

    recorded_features = manifest.get("features")
    if not isinstance(recorded_features, dict):
        raise sweep.SweepError(f"Feature metadata is missing for {model.name}")
    for split, path, actual_info in (
        ("train", train_file, train_info),
        ("val", val_file, val_info),
    ):
        recorded = recorded_features.get(split)
        if not isinstance(recorded, dict):
            raise sweep.SweepError(
                f"{split} feature metadata is missing for {model.name}"
            )
        for key, actual_value in actual_info.items():
            if recorded.get(key) != actual_value:
                raise sweep.SweepError(
                    f"ImageNet {split} feature {key} mismatch for {model.name}: "
                    f"{recorded.get(key)!r} != {actual_value!r}"
                )
        recorded_sha = recorded.get("sha256")
        if not isinstance(recorded_sha, str):
            raise sweep.SweepError(
                f"{split} feature checksum is missing for {model.name}"
            )
        if verify_checksums and sweep.sha256(path) != recorded_sha:
            raise sweep.SweepError(
                f"ImageNet {split} feature checksum mismatch for {model.name}"
            )
    return manifest


def import_extractor(repo_root: Path) -> Any:
    entrypoint = repo_root / "main_imagenet_feature_extraction.py"
    module_name = "_published_imagenet_feature_extraction"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise sweep.SweepError(f"Cannot import published entry point: {entrypoint}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> int:
    models = sweep.load_models(args.config)
    if not 0 <= args.task_id < len(models):
        raise sweep.SweepError(
            f"Task id must be in [0, {len(models) - 1}], got {args.task_id}"
        )
    if args.batch_size < 1 or args.workers < 0:
        raise sweep.SweepError("Batch size must be positive and workers non-negative")
    model = models[args.task_id]
    repo_root = args.repo_root.expanduser().resolve()
    imagenet_root = args.imagenet_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config = args.config.expanduser().resolve()
    sweep.require_file(
        repo_root / "main_imagenet_feature_extraction.py",
        "Published ImageNet extraction entry point",
    )
    sweep.validate_imagenet_split(imagenet_root / "train_set")
    sweep.validate_imagenet_split(imagenet_root / "val_set")
    sweep.check_writable(output_root)

    try:
        validate_model(
            config,
            repo_root,
            imagenet_root,
            output_root,
            model,
            args.batch_size,
            args.workers,
            args.train_count,
            args.val_count,
        )
    except sweep.SweepError:
        pass
    else:
        print(f"Validated existing ImageNet features; skipping {model.name}")
        return 0

    destination = model_root(output_root, model)
    if destination.exists():
        raise sweep.SweepError(
            f"Invalid existing feature directory blocks atomic publish: {destination}"
        )

    with tempfile.TemporaryDirectory(
        prefix=f".{model.slug}.tmp-",
        dir=output_root,
    ) as temporary_name:
        temporary = Path(temporary_name)
        upstream = import_extractor(repo_root)
        model_cfg = {
            "model": model.name,
            "module": model.module_name,
            "source": model.source,
            "device": args.device,
            "extract_cls_token": False,
        }
        upstream.extract(
            imagenet_root=str(imagenet_root),
            model_cfg=model_cfg,
            batch_size=args.batch_size,
            num_workers=args.workers,
            out_path=str(temporary),
            splits=["train", "val"],
            out_format="hdf5",
        )
        train_file = hdf5_path(temporary, "train")
        val_file = hdf5_path(temporary, "val")
        train_info = inspect_hdf5(train_file, args.train_count)
        val_info = inspect_hdf5(val_file, args.val_count)
        if train_info["feature_dim"] != val_info["feature_dim"]:
            raise sweep.SweepError("Train/validation feature dimensions differ")
        manifest = expected_manifest(
            config,
            repo_root,
            imagenet_root,
            model,
            args.batch_size,
            args.workers,
            train_info,
            val_info,
            sweep.sha256(train_file),
            sweep.sha256(val_file),
        )
        sweep.atomic_json(manifest, temporary / "manifest.json")
        os.replace(temporary, destination)
    print(f"Published ImageNet features: {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    count = subparsers.add_parser("count")
    count.add_argument("--config", type=Path, required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, required=True)
    common.add_argument("--repo-root", type=Path, required=True)
    common.add_argument("--imagenet-root", type=Path, required=True)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    common.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    common.add_argument("--train-count", type=int, default=EXPECTED_TRAIN_IMAGES)
    common.add_argument("--val-count", type=int, default=EXPECTED_VAL_IMAGES)
    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--task-id", type=int, required=True)
    run_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    subparsers.add_parser("validate", parents=[common])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "count":
            print(len(sweep.load_models(args.config)))
            return 0
        if args.command == "run":
            return run(args)
        models = sweep.load_models(args.config)
        for model in models:
            validate_model(
                args.config.expanduser().resolve(),
                args.repo_root.expanduser().resolve(),
                args.imagenet_root.expanduser().resolve(),
                args.output_root.expanduser().resolve(),
                model,
                args.batch_size,
                args.workers,
                args.train_count,
                args.val_count,
            )
        print(f"Validated ImageNet features for {len(models)} models.")
        return 0
    except sweep.SweepError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
