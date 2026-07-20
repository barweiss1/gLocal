#!/usr/bin/env python3
"""Export aligned model representations in resumable NPZ batches.

The wrapper extracts each model's raw features once per logical batch and writes
the selected ``none``, ``global``, and ``glocal`` variants under
``<features>/<model>/<dataset>/<transform>/``. Dataset catalogs keep sample order
identical across models and variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRANSFORMS = ("none", "global", "glocal")
DATASET_SPLITS = {
    "cifar100": ("train", "test"),
    "dtd": ("train", "val", "test"),
    "sun397": ("train", "test"),
    "imagenet": ("train", "val"),
}


class ExportError(RuntimeError):
    """Raised when configuration or exported artifacts violate the contract."""


def slug(value: str) -> str:
    """Convert a model or dataset name into a filesystem-safe directory name."""
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise ExportError(f"Cannot create a filename from {value!r}")
    return result


def resolve_path(value: str, base: Path) -> Path:
    """Expand a configured path and resolve relative paths from the config folder."""
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:
        raise ExportError(f"Unresolved environment variable in path: {value}")
    path = Path(expanded)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(
    path: Path,
    selected_models: Optional[Sequence[str]] = None,
    selected_datasets: Optional[Sequence[str]] = None,
    selected_transforms: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load, default, filter, and resolve a representation-export configuration.

    Command-line selections override the configured transform list and filter the
    model and dataset lists. Transform paths are required only for selected
    transformed variants, so a ``none``/``glocal`` run does not need globals.
    """
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["config_path"] = path
    config["features_root"] = resolve_path(config["features_root"], path.parent)
    config.setdefault("device", "cuda")
    config.setdefault("logical_batch_size", 1024)
    config.setdefault("compute_batch_size", 128)
    config.setdefault("num_workers", 4)
    config.setdefault("allow_download", False)
    transforms = list(selected_transforms or config.get("transforms", TRANSFORMS))
    invalid_transforms = sorted(set(transforms) - set(TRANSFORMS))
    if invalid_transforms:
        raise ExportError(f"Unsupported transforms: {invalid_transforms}")
    if not transforms:
        raise ExportError("The selected transform set is empty")
    config["transforms"] = list(dict.fromkeys(transforms))

    models = []
    for item in config["models"]:
        if selected_models and item["name"] not in selected_models:
            continue
        model = dict(item)
        model["slug"] = model.get("slug", slug(model["name"]))
        for kind in config["transforms"]:
            if kind == "none":
                continue
            key = f"{kind}_transform"
            if key not in model:
                raise ExportError(f"{model['name']} is missing {key}")
            model[key] = resolve_path(model[key], path.parent)
        models.append(model)
    datasets = []
    for item in config["datasets"]:
        name = item["name"].lower()
        if selected_datasets and name not in selected_datasets:
            continue
        if name not in DATASET_SPLITS:
            raise ExportError(f"Unsupported dataset: {name}")
        dataset = dict(item)
        dataset["name"] = name
        dataset["slug"] = dataset.get("slug", slug(name))
        dataset["root"] = resolve_path(dataset["root"], path.parent)
        dataset.setdefault("splits", list(DATASET_SPLITS[name]))
        dataset.setdefault("allow_download", config["allow_download"])
        datasets.append(dataset)
    if not models or not datasets:
        raise ExportError("The selected model/dataset set is empty")
    config["models"] = models
    config["datasets"] = datasets
    return config


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically write a JSON-serializable value with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_npz(path: Path, values: Mapping[str, Any]) -> None:
    """Atomically write a compressed NPZ that is safe for ``allow_pickle=False``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load all arrays from an NPZ without enabling Python object deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def text(value: str) -> np.ndarray:
    """Encode metadata text as a NumPy Unicode scalar instead of an object array."""
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def load_transform(path: Path) -> Dict[str, Any]:
    """Load and validate a transform artifact and record its identity.

    Artifacts must contain ``weights``, ``mean``, and ``std``; ``bias`` is
    optional. The weight matrix must be square and standard deviations nonzero.
    """
    if not path.is_file():
        raise ExportError(f"Missing transform: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = {"weights", "mean", "std"} - set(archive.files)
        if missing:
            raise ExportError(f"{path} is missing {sorted(missing)}")
        result = {
            "weights": np.asarray(archive["weights"], dtype=np.float32),
            "mean": np.asarray(archive["mean"], dtype=np.float32),
            "std": np.asarray(archive["std"], dtype=np.float32),
            "bias": (
                np.asarray(archive["bias"], dtype=np.float32)
                if "bias" in archive.files
                else None
            ),
            "path": path,
            "sha256": sha256(path),
        }
    weights = result["weights"]
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ExportError(f"Expected a square transform matrix: {path}")
    if np.any(result["std"] == 0):
        raise ExportError(f"Transform has a zero standard deviation: {path}")
    return result


def apply_transform(features: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    """Apply ``((features - mean) / std) @ weights + bias`` as float32."""
    if features.shape[1] != transform["weights"].shape[0]:
        raise ExportError(
            f"Feature dimension {features.shape[1]} does not match transform "
            f"dimension {transform['weights'].shape[0]}"
        )
    result = ((features - transform["mean"]) / transform["std"]) @ transform["weights"]
    if transform["bias"] is not None:
        result += transform["bias"]
    return np.asarray(result, dtype=np.float32)


def make_dataset(spec: Mapping[str, Any], split: str, transform: Any) -> Any:
    """Construct one supported torchvision dataset split in canonical order."""
    from torchvision.datasets import CIFAR100, DTD, ImageNet, SUN397

    root = str(spec["root"])
    download = bool(spec["allow_download"])
    if spec["name"] == "cifar100":
        return CIFAR100(
            root=root, train=split == "train", transform=transform, download=download
        )
    if spec["name"] == "dtd":
        return DTD(root=root, split=split, transform=transform, download=download)
    if spec["name"] == "sun397":
        dataset = SUN397(root=root, transform=transform, download=download)
        filename = "Training_01.txt" if split == "train" else "Testing_01.txt"
        with (Path(dataset.root) / filename).open("r", encoding="utf-8") as handle:
            names = [line for line in handle.read().splitlines() if line]
        dataset._image_files = [dataset._data_dir / name[1:] for name in names]
        dataset._labels = [
            dataset.class_to_idx["/".join(name.split("/")[2:-1])] for name in names
        ]
        return dataset
    if spec["name"] == "imagenet":
        return ImageNet(root=root, split=split, transform=transform)
    raise ExportError(f"Unsupported dataset: {spec['name']}")


def labels_for(dataset: Any) -> np.ndarray:
    """Read class labels from the dataset interfaces used by torchvision."""
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "_labels"):
        labels = dataset._labels
    else:
        labels = [label for _, label in dataset.samples]
    return np.asarray(labels, dtype=np.int64)


def ids_for(dataset: Any, spec: Mapping[str, Any], split: str) -> np.ndarray:
    """Build stable sample IDs from relative paths or deterministic indices."""
    paths = None
    if hasattr(dataset, "_image_files"):
        paths = dataset._image_files
    elif hasattr(dataset, "samples"):
        paths = [path for path, _ in dataset.samples]
    if paths is None:
        return np.asarray([f"{split}/{i:08d}" for i in range(len(dataset))])
    result = []
    for path in paths:
        try:
            result.append(Path(path).resolve().relative_to(spec["root"]).as_posix())
        except ValueError:
            result.append(Path(path).as_posix())
    return np.asarray(result)


def catalog_path(
    config: Mapping[str, Any], dataset: Mapping[str, Any], split: str
) -> Path:
    """Return the canonical sample-catalog path for a dataset split."""
    return config["features_root"] / "_index" / dataset["slug"] / f"{split}.npz"


def ensure_catalog(
    config: Mapping[str, Any], dataset_spec: Mapping[str, Any], split: str
) -> Dict[str, np.ndarray]:
    """Create or verify the sample IDs, labels, and indices for one split.

    An existing catalog is never silently replaced: changed ordering or labels
    raise :class:`ExportError` to prevent cross-model misalignment.
    """
    dataset = make_dataset(dataset_spec, split, transform=None)
    values = {
        "sample_ids": ids_for(dataset, dataset_spec, split),
        "labels": labels_for(dataset),
        "sample_indices": np.arange(len(dataset), dtype=np.int64),
    }
    path = catalog_path(config, dataset_spec, split)
    if path.is_file():
        existing = read_npz(path)
        if any(
            not np.array_equal(existing[key], value) for key, value in values.items()
        ):
            raise ExportError(f"Dataset order changed: {dataset_spec['name']}:{split}")
    else:
        write_npz(path, values)
    return values


def load_extractor(model: Mapping[str, Any], device: str) -> Any:
    """Create the pretrained ThingsVision extractor described by a model spec."""
    from thingsvision import get_extractor
    from utils.probing.helpers import model_name_to_thingsvision

    name, parameters = model_name_to_thingsvision(model["name"])
    if model.get("extract_cls_token"):
        parameters = dict(parameters or {})
        parameters["extract_cls_token"] = True
    return get_extractor(
        model_name=name,
        source=model["source"],
        device=device,
        pretrained=True,
        model_parameters=parameters,
    )


def extract_features(
    extractor: Any,
    images: Any,
    layer: str,
    batch_size: int,
    cls_token: bool,
) -> Tuple[np.ndarray, int]:
    """Extract one logical batch, reducing the inference microbatch on OOM.

    Returns the concatenated float32 features in input order and the microbatch
    size that succeeded. For token models, only the CLS token is retained.
    """
    import torch

    current = min(batch_size, len(images))
    while current:
        parts = []
        try:
            for start in range(0, len(images), current):
                features = extractor.extract_features(
                    batches=[images[start : start + current]],
                    module_name=layer,
                    flatten_acts=not cls_token,
                )
                if hasattr(features, "detach"):
                    features = features.detach().cpu().numpy()
                features = np.asarray(features)
                if cls_token and features.ndim >= 3:
                    features = features[:, 0]
                parts.append(features.reshape(len(features), -1))
            return np.concatenate(parts).astype(np.float32), current
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current == 1:
                raise
            current //= 2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise ExportError("No usable compute batch size")


def output_dir(
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    transform: str,
) -> Path:
    """Return the output directory for a model/dataset/transform combination."""
    return config["features_root"] / model["slug"] / dataset["slug"] / transform


def batch_name(split: str, index: int) -> str:
    """Format a zero-based logical batch filename."""
    return f"{split}-batch-{index:06d}.npz"


def batch_is_valid(
    path: Path, ids: np.ndarray, transform: str, transform_sha256: str
) -> bool:
    """Check whether a batch can be resumed for the expected samples and transform."""
    if not path.is_file():
        return False
    try:
        values = read_npz(path)
        return (
            np.array_equal(values["sample_ids"], ids)
            and str(values["transform"].item()) == transform
            and str(values["transform_sha256"].item()) == transform_sha256
        )
    except (KeyError, OSError, ValueError):
        return False


def extract(config: Mapping[str, Any]) -> None:
    """Extract and export all selected model, dataset, and transform variants.

    Raw model features are computed once per logical batch. Transformed variants
    are derived from that same array, guaranteeing matching samples and ordering.
    Existing batches with matching sample IDs and transform hashes are reused.
    """
    import torch

    config["features_root"].mkdir(parents=True, exist_ok=True)
    for model in config["models"]:
        transforms = {
            kind: load_transform(model[f"{kind}_transform"])
            for kind in config["transforms"]
            if kind != "none"
        }
        dimensions = {value["weights"].shape for value in transforms.values()}
        if len(dimensions) > 1:
            raise ExportError(f"Global/glocal dimensions differ for {model['name']}")
        extractor = load_extractor(model, config["device"])
        preprocessing = extractor.get_transformations()
        for dataset_spec in config["datasets"]:
            metadata: Dict[str, List[Dict[str, Any]]] = {
                kind: [] for kind in config["transforms"]
            }
            for split in dataset_spec["splits"]:
                catalog = ensure_catalog(config, dataset_spec, split)
                dataset = make_dataset(dataset_spec, split, preprocessing)
                if not np.array_equal(
                    ids_for(dataset, dataset_spec, split), catalog["sample_ids"]
                ):
                    raise ExportError(
                        f"Preprocessing changed order for {dataset_spec['name']}:{split}"
                    )
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config["logical_batch_size"],
                    shuffle=False,
                    drop_last=False,
                    num_workers=config["num_workers"],
                )
                cursor = 0
                for index, (images, labels) in enumerate(loader):
                    size = len(labels)
                    sample_ids = catalog["sample_ids"][cursor : cursor + size]
                    sample_indices = catalog["sample_indices"][cursor : cursor + size]
                    labels = np.asarray(labels, dtype=np.int64)
                    if not np.array_equal(
                        labels, catalog["labels"][cursor : cursor + size]
                    ):
                        raise ExportError(
                            f"Labels changed for {dataset_spec['name']}:{split} batch {index}"
                        )
                    paths = {
                        kind: output_dir(config, model, dataset_spec, kind)
                        / batch_name(split, index)
                        for kind in config["transforms"]
                    }
                    transform_hashes = {"none": "none"}
                    transform_hashes.update(
                        {kind: value["sha256"] for kind, value in transforms.items()}
                    )
                    if all(
                        batch_is_valid(
                            paths[kind], sample_ids, kind, transform_hashes[kind]
                        )
                        for kind in config["transforms"]
                    ):
                        cursor += size
                        for kind, path in paths.items():
                            metadata[kind].append({"file": path.name, "count": size})
                        continue
                    raw, used_batch_size = extract_features(
                        extractor,
                        images,
                        model["layer"],
                        config["compute_batch_size"],
                        bool(model.get("extract_cls_token")),
                    )
                    variants = {}
                    for kind in config["transforms"]:
                        variants[kind] = (
                            raw
                            if kind == "none"
                            else apply_transform(raw, transforms[kind])
                        )
                    for kind, features in variants.items():
                        transform = None if kind == "none" else transforms[kind]
                        values = {
                            "features": features,
                            "labels": labels,
                            "sample_ids": sample_ids,
                            "sample_indices": sample_indices,
                            "model": text(model["name"]),
                            "dataset": text(dataset_spec["name"]),
                            "split": text(split),
                            "transform": text(kind),
                            "layer": text(model["layer"]),
                            "compute_batch_size": np.asarray(used_batch_size),
                            "transform_sha256": text(
                                "none" if transform is None else transform["sha256"]
                            ),
                            "performance_json": text(
                                json.dumps({"status": "not-attached"}, sort_keys=True)
                            ),
                        }
                        write_npz(paths[kind], values)
                        metadata[kind].append({"file": paths[kind].name, "count": size})
                    cursor += size
                if cursor != len(catalog["labels"]):
                    raise ExportError(
                        f"Extraction count differs for {dataset_spec['name']}:{split}"
                    )
            for kind in config["transforms"]:
                transform = None if kind == "none" else transforms[kind]
                write_json(
                    output_dir(config, model, dataset_spec, kind) / "metadata.json",
                    {
                        "model": model["name"],
                        "source": model["source"],
                        "layer": model["layer"],
                        "dataset": dataset_spec["name"],
                        "transform": kind,
                        "transform_path": (
                            None if transform is None else str(transform["path"])
                        ),
                        "transform_sha256": (
                            "none" if transform is None else transform["sha256"]
                        ),
                        "logical_batch_size": config["logical_batch_size"],
                        "batches": metadata[kind],
                    },
                )
        del extractor


def attach_performance(config: Mapping[str, Any], performance_path: Path) -> None:
    """Attach user-produced AD/FS summaries to batches and transform metadata."""
    with performance_path.open("r", encoding="utf-8") as handle:
        performance = json.load(handle)
    for model in config["models"]:
        for dataset in config["datasets"]:
            for kind in config.get("transforms", TRANSFORMS):
                try:
                    summary = performance[model["name"]][dataset["name"]][kind]
                except KeyError as exc:
                    raise ExportError(
                        f"Performance JSON is missing {model['name']}/{dataset['name']}/{kind}"
                    ) from exc
                directory = output_dir(config, model, dataset, kind)
                encoded = text(json.dumps(summary, sort_keys=True))
                for path in sorted(directory.glob("*-batch-*.npz")):
                    values = read_npz(path)
                    values["performance_json"] = encoded
                    write_npz(path, values)
                metadata = read_json(directory / "metadata.json")
                metadata["performance"] = summary
                write_json(directory / "metadata.json", metadata)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(config: Mapping[str, Any]) -> None:
    """Validate completeness, alignment, dimensions, dtype, and finite values."""
    errors: List[str] = []
    for dataset in config["datasets"]:
        for split in dataset["splits"]:
            catalog = read_npz(catalog_path(config, dataset, split))
            batch_count = math.ceil(
                len(catalog["labels"]) / config["logical_batch_size"]
            )
            for model in config["models"]:
                dimensions = set()
                for kind in config.get("transforms", TRANSFORMS):
                    directory = output_dir(config, model, dataset, kind)
                    for index in range(batch_count):
                        path = directory / batch_name(split, index)
                        if not path.is_file():
                            errors.append(f"Missing {path}")
                            continue
                        values = read_npz(path)
                        start = index * config["logical_batch_size"]
                        stop = min(
                            start + config["logical_batch_size"], len(catalog["labels"])
                        )
                        if not np.array_equal(
                            values["sample_ids"], catalog["sample_ids"][start:stop]
                        ):
                            errors.append(f"Misaligned sample IDs in {path}")
                        if not np.array_equal(
                            values["labels"], catalog["labels"][start:stop]
                        ):
                            errors.append(f"Misaligned labels in {path}")
                        if str(values["transform"].item()) != kind:
                            errors.append(f"Wrong transform metadata in {path}")
                        if values["features"].dtype != np.float32:
                            errors.append(f"Non-float32 features in {path}")
                        if not np.all(np.isfinite(values["features"])):
                            errors.append(f"Non-finite features in {path}")
                        dimensions.add(values["features"].shape[1])
                if len(dimensions) != 1:
                    errors.append(
                        f"Feature dimensions differ for {model['name']}/{dataset['name']}"
                    )
    if errors:
        raise ExportError("Validation failed:\n- " + "\n- ".join(errors))


def parse_args() -> argparse.Namespace:
    """Parse command-line stage and selection overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=("extract", "attach-performance", "validate"),
        default="extract",
    )
    parser.add_argument("--performance", type=Path)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument(
        "--transform",
        action="append",
        dest="transforms",
        choices=TRANSFORMS,
        help="representation variant to export; repeat to select more than one",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested stage and convert expected failures into exit status 1."""
    args = parse_args()
    try:
        config = load_config(args.config, args.models, args.datasets, args.transforms)
        if args.stage == "extract":
            extract(config)
        elif args.stage == "attach-performance":
            if args.performance is None:
                raise ExportError("--performance is required for attach-performance")
            attach_performance(config, args.performance.resolve())
        else:
            validate(config)
    except (ExportError, OSError, ValueError, KeyError) as exc:
        print(f"representation export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
