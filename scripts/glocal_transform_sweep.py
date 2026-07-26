#!/usr/bin/env python3
"""Run and validate a configurable raw-ImageNet gLocal transform sweep.

The published ``main_glocal_probing.py`` module owns the training behavior. This
wrapper supplies one parameter tuple at a time, validates all inputs before the
heavy imports, and atomically publishes a transform only after training succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

LAMBDAS = ("0.1", "0.001")
ALPHAS = ("0.05", "0.1", "0.25", "0.5")
TAUS = ("0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1.0")
LEARNING_RATE = "0.001"
OPTIMIZER = "SGD"
REGULARIZATION = "eye"
SIGMA = "0.001"
TRIPLET_BATCH_SIZE = 256
CONTRASTIVE_BATCH_SIZE = 1024
MAX_EPOCHS = 100
BURNIN = 20
PATIENCE = 20
SEED = 42
N_SPLITS = 3
SCHEMA_VERSION = 1
SLUG_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class SweepError(RuntimeError):
    """Raised when a sweep configuration, input, or artifact is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    slug: str
    source: str = "custom"
    module: str = "penultimate"


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    model: str
    model_slug: str
    source: str
    module: str
    lambda_label: str
    alpha_label: str
    tau_label: str

    @property
    def lmbda(self) -> float:
        return float(self.lambda_label)

    @property
    def alpha(self) -> float:
        return float(self.alpha_label)

    @property
    def tau(self) -> float:
        return float(self.tau_label)


@dataclass
class PreparedInputs:
    repo_root: Path
    data_root: Path
    features_path: Path
    imagenet_root: Path
    model_dict_path: Path
    probing_base: Path
    feature_matrix: np.ndarray
    input_metadata: Dict[str, Any]
    repo_revision: str


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepError(f"Cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise SweepError(f"Expected a JSON object in {path}")
    return value


def default_slug(name: str) -> str:
    return name.replace("/", "-")


def load_models(config_path: Path) -> tuple[ModelSpec, ...]:
    """Load and validate the ordered model subset from a sweep configuration."""
    config_path = config_path.expanduser().resolve()
    config = read_json(config_path)
    raw_models = config.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise SweepError(f"'models' must be a non-empty list in {config_path}")

    models = []
    for index, value in enumerate(raw_models):
        if isinstance(value, str):
            model = ModelSpec(name=value, slug=default_slug(value))
        elif isinstance(value, dict):
            name = value.get("name")
            if not isinstance(name, str) or not name:
                raise SweepError(f"models[{index}].name must be a non-empty string")
            slug = value.get("slug", default_slug(name))
            source = value.get("source", "custom")
            module = value.get("module", "penultimate")
            if not all(isinstance(item, str) for item in (slug, source, module)):
                raise SweepError(f"models[{index}] fields must be strings")
            model = ModelSpec(name=name, slug=slug, source=source, module=module)
        else:
            raise SweepError(f"models[{index}] must be a string or object")

        if not SLUG_PATTERN.fullmatch(model.slug):
            raise SweepError(
                f"Filesystem-unsafe model slug {model.slug!r}; use letters, numbers, "
                "periods, underscores, or hyphens"
            )
        if model.source != "custom" or model.module != "penultimate":
            raise SweepError(
                f"{model.name}: this CLIP sweep requires source='custom' and "
                "module='penultimate'"
            )
        models.append(model)

    names = [model.name for model in models]
    slugs = [model.slug for model in models]
    if len(names) != len(set(names)):
        raise SweepError("Model names must be unique")
    if len(slugs) != len(set(slugs)):
        raise SweepError("Model slugs must be unique")
    return tuple(models)


def task_count(models: Sequence[ModelSpec]) -> int:
    return len(models) * len(LAMBDAS) * len(ALPHAS) * len(TAUS)


def all_specs(models: Sequence[ModelSpec]) -> Iterable[TaskSpec]:
    task_id = 0
    for model in models:
        for lambda_label in LAMBDAS:
            for alpha_label in ALPHAS:
                for tau_label in TAUS:
                    yield TaskSpec(
                        task_id=task_id,
                        model=model.name,
                        model_slug=model.slug,
                        source=model.source,
                        module=model.module,
                        lambda_label=lambda_label,
                        alpha_label=alpha_label,
                        tau_label=tau_label,
                    )
                    task_id += 1


def task_spec(models: Sequence[ModelSpec], task_id: int) -> TaskSpec:
    count = task_count(models)
    if not 0 <= task_id < count:
        raise SweepError(f"Task id must be in [0, {count - 1}], got {task_id}")
    return next(spec for spec in all_specs(models) if spec.task_id == task_id)


def validate_stop_policy(burnin: int, patience: int) -> None:
    """Prevent the Lightning 1.8 pre-min-epochs validation loop."""
    if burnin < 0 or patience < 0:
        raise SweepError("Burn-in and patience must be non-negative")
    if patience < burnin:
        raise SweepError(
            f"patience ({patience}) must be greater than or equal to burn-in "
            f"({burnin}); Lightning 1.8 can repeat validation indefinitely if "
            "early stopping is signaled before min_epochs"
        )


def artifact_stem(spec: TaskSpec) -> str:
    return f"lambda_{spec.lambda_label}_alpha_{spec.alpha_label}_tau_{spec.tau_label}"


def param_dir(probing_base: Path, spec: TaskSpec) -> Path:
    return probing_base / "selected" / "glocal" / spec.model_slug / "param_sweep"


def transform_path(probing_base: Path, spec: TaskSpec) -> Path:
    return param_dir(probing_base, spec) / f"transform_{artifact_stem(spec)}.npz"


def result_path(probing_base: Path, spec: TaskSpec) -> Path:
    return param_dir(probing_base, spec) / f"result_{artifact_stem(spec)}.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in (
        "numpy",
        "torch",
        "torchvision",
        "pytorch-lightning",
        "thingsvision",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_revision(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_npz(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SweepError(f"Transform is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            missing = {"weights", "mean", "std"} - names
            if missing:
                raise SweepError(f"{path} is missing arrays: {sorted(missing)}")
            if "bias" in names:
                raise SweepError(f"Unexpected bias in no-bias transform: {path}")
            weights = np.asarray(archive["weights"])
            mean = np.asarray(archive["mean"])
            std = np.asarray(archive["std"])
    except (OSError, ValueError) as error:
        raise SweepError(f"Cannot read transform {path}: {error}") from error

    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise SweepError(
            f"Transform weights must be square, got {weights.shape}: {path}"
        )
    if mean.size != 1 or std.size != 1:
        raise SweepError(f"Transform mean and std must be scalars: {path}")
    if not all(np.isfinite(value).all() for value in (weights, mean, std)):
        raise SweepError(f"Transform contains non-finite values: {path}")
    if float(std.reshape(-1)[0]) <= 0:
        raise SweepError(f"Transform std must be positive: {path}")
    return {"feature_dim": int(weights.shape[0]), "sha256": sha256(path)}


def expected_configuration(
    spec: TaskSpec, burnin: int, patience: int
) -> Dict[str, Any]:
    return {
        "model": spec.model,
        "model_slug": spec.model_slug,
        "source": spec.source,
        "module": spec.module,
        "lambda": spec.lmbda,
        "lambda_label": spec.lambda_label,
        "alpha": spec.alpha,
        "alpha_label": spec.alpha_label,
        "tau": spec.tau,
        "tau_label": spec.tau_label,
        "learning_rate": float(LEARNING_RATE),
        "optimizer": OPTIMIZER.lower(),
        "regularization": REGULARIZATION,
        "sigma": float(SIGMA),
        "triplet_batch_size": TRIPLET_BATCH_SIZE,
        "contrastive_batch_size": CONTRASTIVE_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "burnin": burnin,
        "patience": patience,
        "seed": SEED,
        "bias": False,
        "fold_policy": "first_deterministic_kfold_split",
        "n_splits": N_SPLITS,
    }


def validate_result(
    probing_base: Path,
    spec: TaskSpec,
    burnin: int = BURNIN,
    patience: int = PATIENCE,
    expected_inputs: Optional[Mapping[str, Any]] = None,
    expected_revision: Optional[str] = None,
) -> Dict[str, Any]:
    metadata_path = result_path(probing_base, spec)
    metadata = read_json(metadata_path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise SweepError(f"Schema mismatch in {metadata_path}")
    if metadata.get("configuration") != expected_configuration(spec, burnin, patience):
        raise SweepError(f"Training configuration mismatch in {metadata_path}")
    if expected_inputs is not None and metadata.get("inputs") != dict(expected_inputs):
        raise SweepError(f"Training input mismatch in {metadata_path}")
    if (
        expected_revision is not None
        and metadata.get("repo_revision") != expected_revision
    ):
        raise SweepError(f"Repository revision mismatch in {metadata_path}")

    npz = validate_npz(transform_path(probing_base, spec))
    if metadata.get("transform_sha256") != npz["sha256"]:
        raise SweepError(
            f"Transform checksum mismatch: {transform_path(probing_base, spec)}"
        )
    if metadata.get("feature_dim") != npz["feature_dim"]:
        raise SweepError(
            f"Transform dimension mismatch: {transform_path(probing_base, spec)}"
        )
    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        raise SweepError(f"Metrics are missing from {metadata_path}")
    for name in (
        "test_accuracy",
        "test_overall_loss",
        "test_triplet_loss",
        "test_contrastive_loss",
    ):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SweepError(f"Invalid {name} in {metadata_path}: {value!r}")
    return metadata


def atomic_json(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def save_npz(
    destination: Path,
    weights: np.ndarray,
    mean: Any,
    std: Any,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            weights=np.asarray(weights),
            mean=np.asarray(mean),
            std=np.asarray(std),
        )
    os.replace(temporary, destination)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SweepError(f"{description} is missing: {path}")


def reject_placeholder(path: Path, description: str) -> None:
    if str(path).startswith("/path/to"):
        raise SweepError(f"{description} is still a documentation placeholder: {path}")


def validate_triplets(path: Path, n_objects: int) -> None:
    require_file(path, "THINGS triplet file")
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise SweepError(f"Cannot read triplets {path}: {error}") from error
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise SweepError(f"Triplets must have non-empty shape (N, 3): {path}")
    if values.min() < 0 or values.max() >= n_objects:
        raise SweepError(f"Triplet indices in {path} are outside [0, {n_objects - 1}]")


def validate_imagenet_split(path: Path) -> None:
    if not path.is_dir():
        raise SweepError(f"ImageNet split directory is missing: {path}")
    class_directories = [item for item in path.iterdir() if item.is_dir()]
    if not class_directories:
        raise SweepError(f"ImageNet split has no class directories: {path}")
    if not any(
        candidate.is_file()
        for class_directory in class_directories
        for candidate in class_directory.rglob("*")
    ):
        raise SweepError(f"ImageNet split has no image files: {path}")


def check_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write-test-"):
            pass
    except OSError as error:
        raise SweepError(
            f"Output directory is not writable: {path}: {error}"
        ) from error


def input_metadata(
    config_path: Path,
    repo_root: Path,
    data_root: Path,
    features_path: Path,
    imagenet_root: Path,
    model_dict_path: Path,
) -> Dict[str, Any]:
    return {
        "paths": {
            "config": str(config_path),
            "repo_root": str(repo_root),
            "data_root": str(data_root),
            "features": str(features_path),
            "imagenet_root": str(imagenet_root),
            "model_dict": str(model_dict_path),
        },
        "sha256": {
            "config": sha256(config_path),
            "features": sha256(features_path),
            "train_triplets": sha256(data_root / "triplets" / "train_90.npy"),
            "test_triplets": sha256(data_root / "triplets" / "test_10.npy"),
            "model_dict": sha256(model_dict_path),
            "main_glocal_probing": sha256(repo_root / "main_glocal_probing.py"),
            "glocal_transform_sweep": sha256(Path(__file__).resolve()),
        },
    }


def preflight(
    args: argparse.Namespace,
    models: Sequence[ModelSpec],
    spec: TaskSpec,
) -> PreparedInputs:
    """Validate a task completely before importing Torch, Lightning, or thingsvision."""
    del models  # The selected task contains the relevant validated model entry.
    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    features_path = args.features.expanduser().resolve()
    imagenet_root = args.imagenet_root.expanduser().resolve()
    model_dict_path = args.model_dict.expanduser().resolve()
    probing_base = args.probing_base.expanduser().resolve()
    config_path = args.config.expanduser().resolve()

    for path, description in (
        (repo_root, "Repository root"),
        (data_root, "THINGS root"),
        (features_path, "THINGS features"),
        (imagenet_root, "ImageNet root"),
        (model_dict_path, "Model dictionary"),
        (probing_base, "Probing output root"),
    ):
        reject_placeholder(path, description)

    require_file(repo_root / "main_glocal_probing.py", "Published gLocal entry point")
    require_file(config_path, "Sweep configuration")
    require_file(features_path, "THINGS features")
    require_file(model_dict_path, "Model dictionary")
    validate_triplets(data_root / "triplets" / "train_90.npy", args.n_objects)
    validate_triplets(data_root / "triplets" / "test_10.npy", args.n_objects)
    validate_imagenet_split(imagenet_root / "train_set")
    validate_imagenet_split(imagenet_root / "val_set")

    model_dictionary = read_json(model_dict_path)
    try:
        module_name = model_dictionary[spec.model][spec.module]["module_name"]
    except (KeyError, TypeError) as error:
        raise SweepError(
            f"Model dictionary has no module_name for "
            f"{spec.model}/{spec.module}: {model_dict_path}"
        ) from error
    if not isinstance(module_name, str) or not module_name:
        raise SweepError(f"Invalid module_name for {spec.model}/{spec.module}")

    try:
        with features_path.open("rb") as source:
            features = pickle.load(source)
        feature_matrix = np.asarray(
            features[spec.source][spec.model][spec.module],
            dtype=np.float32,
        )
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise SweepError(
            f"Cannot load features[{spec.source!r}][{spec.model!r}]"
            f"[{spec.module!r}] from {features_path}: {error}"
        ) from error
    if feature_matrix.ndim != 2 or feature_matrix.shape[0] != args.n_objects:
        raise SweepError(
            f"Expected {args.n_objects} × D THINGS features for {spec.model}, "
            f"got {feature_matrix.shape}"
        )
    if not np.isfinite(feature_matrix).all():
        raise SweepError(f"THINGS features contain non-finite values: {spec.model}")

    check_writable(probing_base)
    metadata = input_metadata(
        config_path,
        repo_root,
        data_root,
        features_path,
        imagenet_root,
        model_dict_path,
    )
    return PreparedInputs(
        repo_root=repo_root,
        data_root=data_root,
        features_path=features_path,
        imagenet_root=imagenet_root,
        model_dict_path=model_dict_path,
        probing_base=probing_base,
        feature_matrix=feature_matrix,
        input_metadata=metadata,
        repo_revision=git_revision(repo_root),
    )


def import_upstream(repo_root: Path) -> Any:
    entrypoint = repo_root / "main_glocal_probing.py"
    module_name = "_published_main_glocal_probing"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise SweepError(f"Cannot import published gLocal entry point: {entrypoint}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_openclip_extractor_compat(upstream: Any) -> None:
    """Allow the published raw runner to load OpenCLIP dataset names with underscores.

    ``main_glocal_probing.py`` unpacks an OpenCLIP model name into exactly three
    underscore-separated tokens. The two published LAION identifiers have more
    tokens, so the original loader raises before creating the extractor. Keep the
    published loader for every other model and replace only this name parsing in
    the imported module object.
    """

    published_loader = upstream.load_extractor

    def load_extractor(model_cfg: Mapping[str, str]) -> Any:
        model_name = model_cfg["model"]
        if not model_name.startswith("OpenCLIP"):
            return published_loader(model_cfg)
        tokens = model_name.split("_")
        if len(tokens) < 3:
            raise SweepError(f"Invalid OpenCLIP model identifier: {model_name}")
        name, variant = tokens[:2]
        data = "_".join(tokens[2:])
        return upstream.get_extractor(
            model_name=name,
            source=model_cfg["source"],
            device=model_cfg["device"],
            pretrained=True,
            model_parameters={"variant": variant, "dataset": data},
        )

    upstream.load_extractor = load_extractor


def finite_metric(results: Mapping[str, Any], source_name: str) -> float:
    values = results.get(source_name)
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise SweepError(
            f"Expected exactly one first-fold value for {source_name}, got {values!r}"
        )
    value = float(values[0])
    if not math.isfinite(value):
        raise SweepError(f"Non-finite training metric {source_name}: {value}")
    return value


def execute_upstream(
    args: argparse.Namespace,
    spec: TaskSpec,
    prepared: PreparedInputs,
    temporary: Path,
) -> tuple[Path, Dict[str, float]]:
    """Call the published raw-image training function once for one parameter tuple."""
    upstream = import_upstream(prepared.repo_root)
    install_openclip_extractor_compat(upstream)
    snapshot_dir = temporary / "snapshots"
    checkpoint_dir = temporary / "checkpoints"
    snapshot_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    upstream_args = SimpleNamespace(
        optim=OPTIMIZER,
        regularization=REGULARIZATION,
        triplet_batch_size=TRIPLET_BATCH_SIZE,
        epochs=MAX_EPOCHS,
        burnin=args.burnin,
        patience=args.patience,
        use_bias=False,
        log_dir=str(checkpoint_dir),
        model=spec.model,
        module=spec.module,
        sigma=float(SIGMA),
        model_dict_path=str(prepared.model_dict_path),
        source=spec.source,
        device=args.device,
    )
    optim_cfg = upstream.create_optimization_config(
        args=upstream_args,
        eta=float(LEARNING_RATE),
        lmbda=spec.lmbda,
        alpha=spec.alpha,
        tau=spec.tau,
        contrastive_batch_size=CONTRASTIVE_BATCH_SIZE,
        out_path=str(snapshot_dir),
    )
    model_cfg = upstream.create_model_config(upstream_args)
    upstream.seed_everything(SEED, workers=True)
    _choices, results, transform, mean, std = upstream.run(
        features=prepared.feature_matrix,
        imagenet_root=str(prepared.imagenet_root),
        data_root=str(prepared.data_root),
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        n_objects=args.n_objects,
        device=args.device,
        rnd_seed=SEED,
        num_processes=args.num_processes,
    )
    metrics = {
        "test_accuracy": finite_metric(results, "test_acc"),
        "test_overall_loss": finite_metric(results, "test_overall_loss"),
        "test_triplet_loss": finite_metric(results, "test_triplet_loss"),
        "test_contrastive_loss": finite_metric(results, "test_contrastive_loss"),
    }
    if "bias" in transform:
        raise SweepError("Published runner unexpectedly returned a bias")
    temporary_transform = temporary / "transform.npz"
    save_npz(temporary_transform, transform["weights"], mean, std)
    validate_npz(temporary_transform)
    return temporary_transform, metrics


def run_task(args: argparse.Namespace) -> int:
    validate_stop_policy(args.burnin, args.patience)
    models = load_models(args.config)
    spec = task_spec(models, args.task_id)
    prepared = preflight(args, models, spec)

    try:
        validate_result(
            prepared.probing_base,
            spec,
            args.burnin,
            args.patience,
            prepared.input_metadata,
            prepared.repo_revision,
        )
    except SweepError:
        pass
    else:
        print(
            f"Validated existing artifact; skipping task {spec.task_id}: "
            f"{transform_path(prepared.probing_base, spec)}"
        )
        return 0

    scratch_root = (
        args.scratch_root.expanduser().resolve() if args.scratch_root else None
    )
    if scratch_root is not None:
        check_writable(scratch_root)
    prefix = (
        f"glocal-{spec.model_slug}-lambda-{spec.lambda_label}-"
        f"alpha-{spec.alpha_label}-tau-{spec.tau_label}-"
    )
    with tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_root) as temporary_name:
        temporary = Path(temporary_name)
        print(json.dumps({"task": asdict(spec)}, indent=2), flush=True)
        temporary_transform, metrics = execute_upstream(args, spec, prepared, temporary)
        npz = validate_npz(temporary_transform)
        if npz["feature_dim"] != prepared.feature_matrix.shape[1]:
            raise SweepError(
                f"Transform dimension {npz['feature_dim']} does not match "
                f"feature dimension {prepared.feature_matrix.shape[1]}"
            )

        destination = transform_path(prepared.probing_base, spec)
        atomic_copy(temporary_transform, destination)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "configuration": expected_configuration(spec, args.burnin, args.patience),
            "feature_dim": npz["feature_dim"],
            "inputs": prepared.input_metadata,
            "metrics": metrics,
            "packages": package_versions(),
            "repo_revision": prepared.repo_revision,
            "transform_path": str(destination),
            "transform_sha256": sha256(destination),
        }
        atomic_json(metadata, result_path(prepared.probing_base, spec))
        validate_result(
            prepared.probing_base,
            spec,
            args.burnin,
            args.patience,
            prepared.input_metadata,
            prepared.repo_revision,
        )
        print(f"Published: {destination}")
    return 0


def validate_sweep(
    config: Path,
    probing_base: Path,
    burnin: int = BURNIN,
    patience: int = PATIENCE,
) -> int:
    validate_stop_policy(burnin, patience)
    models = load_models(config)
    probing_base = probing_base.expanduser().resolve()
    errors = []
    for spec in all_specs(models):
        try:
            validate_result(probing_base, spec, burnin, patience)
        except SweepError as error:
            errors.append(str(error))
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SweepError(f"gLocal sweep validation failed:\n{details}")
    print(f"Validated all {task_count(models)} configured gLocal transforms.")
    return 0


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    count_parser = subparsers.add_parser("count", help="print the task count")
    add_config_argument(count_parser)

    describe_parser = subparsers.add_parser("describe", help="print one task mapping")
    add_config_argument(describe_parser)
    describe_parser.add_argument("--task-id", type=int, required=True)

    run_parser = subparsers.add_parser("run", help="run one SLURM array task")
    add_config_argument(run_parser)
    run_parser.add_argument("--task-id", type=int, required=True)
    run_parser.add_argument("--repo-root", type=Path, required=True)
    run_parser.add_argument("--data-root", type=Path, required=True)
    run_parser.add_argument("--features", type=Path, required=True)
    run_parser.add_argument("--imagenet-root", type=Path, required=True)
    run_parser.add_argument("--model-dict", type=Path, required=True)
    run_parser.add_argument("--probing-base", type=Path, required=True)
    run_parser.add_argument("--scratch-root", type=Path)
    run_parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    run_parser.add_argument("--num-processes", type=int, default=4)
    run_parser.add_argument("--n-objects", type=int, default=1854)
    run_parser.add_argument("--burnin", type=int, default=BURNIN)
    run_parser.add_argument("--patience", type=int, default=PATIENCE)

    validate_parser = subparsers.add_parser(
        "validate", help="validate every configured transform"
    )
    add_config_argument(validate_parser)
    validate_parser.add_argument("--probing-base", type=Path, required=True)
    validate_parser.add_argument("--burnin", type=int, default=BURNIN)
    validate_parser.add_argument("--patience", type=int, default=PATIENCE)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "count":
            print(task_count(load_models(args.config)))
            return 0
        if args.command == "describe":
            print(
                json.dumps(
                    asdict(task_spec(load_models(args.config), args.task_id)),
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            return run_task(args)
        return validate_sweep(
            args.config,
            args.probing_base,
            args.burnin,
            args.patience,
        )
    except SweepError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
