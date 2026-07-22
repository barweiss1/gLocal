#!/usr/bin/env python3
"""Run, validate, and select the fixed-learning-rate CLIP transform sweep."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


MODELS = (
    ("clip_RN50", "clip_RN50"),
    ("clip_ViT-L/14", "clip_ViT-L-14"),
    ("OpenCLIP_ViT-L-14_laion400m_e32", "OpenCLIP_ViT-L-14_laion400m_e32"),
    (
        "OpenCLIP_ViT-L-14_laion2b_s32b_b82k",
        "OpenCLIP_ViT-L-14_laion2b_s32b_b82k",
    ),
)
KINDS = (("naive", "l2"), ("global", "eye"))
LAMBDAS = ("0.01", "0.1", "1.0", "10.0")
LEARNING_RATE = "0.001"
OPTIMIZER = "SGD"
N_FOLDS = 3
SEED = 42
SCHEMA_VERSION = 1


class SweepError(RuntimeError):
    """Raised when a sweep artifact or configuration is incomplete."""


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    model: str
    model_slug: str
    kind: str
    regularization: str
    lambda_label: str

    @property
    def lmbda(self) -> float:
        return float(self.lambda_label)


def task_spec(task_id: int) -> TaskSpec:
    """Map one SLURM array index to a unique model, transform, and lambda."""
    task_count = len(MODELS) * len(KINDS) * len(LAMBDAS)
    if not 0 <= task_id < task_count:
        raise SweepError(f"Task id must be in [0, {task_count - 1}], got {task_id}")
    model_index = task_id % len(MODELS)
    kind_index = (task_id // len(MODELS)) % len(KINDS)
    lambda_index = task_id // (len(MODELS) * len(KINDS))
    model, model_slug = MODELS[model_index]
    kind, regularization = KINDS[kind_index]
    return TaskSpec(
        task_id=task_id,
        model=model,
        model_slug=model_slug,
        kind=kind,
        regularization=regularization,
        lambda_label=LAMBDAS[lambda_index],
    )


def validate_stop_policy(burnin: int, patience: int) -> None:
    """Reject settings that can trap Lightning 1.8 in repeated validation."""
    if patience < burnin:
        raise SweepError(
            f"patience ({patience}) must be greater than or equal to burn-in "
            f"({burnin}); Lightning 1.8 can repeat validation indefinitely if "
            "early stopping is signaled before min_epochs"
        )


def param_dir(probing_base: Path, spec: TaskSpec) -> Path:
    return probing_base / "selected" / spec.kind / spec.model_slug / "param_sweep"


def transform_path(probing_base: Path, spec: TaskSpec) -> Path:
    return param_dir(probing_base, spec) / f"transform_lambda_{spec.lambda_label}.npz"


def result_path(probing_base: Path, spec: TaskSpec) -> Path:
    return param_dir(probing_base, spec) / f"result_lambda_{spec.lambda_label}.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "torch", "torchvision", "pytorch-lightning"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def validate_npz(path: Path) -> Dict[str, Any]:
    """Validate the no-bias transform format emitted by global probing."""
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
        raise SweepError(f"Transform weights must be square, got {weights.shape}: {path}")
    if mean.size != 1 or std.size != 1:
        raise SweepError(f"Transform mean and std must be scalars: {path}")
    if not np.isfinite(weights).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise SweepError(f"Transform contains non-finite values: {path}")
    if float(std.reshape(-1)[0]) <= 0:
        raise SweepError(f"Transform std must be positive: {path}")
    return {
        "feature_dim": int(weights.shape[0]),
        "sha256": sha256(path),
    }


def expected_metadata(spec: TaskSpec) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": spec.model,
        "model_slug": spec.model_slug,
        "kind": spec.kind,
        "regularization": spec.regularization,
        "lambda": spec.lmbda,
        "lambda_label": spec.lambda_label,
        "learning_rate": float(LEARNING_RATE),
        "optimizer": OPTIMIZER.lower(),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "bias": False,
    }


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepError(f"Cannot read metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise SweepError(f"Metadata must contain a JSON object: {path}")
    return value


def validate_result(probing_base: Path, spec: TaskSpec) -> Dict[str, Any]:
    """Validate a published lambda artifact and its exact training metadata."""
    metadata_path = result_path(probing_base, spec)
    if not metadata_path.is_file():
        raise SweepError(f"Result metadata is missing: {metadata_path}")
    metadata = read_json(metadata_path)
    for key, expected in expected_metadata(spec).items():
        if metadata.get(key) != expected:
            raise SweepError(
                f"Metadata mismatch for {key} in {metadata_path}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    npz = validate_npz(transform_path(probing_base, spec))
    if metadata.get("transform_sha256") != npz["sha256"]:
        raise SweepError(f"Transform checksum mismatch: {transform_path(probing_base, spec)}")
    if metadata.get("feature_dim") != npz["feature_dim"]:
        raise SweepError(f"Transform dimension mismatch: {transform_path(probing_base, spec)}")
    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        raise SweepError(f"Metrics are missing from {metadata_path}")
    for key in ("mean_cv_cross_entropy", "mean_cv_accuracy"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SweepError(f"Invalid {key} in {metadata_path}: {value!r}")
    return metadata


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def atomic_json(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def scalar(row: pd.Series, name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise SweepError(f"Non-finite {name} in probing results: {value}")
    return value


def read_upstream_result(path: Path, spec: TaskSpec) -> Dict[str, float]:
    if not path.is_file():
        raise SweepError(f"Upstream probing results are missing: {path}")
    frame = pd.read_pickle(path)
    if len(frame.index) != 1:
        raise SweepError(f"Expected one probing result row, found {len(frame.index)}: {path}")
    row = frame.iloc[0]
    checks = {
        "model": spec.model,
        "module": "penultimate",
        "source": "custom",
        "reg": spec.regularization,
        "optim": OPTIMIZER.lower(),
        "n_folds": N_FOLDS,
        "bias": False,
    }
    for key, expected in checks.items():
        actual = row[key]
        if isinstance(expected, bool):
            actual = bool(actual)
        elif isinstance(expected, int):
            actual = int(actual)
        else:
            actual = str(actual)
        if actual != expected:
            raise SweepError(
                f"Unexpected {key} in probing results: expected {expected!r}, got {actual!r}"
            )
    if not math.isclose(float(row["lr"]), float(LEARNING_RATE), rel_tol=0, abs_tol=1e-12):
        raise SweepError(f"Unexpected learning rate in probing results: {row['lr']}")
    lambda_column = "lmbda" if "lmbda" in frame.columns else "lambda"
    if not math.isclose(float(row[lambda_column]), spec.lmbda, rel_tol=0, abs_tol=1e-12):
        raise SweepError(f"Unexpected lambda in probing results: {row[lambda_column]}")
    return {
        "mean_cv_cross_entropy": scalar(row, "cross-entropy"),
        "mean_cv_accuracy": scalar(row, "probing"),
    }


def build_probing_command(
    repo_root: Path,
    data_root: Path,
    probing_root: Path,
    log_dir: Path,
    spec: TaskSpec,
    args: argparse.Namespace,
) -> List[str]:
    """Build the published probing command for one validated sweep task."""
    return [
        sys.executable,
        str(repo_root / "scripts" / "run_global_probing_compat.py"),
        "--repo-root",
        str(repo_root),
        "--data_root",
        str(data_root),
        "--probing_root",
        str(probing_root),
        "--log_dir",
        str(log_dir),
        "--model",
        spec.model,
        "--source",
        "custom",
        "--module",
        "penultimate",
        "--n_folds",
        str(N_FOLDS),
        "--optim",
        OPTIMIZER,
        "--learning_rate",
        LEARNING_RATE,
        "--regularization",
        spec.regularization,
        "--lmbda",
        spec.lambda_label,
        "--sigma",
        args.sigma,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--burnin",
        str(args.burnin),
        "--patience",
        str(args.patience),
        "--rnd_seed",
        str(SEED),
        "--device",
        args.device,
    ]


def run_task(args: argparse.Namespace) -> int:
    validate_stop_policy(args.burnin, args.patience)
    spec = task_spec(args.task_id)
    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    features = args.features.expanduser().resolve()
    probing_base = args.probing_base.expanduser().resolve()
    required = (
        repo_root / "main_global_probing.py",
        repo_root / "scripts" / "run_global_probing_compat.py",
        data_root / "triplets" / "train_90.npy",
        data_root / "triplets" / "test_10.npy",
        features,
    )
    for path in required:
        if not path.is_file():
            raise SweepError(f"Required input is missing: {path}")

    try:
        existing = validate_result(probing_base, spec)
    except SweepError:
        existing = None
    if existing is not None:
        print(f"Validated existing artifact; skipping task {spec.task_id}: {transform_path(probing_base, spec)}")
        return 0

    scratch_root = args.scratch_root.expanduser().resolve() if args.scratch_root else None
    if scratch_root is not None:
        scratch_root.mkdir(parents=True, exist_ok=True)
    prefix = f"glocal-{spec.kind}-{spec.model_slug}-lambda-{spec.lambda_label}-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_root) as temporary_name:
        temporary = Path(temporary_name)
        probing_root = temporary / "probing"
        embeddings = probing_root / "embeddings"
        embeddings.mkdir(parents=True)
        (embeddings / "features.pkl").symlink_to(features)
        log_dir = temporary / "checkpoints"
        command = build_probing_command(
            repo_root,
            data_root,
            probing_root,
            log_dir,
            spec,
            args,
        )
        print(json.dumps({"task": asdict(spec), "command": command}, indent=2), flush=True)
        subprocess.run(command, cwd=repo_root, check=True)

        upstream_transform = (
            probing_root
            / "results"
            / "custom"
            / Path(spec.model)
            / "penultimate"
            / str(N_FOLDS)
            / spec.lambda_label
            / OPTIMIZER.lower()
            / LEARNING_RATE
            / "transform.npz"
        )
        npz = validate_npz(upstream_transform)
        metrics = read_upstream_result(
            probing_root / "results" / "probing_results.pkl", spec
        )
        destination = transform_path(probing_base, spec)
        atomic_copy(upstream_transform, destination)
        metadata = {
            **expected_metadata(spec),
            "feature_dim": npz["feature_dim"],
            "metrics": metrics,
            "packages": package_versions(),
            "transform_path": str(destination),
            "transform_sha256": sha256(destination),
        }
        atomic_json(metadata, result_path(probing_base, spec))
        validate_result(probing_base, spec)
        print(f"Published: {destination}")
    return 0


def all_specs() -> Iterable[TaskSpec]:
    for task_id in range(len(MODELS) * len(KINDS) * len(LAMBDAS)):
        yield task_spec(task_id)


def validate_candidates(probing_base: Path) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    errors: List[str] = []
    for spec in all_specs():
        try:
            metadata = validate_result(probing_base, spec)
        except SweepError as error:
            errors.append(str(error))
            continue
        grouped.setdefault((spec.kind, spec.model_slug), []).append(metadata)
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SweepError(f"Cannot select transforms; sweep is incomplete:\n{details}")
    return grouped


def select_transforms(probing_base: Path) -> int:
    probing_base = probing_base.expanduser().resolve()
    grouped = validate_candidates(probing_base)
    selections: List[tuple[Path, Path, Dict[str, Any]]] = []
    for kind, _regularization in KINDS:
        for model, model_slug in MODELS:
            candidates = grouped[(kind, model_slug)]
            candidates.sort(
                key=lambda item: (
                    item["metrics"]["mean_cv_cross_entropy"],
                    -item["metrics"]["mean_cv_accuracy"],
                    item["lambda"],
                )
            )
            winner = candidates[0]
            winner_spec = TaskSpec(
                task_id=-1,
                model=model,
                model_slug=model_slug,
                kind=kind,
                regularization=winner["regularization"],
                lambda_label=winner["lambda_label"],
            )
            source = transform_path(probing_base, winner_spec)
            destination = probing_base / "selected" / kind / model_slug / "transform.npz"
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "model": model,
                "model_slug": model_slug,
                "kind": kind,
                "regularization": winner["regularization"],
                "selection_rule": [
                    "lowest mean CV cross-entropy",
                    "highest mean CV accuracy",
                    "lowest lambda",
                ],
                "fixed_learning_rate": float(LEARNING_RATE),
                "optimizer": OPTIMIZER.lower(),
                "n_folds": N_FOLDS,
                "seed": SEED,
                "bias": False,
                "candidates": candidates,
                "selected_lambda": winner["lambda"],
                "selected_lambda_label": winner["lambda_label"],
                "selected_source": str(source),
                "transform_path": str(destination),
                "transform_sha256": winner["transform_sha256"],
            }
            selections.append((source, destination, manifest))

    for source, destination, manifest in selections:
        atomic_copy(source, destination)
        manifest["transform_sha256"] = sha256(destination)
        atomic_json(manifest, destination.parent / "manifest.json")
        print(f"Selected lambda={manifest['selected_lambda_label']}: {destination}")
    validate_selected(probing_base)
    return 0


def validate_selected(probing_base: Path) -> int:
    probing_base = probing_base.expanduser().resolve()
    errors: List[str] = []
    for kind, _regularization in KINDS:
        for model, model_slug in MODELS:
            directory = probing_base / "selected" / kind / model_slug
            selected = directory / "transform.npz"
            manifest_path = directory / "manifest.json"
            try:
                npz = validate_npz(selected)
                manifest = read_json(manifest_path)
                expected = {
                    "model": model,
                    "model_slug": model_slug,
                    "kind": kind,
                    "fixed_learning_rate": float(LEARNING_RATE),
                    "optimizer": OPTIMIZER.lower(),
                    "n_folds": N_FOLDS,
                    "seed": SEED,
                    "bias": False,
                }
                for key, value in expected.items():
                    if manifest.get(key) != value:
                        raise SweepError(
                            f"Selected manifest mismatch for {key} in {manifest_path}"
                        )
                if manifest.get("transform_sha256") != npz["sha256"]:
                    raise SweepError(f"Selected checksum mismatch: {selected}")
            except SweepError as error:
                errors.append(str(error))
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SweepError(f"Selected transform validation failed:\n{details}")
    print("Validated all eight selected naive/global transforms.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one SLURM array task")
    run_parser.add_argument("--task-id", type=int, required=True)
    run_parser.add_argument("--repo-root", type=Path, required=True)
    run_parser.add_argument("--data-root", type=Path, required=True)
    run_parser.add_argument("--features", type=Path, required=True)
    run_parser.add_argument("--probing-base", type=Path, required=True)
    run_parser.add_argument("--scratch-root", type=Path)
    run_parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    run_parser.add_argument("--batch-size", type=int, default=256)
    run_parser.add_argument("--epochs", type=int, default=100)
    run_parser.add_argument("--burnin", type=int, default=15)
    run_parser.add_argument("--patience", type=int, default=15)
    run_parser.add_argument("--sigma", default="0.001")

    select_parser = subparsers.add_parser("select", help="select the best lambda")
    select_parser.add_argument("--probing-base", type=Path, required=True)

    validate_parser = subparsers.add_parser(
        "validate-selected", help="validate the eight selected transforms"
    )
    validate_parser.add_argument("--probing-base", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return run_task(args)
        if args.command == "select":
            return select_transforms(args.probing_base)
        return validate_selected(args.probing_base)
    except (SweepError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
