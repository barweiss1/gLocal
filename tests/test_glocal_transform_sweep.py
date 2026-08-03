from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scripts import glocal_transform_sweep as sweep


class FakeUpstream:
    def __init__(self) -> None:
        self.run_calls = []
        self.optim_configs = []
        self.seed_calls = []
        self.KFold = mock.Mock(return_value="kfold")

    def create_optimization_config(self, **kwargs):
        self.optim_configs.append(kwargs)
        args = kwargs["args"]
        return {
            "optim": args.optim,
            "reg": args.regularization,
            "lr": kwargs["eta"],
            "lmbda": kwargs["lmbda"],
            "alpha": kwargs["alpha"],
            "tau": kwargs["tau"],
            "contrastive_batch_size": kwargs["contrastive_batch_size"],
            "triplet_batch_size": args.triplet_batch_size,
            "max_epochs": args.epochs,
            "min_epochs": args.burnin,
            "patience": args.patience,
            "use_bias": args.use_bias,
            "sigma": args.sigma,
            "out_path": kwargs["out_path"],
        }

    def seed_everything(self, seed, workers):
        self.seed_calls.append((seed, workers))

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        dimension = kwargs["features"].shape[1]
        results = {
            "test_acc": [0.61],
            "test_overall_loss": [0.91],
            "test_triplet_loss": [0.71],
            "test_contrastive_loss": [0.20],
        }
        transform = {"weights": np.eye(dimension, dtype=np.float32)}
        return np.asarray([0]), results, transform, np.float32(0), np.float32(1)


class GlocalTransformSweepTests(unittest.TestCase):
    def write_config(self, root: Path, models=None) -> Path:
        if models is None:
            models = [{"name": "clip_RN50", "slug": "clip_RN50"}]
        path = root / "sweep.json"
        path.write_text(json.dumps({"models": models}), encoding="utf-8")
        return path

    def test_task_mapping_has_64_stable_combinations_per_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(
                root,
                [
                    {"name": "clip_RN50", "slug": "clip_RN50"},
                    {"name": "clip_ViT-L/14", "slug": "clip_ViT-L-14"},
                ],
            )
            models = sweep.load_models(config)
            specs = list(sweep.all_specs(models))
            self.assertEqual(sweep.task_count(models), 128)
            self.assertEqual(len(specs), 128)
            combinations = {
                (
                    spec.model,
                    spec.lambda_label,
                    spec.alpha_label,
                    spec.tau_label,
                )
                for spec in specs
            }
            self.assertEqual(len(combinations), 128)
            self.assertEqual(sweep.task_spec(models, 10).model, "clip_RN50")
            self.assertEqual(sweep.task_spec(models, 10).lambda_label, "0.1")
            self.assertEqual(sweep.task_spec(models, 10).alpha_label, "0.5")
            self.assertEqual(sweep.task_spec(models, 10).tau_label, "0.5")
            self.assertEqual(sweep.task_spec(models, 32).lambda_label, "0.01")
            self.assertEqual(sweep.task_spec(models, 48).lambda_label, "1.0")
            with self.assertRaisesRegex(sweep.SweepError, r"\[0, 127\]"):
                sweep.task_spec(models, 128)

    def test_config_supports_model_subset_and_rejects_unsafe_slugs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root, ["clip_ViT-L/14"])
            models = sweep.load_models(config)
            self.assertEqual(models[0].slug, "clip_ViT-L-14")
            self.assertEqual(sweep.task_count(models), 64)

            invalid = self.write_config(
                root,
                [{"name": "clip_RN50", "slug": "../clip_RN50"}],
            )
            with self.assertRaisesRegex(sweep.SweepError, "Filesystem-unsafe"):
                sweep.load_models(invalid)

    def test_artifact_names_use_stable_decimal_labels(self) -> None:
        spec = sweep.TaskSpec(
            task_id=0,
            model="clip_RN50",
            model_slug="clip_RN50",
            source="custom",
            module="penultimate",
            module_name="visual",
            lambda_label="0.001",
            alpha_label="0.05",
            tau_label="0.025",
        )
        expected = "transform_lambda_0.001_alpha_0.05_tau_0.025.npz"
        self.assertEqual(sweep.transform_path(Path("/output"), spec).name, expected)
        self.assertEqual(
            sweep.result_path(Path("/output"), spec).name,
            "result_lambda_0.001_alpha_0.05_tau_0.025.json",
        )

    def test_stop_policy_fails_before_any_input_access(self) -> None:
        sweep.validate_stop_policy(10, 10)
        with self.assertRaisesRegex(sweep.SweepError, "repeat validation indefinitely"):
            sweep.run_task(argparse.Namespace(burnin=10, patience=9))
        with self.assertRaisesRegex(sweep.SweepError, "cannot exceed max epochs"):
            sweep.validate_stop_policy(11, 11)

    def test_efficient_runner_preserves_three_way_fold(self) -> None:
        upstream = FakeUpstream()
        published_kfold = upstream.KFold
        sweep.install_three_fold_compat(upstream)
        self.assertEqual(
            upstream.KFold(n_splits=4, random_state=42, shuffle=True),
            "kfold",
        )
        published_kfold.assert_called_once_with(
            n_splits=3,
            random_state=42,
            shuffle=True,
        )

    def make_run_fixture(self, root: Path) -> argparse.Namespace:
        config = self.write_config(root)
        data_root = root / "things"
        triplet_root = data_root / "triplets"
        triplet_root.mkdir(parents=True)
        triplets = np.asarray([[0, 1, 2]], dtype=np.int64)
        np.save(triplet_root / "train_90.npy", triplets)
        np.save(triplet_root / "test_10.npy", triplets)

        features_path = data_root / "features.pkl"
        with features_path.open("wb") as output:
            pickle.dump(
                {
                    "custom": {
                        "clip_RN50": {
                            "penultimate": np.asarray(
                                [[1, 2], [3, 4], [5, 6]], dtype=np.float32
                            )
                        }
                    }
                },
                output,
            )

        imagenet_features_base = root / "imagenet-features"
        imagenet_model_root = imagenet_features_base / "clip_RN50"
        split_metadata = {}
        for split, count in (("train", 5), ("val", 3)):
            feature_file = imagenet_model_root / split / "features.hdf5"
            feature_file.parent.mkdir(parents=True)
            with h5py.File(feature_file, "w") as archive:
                archive.create_dataset(
                    "features",
                    data=np.ones((count, 2), dtype=np.float32),
                )
            split_metadata[split] = {
                "count": count,
                "feature_dim": 2,
                "dtype": "float32",
                "key": "features",
                "bytes": feature_file.stat().st_size,
                "sha256": "fixture-checksum",
            }
        sweep.atomic_json(
            {
                "configuration": {
                    "model": "clip_RN50",
                    "model_slug": "clip_RN50",
                    "source": "custom",
                    "module": "penultimate",
                    "module_name": "visual",
                    "format": "hdf5",
                },
                "features": split_metadata,
            },
            imagenet_model_root / "manifest.json",
        )
        repo_root = Path(__file__).resolve().parents[1]
        return argparse.Namespace(
            config=config,
            task_id=0,
            repo_root=repo_root,
            data_root=data_root,
            features=features_path,
            imagenet_features_base=imagenet_features_base,
            probing_base=root / "published",
            scratch_root=root / "scratch",
            device="gpu",
            feature_workers=2,
            n_objects=3,
            burnin=10,
            patience=10,
        )

    def test_run_publishes_and_then_resumes_exact_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_fixture(root)
            upstream = FakeUpstream()
            with mock.patch.object(sweep, "import_upstream", return_value=upstream):
                self.assertEqual(sweep.run_task(args), 0)
                spec = sweep.task_spec(sweep.load_models(args.config), 0)
                metadata_path = sweep.result_path(args.probing_base, spec)
                metadata = sweep.read_json(metadata_path)
                metadata["inputs"]["sha256"]["glocal_transform_sweep"] = (
                    "historical-wrapper-checksum"
                )
                sweep.atomic_json(metadata, metadata_path)
                self.assertEqual(sweep.run_task(args), 0)

            self.assertEqual(len(upstream.run_calls), 1)
            self.assertEqual(len(upstream.optim_configs), 1)
            optim_call = upstream.optim_configs[0]
            self.assertEqual(optim_call["eta"], 0.001)
            self.assertEqual(optim_call["lmbda"], 0.1)
            self.assertEqual(optim_call["alpha"], 0.1)
            self.assertEqual(optim_call["tau"], 0.1)
            self.assertEqual(optim_call["contrastive_batch_size"], 1024)
            self.assertEqual(upstream.run_calls[0]["rnd_seed"], 42)

            models = sweep.load_models(args.config)
            spec = sweep.task_spec(models, 0)
            metadata = sweep.validate_result(args.probing_base, spec)
            self.assertEqual(metadata["configuration"]["feature_workers"], 2)
            self.assertEqual(metadata["configuration"]["max_epochs"], 10)
            self.assertEqual(metadata["configuration"]["burnin"], 10)
            self.assertEqual(metadata["configuration"]["patience"], 10)
            self.assertEqual(metadata["configuration"]["optimizer"], "sgd")
            self.assertEqual(metadata["configuration"]["regularization"], "eye")
            self.assertEqual(metadata["configuration"]["module_name"], "visual")
            self.assertEqual(
                metadata["configuration"]["fold_policy"],
                "first_deterministic_kfold_split",
            )
            self.assertEqual(
                metadata["configuration"]["locality_input"],
                "precomputed_imagenet_features_hdf5",
            )
            self.assertEqual(metadata["metrics"]["test_accuracy"], 0.61)

    def test_corruption_and_metadata_mismatch_invalidate_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_fixture(root)
            upstream = FakeUpstream()
            with mock.patch.object(sweep, "import_upstream", return_value=upstream):
                sweep.run_task(args)

            spec = sweep.task_spec(sweep.load_models(args.config), 0)
            metadata_path = sweep.result_path(args.probing_base, spec)
            metadata = sweep.read_json(metadata_path)
            metadata["configuration"]["learning_rate"] = 0.01
            sweep.atomic_json(metadata, metadata_path)
            with self.assertRaisesRegex(sweep.SweepError, "configuration mismatch"):
                sweep.validate_result(args.probing_base, spec)

            transform = sweep.transform_path(args.probing_base, spec)
            np.savez_compressed(
                transform,
                weights=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
                mean=np.float32(0),
                std=np.float32(1),
            )
            with self.assertRaisesRegex(sweep.SweepError, "Unexpected bias"):
                sweep.validate_npz(transform)

    def test_preflight_rejects_missing_cached_features_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_run_fixture(Path(directory))
            missing = (
                args.imagenet_features_base / "clip_RN50" / "val" / "features.hdf5"
            )
            missing.unlink()
            with mock.patch.object(sweep, "import_upstream") as importer:
                with self.assertRaisesRegex(sweep.SweepError, "feature file"):
                    sweep.run_task(args)
                importer.assert_not_called()

    def test_cached_feature_dimension_must_match_things(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_run_fixture(Path(directory))
            feature_file = (
                args.imagenet_features_base / "clip_RN50" / "train" / "features.hdf5"
            )
            with h5py.File(feature_file, "w") as archive:
                archive.create_dataset(
                    "features",
                    data=np.ones((5, 3), dtype=np.float32),
                )
            with self.assertRaisesRegex(sweep.SweepError, "does not match THINGS"):
                sweep.run_task(args)


if __name__ == "__main__":
    unittest.main()
