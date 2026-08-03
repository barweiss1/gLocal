from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts import clip_transform_sweep as sweep
from scripts.run_global_probing_compat import fold_isolated_trainer


class ClipTransformSweepTests(unittest.TestCase):
    def test_stop_policy_accepts_equal_values_and_rejects_early_patience(self) -> None:
        sweep.validate_stop_policy(burnin=15, patience=15)
        with self.assertRaisesRegex(sweep.SweepError, r"patience \(10\).+burn-in \(15\)"):
            sweep.validate_stop_policy(burnin=15, patience=10)

        # The guard is the first operation in run_task, before input paths are read.
        with self.assertRaisesRegex(sweep.SweepError, "repeat validation indefinitely"):
            sweep.run_task(argparse.Namespace(burnin=15, patience=10))

    def test_probing_command_uses_safe_stop_defaults(self) -> None:
        parser = sweep.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--task-id",
                "0",
                "--repo-root",
                "/repo",
                "--data-root",
                "/data",
                "--features",
                "/data/features.pkl",
                "--probing-base",
                "/output",
            ]
        )
        self.assertEqual(args.burnin, 15)
        self.assertEqual(args.patience, 15)
        command = sweep.build_probing_command(
            Path("/repo"),
            Path("/data"),
            Path("/scratch/probing"),
            Path("/scratch/checkpoints"),
            sweep.task_spec(0),
            args,
        )
        self.assertEqual(command[command.index("--burnin") + 1], "15")
        self.assertEqual(command[command.index("--patience") + 1], "15")

    def test_task_mapping_covers_cartesian_product_once(self) -> None:
        specs = [sweep.task_spec(index) for index in range(32)]
        combinations = {
            (spec.model, spec.kind, spec.lambda_label) for spec in specs
        }
        self.assertEqual(len(combinations), 32)
        self.assertEqual({spec.model for spec in specs}, {item[0] for item in sweep.MODELS})
        self.assertEqual({spec.kind for spec in specs}, {item[0] for item in sweep.KINDS})
        self.assertEqual({spec.lambda_label for spec in specs}, set(sweep.LAMBDAS))

    def test_callback_wrapper_does_not_mutate_shared_callbacks(self) -> None:
        class FakeTrainer:
            seen_callbacks = []

            def __init__(self, callbacks=None, **_kwargs):
                callbacks[0]["wait_count"] += 1
                self.seen_callbacks.append(callbacks)

        trainer = fold_isolated_trainer(FakeTrainer)
        original = [{"wait_count": 0}]
        trainer(callbacks=original)
        trainer(callbacks=original)

        self.assertEqual(original[0]["wait_count"], 0)
        self.assertEqual(FakeTrainer.seen_callbacks[0][0]["wait_count"], 1)
        self.assertEqual(FakeTrainer.seen_callbacks[1][0]["wait_count"], 1)
        self.assertIsNot(FakeTrainer.seen_callbacks[0], FakeTrainer.seen_callbacks[1])

    def publish_candidate(
        self,
        root: Path,
        spec: sweep.TaskSpec,
        cross_entropy: float,
        accuracy: float,
    ) -> None:
        path = sweep.transform_path(root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=np.eye(3, dtype=np.float32),
            mean=np.asarray(0, dtype=np.float32),
            std=np.asarray(1, dtype=np.float32),
        )
        metadata = {
            **sweep.expected_metadata(spec),
            "feature_dim": 3,
            "metrics": {
                "mean_cv_cross_entropy": cross_entropy,
                "mean_cv_accuracy": accuracy,
            },
            "packages": {},
            "transform_path": str(path),
            "transform_sha256": sweep.sha256(path),
        }
        sweep.atomic_json(metadata, sweep.result_path(root, spec))

    def make_run_fixture(self, root: Path) -> argparse.Namespace:
        data_root = root / "things"
        triplet_root = data_root / "triplets"
        triplet_root.mkdir(parents=True)
        (triplet_root / "train_90.npy").touch()
        (triplet_root / "test_10.npy").touch()
        features_path = data_root / "features.pkl"
        features_path.touch()
        return argparse.Namespace(
            task_id=0,
            repo_root=Path(__file__).resolve().parents[1],
            data_root=data_root,
            features=features_path,
            probing_base=root / "published",
            scratch_root=root / "scratch",
            device="gpu",
            batch_size=256,
            epochs=100,
            burnin=15,
            patience=15,
            sigma="0.001",
            overwrite=False,
        )

    def fake_training_run(self, spec: sweep.TaskSpec):
        """Fabricate the published probing outputs a real subprocess call would leave behind."""

        def run(command, cwd, check):
            del cwd, check
            probing_root = Path(command[command.index("--probing_root") + 1])
            transform_dir = (
                probing_root
                / "results"
                / "custom"
                / Path(spec.model)
                / "penultimate"
                / str(sweep.N_FOLDS)
                / spec.lambda_label
                / sweep.OPTIMIZER.lower()
                / sweep.LEARNING_RATE
            )
            transform_dir.mkdir(parents=True)
            np.savez_compressed(
                transform_dir / "transform.npz",
                weights=np.eye(3, dtype=np.float32),
                mean=np.asarray(0, dtype=np.float32),
                std=np.asarray(1, dtype=np.float32),
            )
            frame = pd.DataFrame(
                [
                    {
                        "model": spec.model,
                        "module": "penultimate",
                        "source": "custom",
                        "reg": spec.regularization,
                        "optim": sweep.OPTIMIZER.lower(),
                        "n_folds": sweep.N_FOLDS,
                        "bias": False,
                        "lr": float(sweep.LEARNING_RATE),
                        "lmbda": spec.lmbda,
                        "cross-entropy": 0.5,
                        "probing": 0.7,
                    }
                ]
            )
            frame.to_pickle(probing_root / "results" / "probing_results.pkl")

        return run

    def test_run_task_skips_when_valid_result_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_fixture(root)
            spec = sweep.task_spec(args.task_id)
            self.publish_candidate(args.probing_base, spec, 0.5, 0.7)

            with mock.patch.object(sweep.subprocess, "run") as run:
                self.assertEqual(sweep.run_task(args), 0)
                run.assert_not_called()

    def test_overwrite_flag_forces_retrain_over_valid_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_fixture(root)
            spec = sweep.task_spec(args.task_id)
            self.publish_candidate(args.probing_base, spec, 0.5, 0.7)
            args.overwrite = True

            with mock.patch.object(
                sweep.subprocess, "run", side_effect=self.fake_training_run(spec)
            ) as run:
                self.assertEqual(sweep.run_task(args), 0)
                run.assert_called_once()

            metadata = sweep.validate_result(args.probing_base, spec)
            self.assertEqual(metadata["metrics"]["mean_cv_accuracy"], 0.7)

    def test_selection_publishes_all_lambdas_and_best_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for spec in sweep.all_specs():
                cross_entropy = {
                    "0.01": 0.9,
                    "0.1": 0.8,
                    "1.0": 0.7,
                    "10.0": 0.7,
                }[spec.lambda_label]
                accuracy = 0.8 if spec.lambda_label in {"1.0", "10.0"} else 0.7
                self.publish_candidate(root, spec, cross_entropy, accuracy)

            self.assertEqual(sweep.validate_sweep(root), 0)
            self.assertEqual(sweep.select_transforms(root), 0)
            for kind, _regularization in sweep.KINDS:
                for _model, model_slug in sweep.MODELS:
                    directory_path = root / "selected" / kind / model_slug
                    self.assertTrue((directory_path / "transform.npz").is_file())
                    manifest = sweep.read_json(directory_path / "manifest.json")
                    self.assertEqual(manifest["selected_lambda_label"], "1.0")
                    self.assertEqual(len(manifest["candidates"]), 4)
                    for lambda_label in sweep.LAMBDAS:
                        self.assertTrue(
                            (
                                directory_path
                                / "param_sweep"
                                / f"transform_lambda_{lambda_label}.npz"
                            ).is_file()
                        )

    def test_validation_rejects_bias_and_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = sweep.task_spec(0)
            self.publish_candidate(root, spec, 0.5, 0.7)
            metadata_path = sweep.result_path(root, spec)
            metadata = sweep.read_json(metadata_path)
            metadata["optimizer"] = "adam"
            sweep.atomic_json(metadata, metadata_path)
            with self.assertRaises(sweep.SweepError):
                sweep.validate_result(root, spec)

            path = sweep.transform_path(root, spec)
            np.savez_compressed(
                path,
                weights=np.eye(3, dtype=np.float32),
                bias=np.zeros(3, dtype=np.float32),
                mean=np.asarray(0, dtype=np.float32),
                std=np.asarray(1, dtype=np.float32),
            )
            with self.assertRaises(sweep.SweepError):
                sweep.validate_npz(path)

    def test_selection_requires_complete_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.publish_candidate(root, sweep.task_spec(0), 0.5, 0.7)
            with self.assertRaisesRegex(sweep.SweepError, "sweep is incomplete"):
                sweep.validate_candidates(root)


if __name__ == "__main__":
    unittest.main()
