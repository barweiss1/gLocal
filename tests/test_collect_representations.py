from __future__ import annotations

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import collect_representations as collect


class FakeExtractor:
    def extract_features(self, batches, module_name, flatten_acts=True):
        batch = batches[0]
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory")
        return batch.numpy() * 2


class CollectRepresentationsTests(unittest.TestCase):
    def test_combined_sweep_config_expands_all_glocal_variants(self):
        probing_root = Path("/tmp/example-probing-root").resolve()
        with patch.dict(os.environ, {"PROBING_BASE": str(probing_root)}):
            config = collect.load_config(
                Path("scripts/representation_export.clip-all-sweeps.json"),
                selected_models=["clip_RN50"],
                selected_datasets=["cifar100"],
            )

        self.assertEqual(len(config["transforms"]), 41)
        variant = "glocal-lambda-0.001-alpha-0.75-tau-1.0"
        self.assertEqual(config["transform_specs"][variant]["kind"], "glocal")
        self.assertEqual(
            config["models"][0][f"{variant}_transform"],
            probing_root
            / "selected"
            / "glocal"
            / "clip_RN50"
            / "param_sweep"
            / "transform_lambda_0.001_alpha_0.75_tau_1.0.npz",
        )

    def test_parameter_sweep_config_resolves_named_variants(self):
        probing_root = Path("/tmp/example-probing-root").resolve()
        with patch.dict(os.environ, {"PROBING_BASE": str(probing_root)}):
            config = collect.load_config(
                Path("scripts/representation_export.clip-param-sweep.json"),
                selected_models=["clip_RN50"],
                selected_datasets=["cifar100"],
            )

        self.assertEqual(len(config["transforms"]), 9)
        self.assertEqual(config["transform_specs"]["naive-lambda-0.1"]["kind"], "naive")
        self.assertEqual(
            config["transform_specs"]["global-lambda-10.0"]["kind"], "global"
        )
        self.assertEqual(
            config["models"][0]["naive-lambda-0.1_transform"],
            probing_root
            / "selected"
            / "naive"
            / "clip_RN50"
            / "param_sweep"
            / "transform_lambda_0.1.npz",
        )

    def test_parameter_sweep_config_can_filter_one_variant(self):
        with patch.dict(os.environ, {"PROBING_BASE": "/tmp/example-probing-root"}):
            config = collect.load_config(
                Path("scripts/representation_export.clip-param-sweep.json"),
                selected_models=["clip_RN50"],
                selected_datasets=["cifar100"],
                selected_transforms=["none", "global-lambda-1.0"],
            )
        self.assertEqual(config["transforms"], ["none", "global-lambda-1.0"])

    def test_transform_and_microbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transform.npz"
            np.savez_compressed(
                path,
                weights=np.eye(3, dtype=np.float32) * 2,
                mean=np.asarray(1, dtype=np.float32),
                std=np.asarray(2, dtype=np.float32),
            )
            transform = collect.load_transform(path)
            features = np.asarray([[1, 2, 3]], dtype=np.float32)
            np.testing.assert_allclose(
                collect.apply_transform(features, transform), [[0, 1, 2]]
            )

        images = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        extracted, batch_size = collect.extract_features(
            FakeExtractor(), images, "layer", batch_size=5, cls_token=False
        )
        self.assertEqual(batch_size, 2)
        np.testing.assert_array_equal(extracted, images.numpy() * 2)

    def test_naive_transform_uses_glocal_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats_path = root / "glocal.npz"
            np.savez_compressed(
                stats_path,
                weights=np.eye(2, dtype=np.float32),
                mean=np.asarray(1, dtype=np.float32),
                std=np.asarray(2, dtype=np.float32),
            )
            naive_path = root / "naive.pkl"
            with naive_path.open("wb") as handle:
                pickle.dump(
                    {"custom": {"clip": {"visual": np.eye(2, dtype=np.float32)}}},
                    handle,
                )
            transform = collect.load_naive_transform(
                naive_path,
                {"source": "custom", "name": "clip", "layer": "visual"},
                stats_path,
            )
            np.testing.assert_allclose(
                collect.apply_transform(
                    np.asarray([[1, 3]], dtype=np.float32), transform
                ),
                [[0, 1]],
            )

    def test_naive_transform_accepts_trained_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "naive.npz"
            np.savez_compressed(
                path,
                weights=np.eye(2, dtype=np.float32),
                mean=np.asarray(0, dtype=np.float32),
                std=np.asarray(1, dtype=np.float32),
            )
            transform = collect.load_model_transform({"naive_transform": path}, "naive")
            np.testing.assert_array_equal(transform["weights"], np.eye(2))

    def test_attach_and_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = {
                "name": "model",
                "slug": "model",
                "source": "test",
                "layer": "layer",
            }
            dataset = {
                "name": "cifar100",
                "slug": "cifar100",
                "splits": ["train", "test"],
            }
            config = {
                "features_root": root / "features",
                "logical_batch_size": 4,
                "models": [model],
                "datasets": [dataset],
            }
            for split in dataset["splits"]:
                catalog = {
                    "sample_ids": np.asarray([f"{split}/{i}" for i in range(4)]),
                    "labels": np.asarray([0, 0, 1, 1], dtype=np.int64),
                    "sample_indices": np.arange(4, dtype=np.int64),
                }
                collect.write_npz(collect.catalog_path(config, dataset, split), catalog)
                for kind in collect.TRANSFORMS:
                    path = collect.output_dir(
                        config, model, dataset, kind
                    ) / collect.batch_name(split, 0)
                    collect.write_npz(
                        path,
                        {
                            "features": np.ones((4, 2), dtype=np.float32),
                            "labels": catalog["labels"],
                            "sample_ids": catalog["sample_ids"],
                            "sample_indices": catalog["sample_indices"],
                            "transform": collect.text(kind),
                            "performance_json": collect.text(
                                json.dumps({"status": "not-attached"})
                            ),
                        },
                    )
                    collect.write_json(
                        path.parent / "metadata.json",
                        {"model": "model", "dataset": "cifar100"},
                    )

            performance = {
                "model": {
                    "cifar100": {
                        kind: {"ad": {"auroc": 0.9}, "fs": {"accuracy": 0.8}}
                        for kind in collect.TRANSFORMS
                    }
                }
            }
            performance_path = root / "performance.json"
            collect.write_json(performance_path, performance)
            collect.attach_performance(config, performance_path)
            collect.validate(config)

            sample = collect.read_npz(
                root
                / "features"
                / "model"
                / "cifar100"
                / "global"
                / "train-batch-000000.npz"
            )
            self.assertEqual(
                json.loads(str(sample["performance_json"].item()))["fs"]["accuracy"],
                0.8,
            )


if __name__ == "__main__":
    unittest.main()
