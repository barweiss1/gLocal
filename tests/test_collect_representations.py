from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
