from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scripts import precompute_imagenet_clip_features as precompute


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = []

    def extract(self, **kwargs) -> None:
        self.calls.append(kwargs)
        output = Path(kwargs["out_path"])
        for split, count in (("train", 5), ("val", 3)):
            path = output / split / "features.hdf5"
            path.parent.mkdir(parents=True)
            with h5py.File(path, "w") as archive:
                archive.create_dataset(
                    "features",
                    data=np.arange(count * 2, dtype=np.float32).reshape(count, 2),
                )


class PrecomputeImageNetClipFeatureTests(unittest.TestCase):
    def make_args(self, root: Path) -> argparse.Namespace:
        config = root / "sweep.json"
        config.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "name": "clip_RN50",
                            "slug": "clip_RN50",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        imagenet = root / "imagenet"
        for split in ("train_set", "val_set"):
            class_root = imagenet / split / "class-a"
            class_root.mkdir(parents=True)
            (class_root / "image.jpg").write_bytes(b"fixture")
        return argparse.Namespace(
            config=config,
            task_id=0,
            repo_root=Path(__file__).resolve().parents[1],
            imagenet_root=imagenet,
            output_root=root / "features",
            batch_size=128,
            workers=2,
            train_count=5,
            val_count=3,
            device="cuda",
        )

    def test_extract_publish_validate_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))
            upstream = FakeExtractor()
            with mock.patch.object(
                precompute,
                "import_extractor",
                return_value=upstream,
            ):
                self.assertEqual(precompute.run(args), 0)
                self.assertEqual(precompute.run(args), 0)

            self.assertEqual(len(upstream.calls), 1)
            self.assertEqual(upstream.calls[0]["batch_size"], 128)
            self.assertEqual(upstream.calls[0]["num_workers"], 2)
            model = precompute.sweep.load_models(args.config)[0]
            manifest = precompute.validate_model(
                args.config.resolve(),
                args.repo_root.resolve(),
                args.imagenet_root.resolve(),
                args.output_root.resolve(),
                model,
                args.batch_size,
                args.workers,
                args.train_count,
                args.val_count,
            )
            self.assertEqual(manifest["features"]["train"]["count"], 5)
            self.assertEqual(manifest["features"]["val"]["feature_dim"], 2)

    def test_invalid_task_fails_before_extractor_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))
            args.task_id = 1
            with mock.patch.object(precompute, "import_extractor") as importer:
                with self.assertRaisesRegex(precompute.sweep.SweepError, "Task id"):
                    precompute.run(args)
                importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
