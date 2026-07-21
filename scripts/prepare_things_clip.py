#!/usr/bin/env python3
"""Prepare the THINGS inputs needed to train the CLIP transforms.

The repository's THINGS loader reads local images and triplet arrays; its
``download`` flag only fetches the concept table. This wrapper downloads the
official THINGS files, arranges them in the layout expected by ``THINGSBehavior``,
and uses that loader to extract one feature vector per concept for the four CLIP
models used by the representation-export configuration.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONCEPTS_URL = (
    "https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/"
    "master/data/files/things_concepts.tsv"
)
TRAIN_TRIPLETS_URL = "https://osf.io/download/5mwts/"
TEST_TRIPLETS_URL = "https://osf.io/download/b2a4j/"
CC0_IMAGES_URL = "https://osf.io/download/wb36u/"
N_CONCEPTS = 1854

MODELS = (
    "clip_RN50",
    "clip_ViT-L/14",
    "OpenCLIP_ViT-L-14_laion400m_e32",
    "OpenCLIP_ViT-L-14_laion2b_s32b_b82k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--use-existing-images",
        action="store_true",
        help="Do not download CC0 images; require <data-root>/images/*.jpg.",
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    """Download a file atomically, reusing a completed destination."""
    if destination.is_file():
        print(f"Using existing download: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "gLocal/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        copied = 0
        next_report = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if total and copied >= next_report:
                print(f"  {destination.name}: {100 * copied / total:5.1f}%", flush=True)
                next_report = copied + max(total // 20, 1)
    os.replace(partial, destination)


def prepare_triplets(data_root: Path) -> None:
    """Download the official split text files and convert them to NumPy arrays."""
    triplet_root = data_root / "triplets"
    triplet_root.mkdir(parents=True, exist_ok=True)
    files = (
        ("train_90", TRAIN_TRIPLETS_URL),
        ("test_10", TEST_TRIPLETS_URL),
    )
    for stem, url in files:
        output = triplet_root / f"{stem}.npy"
        if output.is_file():
            continue
        text_path = triplet_root / f"{stem}.txt"
        download(url, text_path)
        print(f"Converting triplets: {text_path}")
        triplets = np.loadtxt(text_path, dtype=np.int64)
        if triplets.ndim != 2 or triplets.shape[1] != 3:
            raise RuntimeError(f"Unexpected triplet shape in {text_path}: {triplets.shape}")
        if triplets.min() < 0 or triplets.max() >= N_CONCEPTS:
            raise RuntimeError(f"Triplet indices are outside [0, {N_CONCEPTS - 1}]")
        np.save(output, triplets)


def read_concept_ids(path: Path) -> list[str]:
    """Read unique IDs without requiring pandas during data preparation."""
    with path.open(encoding="utf-8") as table:
        header = table.readline().rstrip("\n").split("\t")
        index = header.index("uniqueID")
        ids = [line.rstrip("\n").split("\t")[index] for line in table if line.strip()]
    if len(ids) != N_CONCEPTS:
        raise RuntimeError(f"Expected {N_CONCEPTS} concepts, found {len(ids)}")
    return ids


def prepare_images(data_root: Path, use_existing: bool) -> None:
    """Create the flat ``images/<uniqueID>.jpg`` layout used by THINGSBehavior."""
    concepts = data_root / "concepts" / "things_concepts.tsv"
    download(CONCEPTS_URL, concepts)
    concept_ids = read_concept_ids(concepts)
    image_root = data_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in concept_ids if not (image_root / f"{name}.jpg").is_file()]
    if not missing:
        return
    if use_existing:
        raise RuntimeError(
            f"Missing {len(missing)} THINGS images under {image_root}; "
            "remove --use-existing-images to download the CC0 subset."
        )

    archive = data_root / "downloads" / "images_THINGSplus-CC0.zip"
    download(CC0_IMAGES_URL, archive)
    print(f"Extracting {archive}")
    with zipfile.ZipFile(archive) as source:
        members = {
            Path(info.filename).name: info
            for info in source.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".jpg")
        }
        for concept_id in missing:
            name = f"{concept_id}.jpg"
            if name not in members:
                raise RuntimeError(f"CC0 archive does not contain {name}")
            target = image_root / name
            temporary = target.with_suffix(".jpg.part")
            with source.open(members[name]) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            os.replace(temporary, target)


def load_saved_features(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"custom": {}}
    with path.open("rb") as source:
        features = pickle.load(source)
    if not isinstance(features, dict):
        raise RuntimeError(f"Invalid feature dictionary: {path}")
    features.setdefault("custom", {})
    return features


def save_features(features: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as destination:
        pickle.dump(features, destination, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def extract_clip_features(data_root: Path, output: Path, device: str, batch_size: int) -> None:
    """Use the repository loader to extract and save the four CLIP matrices."""
    from thingsvision.utils.data import DataLoader

    from data.things import THINGSBehavior
    from scripts.collect_representations import load_extractor

    saved = load_saved_features(output)
    for model_name in MODELS:
        existing = saved["custom"].get(model_name, {}).get("penultimate")
        if isinstance(existing, np.ndarray) and existing.shape[0] == N_CONCEPTS:
            print(f"Using existing features: {model_name}")
            continue

        print(f"Extracting THINGS features: {model_name}")
        extractor = load_extractor(
            {"name": model_name, "source": "custom"}, device=device
        )
        dataset = THINGSBehavior(
            root=str(data_root),
            aligned=False,
            download=False,
            transform=extractor.get_transformations(),
        )
        batches = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            backend=extractor.get_backend(),
        )
        values = extractor.extract_features(
            batches=batches, module_name="visual", flatten_acts=True
        )
        values = np.asarray(values, dtype=np.float32).reshape(N_CONCEPTS, -1)
        saved["custom"].setdefault(model_name, {})["penultimate"] = values
        save_features(saved, output)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = (args.features or data_root / "features.pkl").expanduser().resolve()
    prepare_triplets(data_root)
    prepare_images(data_root, args.use_existing_images)
    extract_clip_features(data_root, output, args.device, args.batch_size)
    print(f"THINGS data: {data_root}")
    print(f"CLIP features: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
