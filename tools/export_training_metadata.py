#!/usr/bin/env python3
"""Export only aggregate counts from the frozen data actually used to fit the head."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export_training_metadata(features: Path, classifier: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classifier_digest = sha256(classifier)
    if classifier_digest != manifest["assets"]["classifier.npz"]:
        raise ValueError("Classifier checksum differs from the selected manifest.")
    with np.load(classifier, allow_pickle=False) as head:
        names = head["class_names"].tolist()
    with np.load(features, allow_pickle=False) as data:
        labels = data["labels"]
        splits = [data[key] for key in ("train_indices", "validation_indices", "test_indices")]
        if (
            data["class_names"].tolist() != names
            or len(names) != manifest["classes"]
            or labels.ndim != 1
            or labels.dtype.kind not in "iu"
            or any(x.ndim != 1 or x.dtype.kind not in "iu" for x in splits)
        ):
            raise ValueError("Model class order or split arrays do not match the frozen dataset.")
        all_indices = np.concatenate(splits)
        if len(all_indices) != len(labels) or not np.array_equal(
            np.sort(all_indices), np.arange(len(labels))
        ):
            raise ValueError("Invalid or overlapping train/validation/test split.")
        if np.any(labels < 0) or np.any(labels >= len(names)):
            raise ValueError("Invalid frozen labels.")
        fit = np.concatenate(splits[:2])
        if len(fit) != manifest["training_faces"]:
            raise ValueError("Actual fit count differs from the selected manifest.")
        counts = np.bincount(labels[fit], minlength=len(names))
        if np.any(counts <= 0):
            raise ValueError("A model identity has no actual training rows.")
        return {
            "schema_version": 1,
            "classifier_sha256": classifier_digest,
            "features_and_splits_sha256": sha256(features),
            "fit_splits": ["train", "validation"],
            "split_sizes": {
                "train": len(splits[0]),
                "validation": len(splits[1]),
                "test_excluded": len(splits[2]),
            },
            "count_unit": "aligned_face_crop",
            "identities": [
                {"name": str(name), "training_samples": int(count)}
                for name, count in zip(names, counts, strict=True)
            ],
        }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=root / "data/training/features_and_splits.npz"
    )
    parser.add_argument("--classifier", type=Path, default=root / "models/selected/classifier.npz")
    parser.add_argument("--manifest", type=Path, default=root / "models/selected/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_training_metadata(args.features, args.classifier, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
