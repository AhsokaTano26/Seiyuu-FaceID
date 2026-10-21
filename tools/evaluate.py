#!/usr/bin/env python3
"""Evaluate the selected classifier on its frozen, source-group-disjoint holdout."""

from __future__ import annotations

import argparse
import hashlib
import json

import _bootstrap  # noqa: F401
import numpy as np

from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.validation import classification_metrics
from seiyuu_faceid.vision.classifier import IdentityClassifier, LinearIdentityHead


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--reextract", action="store_true", help="Re-run both ONNX backbones.")
    args = parser.parse_args()
    config = load_config(_bootstrap.ROOT / args.config)
    training = config.root / "data/training"
    with np.load(training / "features_and_splits.npz", allow_pickle=False) as archive:
        test = archive["test_indices"]
        labels = archive["labels"][test]
        features = archive["features"][test]
        class_names = archive["class_names"].tolist()
        rows = [json.loads(line) for line in (training / "samples.jsonl").read_text().splitlines()]
        fit = np.concatenate([archive["train_indices"], archive["validation_indices"]])
        if set(fit) & set(test) or len(set(fit) | set(test)) != len(rows):
            raise ValueError("Train/holdout indices overlap or do not cover the dataset.")
        for field in ("sample_id", "source_sha256", "source_group", "crop_sha256"):
            if {rows[i][field] for i in fit} & {rows[i][field] for i in test}:
                raise ValueError(f"Train/holdout leakage in {field}.")
        for index, row in enumerate(rows):
            if row["identity"] != class_names[archive["labels"][index]]:
                raise ValueError("Training labels do not match the preserved manifest.")
        head = LinearIdentityHead(config.classifier_bundle / "classifier.npz")
        if head.class_names != class_names:
            raise ValueError("Model class order differs from dataset class order.")
        if args.reextract:
            import cv2

            classifier = IdentityClassifier(
                config.classifier_bundle,
                intra_op_threads=config.intra_op_threads,
                inter_op_threads=config.inter_op_threads,
            )
            features = []
            for index in test:
                path = config.root / rows[index]["crop_image"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != rows[index]["crop_sha256"]:
                    raise ValueError(f"Crop checksum mismatch: {path}")
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Could not decode {path}")
                features.append(classifier.embed(image))
            features = np.asarray(features)
        scores = head.predict_proba(features)
    metrics = classification_metrics(labels, scores)
    accepted = scores.max(axis=1) >= config.probability_threshold
    accepted_count = int(accepted.sum())
    correct_accepted = int(((scores.argmax(axis=1) == labels) & accepted).sum())
    metrics["acceptance"] = {
        "probability_threshold": config.probability_threshold,
        "accepted": accepted_count,
        "review": len(labels) - accepted_count,
        "coverage": accepted_count / len(labels),
        "accepted_correct": correct_accepted,
        "accepted_errors": accepted_count - correct_accepted,
        "accepted_precision": correct_accepted / accepted_count if accepted_count else None,
    }
    with np.load(config.classifier_bundle / "holdout_scores.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["labels"], labels)
        metrics["matches_recorded_predictions"] = bool(
            np.array_equal(scores.argmax(1), saved["scores"].argmax(1))
        )
    metrics["evaluation_scope"] = "exploratory reused holdout; closed-set classification"
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
