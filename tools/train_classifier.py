#!/usr/bin/env python3
"""Reproduce the selected logistic head on frozen AdaFace + R50 training features."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new classifier.npz path.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path; the deployed model is not overwritten.")
    feature_path = _bootstrap.ROOT / "data/training/features_and_splits.npz"
    with np.load(feature_path, allow_pickle=False) as a:
        fit = np.concatenate([a["train_indices"], a["validation_indices"]])
        if set(fit) & set(a["test_indices"]):
            raise ValueError("Training indices overlap the holdout.")
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1500,
                        random_state=20261014,
                    ),
                ),
            ]
        ).fit(a["features"][fit], a["labels"][fit])
        scaler, head = model.named_steps["scale"], model.named_steps["classifier"]
        np.testing.assert_array_equal(head.classes_, np.arange(len(a["class_names"])))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            mean=scaler.mean_,
            scale=scaler.scale_,
            coefficients=head.coef_,
            intercept=head.intercept_,
            class_names=a["class_names"],
        )
    print(f"Saved the selected classifier head to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
