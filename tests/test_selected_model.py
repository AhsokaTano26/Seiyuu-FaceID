"""Local regression against the preserved experiment; no private data in CI."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from seiyuu_faceid.vision.classifier import IdentityClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/training"
BUNDLE = ROOT / "models/selected"


@pytest.mark.skipif(
    not (DATA / "features_and_splits.npz").is_file(), reason="Private training data is local-only"
)
def test_selected_onnx_classifier_preserves_experiment_predictions():
    classifier = IdentityClassifier(BUNDLE)
    with np.load(DATA / "features_and_splits.npz", allow_pickle=False) as a:
        test = a["test_indices"]
        with np.load(BUNDLE / "holdout_scores.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(a["sample_ids"][test], saved["sample_ids"])
            scores = classifier.head.predict_proba(a["features"][test])
            np.testing.assert_allclose(scores, saved["scores"], atol=2e-6, rtol=0)
            np.testing.assert_array_equal(scores.argmax(1), saved["scores"].argmax(1))
        rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines()]
        # Real crops catch channel swaps, backbone order and export regressions.
        for index in test[:8]:
            features = classifier.embed(cv2.imread(str(ROOT / rows[index]["crop_image"])))
            np.testing.assert_allclose(features, a["features"][index], atol=2e-5, rtol=0)
