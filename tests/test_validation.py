from __future__ import annotations

import numpy as np

from seiyuu_faceid.validation import evaluate_identification, evaluate_pairs


def test_pair_evaluation_reports_separated_scores() -> None:
    left = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    right = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
    report = evaluate_pairs(left, right, np.array([True, True, False, False]))
    assert report["same_person"]["mean"] > report["different_person"]["mean"]
    assert report["best_balanced_tar"] == 1.0


def test_identification_evaluation_reports_accuracy_and_threshold_advice() -> None:
    report = evaluate_identification(
        ["a", "a", "b"],
        [["a", "b"], ["b", "a"], ["b", "a"]],
        np.array([0.92, 0.70, 0.81], dtype=np.float32),
    )
    assert report["top1_accuracy"] == 2 / 3
    assert report["top5_accuracy"] == 1.0
    advice = report["threshold_advice"]
    assert advice["high_threshold_no_observed_misidentification"] is not None
