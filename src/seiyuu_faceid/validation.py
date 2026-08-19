"""Dependency-light verification and identification metrics."""

from __future__ import annotations

import numpy as np

from seiyuu_faceid.vision.recognizer import l2_normalize


def evaluate_pairs(left: np.ndarray, right: np.ndarray, same: np.ndarray) -> dict[str, object]:
    """Report pair score distributions and thresholds across observed scores.

    `same` is truthy for same-person pairs. Threshold suggestions maximize
    balanced accuracy on this supplied validation set; they are not universal.
    """
    left_vectors = np.asarray(left, dtype=np.float32)
    right_vectors = np.asarray(right, dtype=np.float32)
    labels = np.asarray(same, dtype=bool).reshape(-1)
    if left_vectors.shape != right_vectors.shape or left_vectors.ndim != 2:
        raise ValueError("left and right must have equal N×D shapes.")
    if len(left_vectors) != len(labels) or not labels.any() or labels.all():
        raise ValueError("Pairs need both same-person and different-person labels.")
    scores = np.asarray(
        [
            np.dot(l2_normalize(a), l2_normalize(b))
            for a, b in zip(left_vectors, right_vectors, strict=True)
        ]
    )
    thresholds = np.unique(scores)
    best: tuple[float, float, float, float] | None = None
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        accepted = scores >= threshold
        tar = float(accepted[labels].mean())
        far = float(accepted[~labels].mean())
        balanced_accuracy = (tar + (1 - far)) / 2
        curve.append({"threshold": float(threshold), "tar": tar, "far": far, "frr": 1 - tar})
        candidate = (balanced_accuracy, float(threshold), tar, far)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    same_scores = scores[labels]
    different_scores = scores[~labels]
    return {
        "pairs": int(len(scores)),
        "same_person": _distribution(same_scores),
        "different_person": _distribution(different_scores),
        "best_balanced_threshold": best[1],
        "best_balanced_tar": best[2],
        "best_balanced_far": best[3],
        "roc": curve,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def evaluate_identification(
    expected: list[str], rankings: list[list[str]], scores: np.ndarray
) -> dict[str, object]:
    """Summarise labelled top-k identification results without changing thresholds.

    ``scores`` contains the score of the top predicted identity.  The supplied
    images must be independent from the references used to build the Gallery.
    The suggested high threshold accepts no *observed* wrong top-1 result; it
    is a starting point for review, not a guarantee for unseen identities.
    """
    top_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not expected or len(expected) != len(rankings) or len(expected) != len(top_scores):
        raise ValueError("Expected labels, rankings and scores must have equal non-zero lengths.")
    if any(not ranking for ranking in rankings):
        raise ValueError("Every ranking must contain at least one candidate.")

    predicted = [ranking[0] for ranking in rankings]
    correct = np.asarray(
        [
            actual == predicted_identity
            for actual, predicted_identity in zip(expected, predicted, strict=True)
        ],
        dtype=bool,
    )
    top_five = np.asarray(
        [
            actual in ranking[:5]
            for actual, ranking in zip(expected, rankings, strict=True)
        ],
        dtype=bool,
    )
    correct_scores = top_scores[correct]
    wrong_scores = top_scores[~correct]
    per_identity: dict[str, dict[str, float | int]] = {}
    for identity in sorted(set(expected)):
        indexes = np.asarray([item == identity for item in expected], dtype=bool)
        per_identity[identity] = {
            "samples": int(indexes.sum()),
            "top1_accuracy": float(correct[indexes].mean()),
            "top5_accuracy": float(top_five[indexes].mean()),
        }

    threshold_advice: dict[str, float | None] = {
        "high_threshold_no_observed_misidentification": None,
        "low_threshold_preserving_99_percent_correct": None,
    }
    if len(wrong_scores):
        threshold_advice["high_threshold_no_observed_misidentification"] = float(
            np.nextafter(wrong_scores.max(), np.float32(1.0))
        )
    if len(correct_scores):
        threshold_advice["low_threshold_preserving_99_percent_correct"] = float(
            np.quantile(correct_scores, 0.01)
        )
    return {
        "samples": len(expected),
        "top1_accuracy": float(correct.mean()),
        "top5_accuracy": float(top_five.mean()),
        "correct_top1_score": _distribution(correct_scores) if len(correct_scores) else None,
        "wrong_top1_score": _distribution(wrong_scores) if len(wrong_scores) else None,
        "threshold_advice": threshold_advice,
        "per_identity": per_identity,
    }
