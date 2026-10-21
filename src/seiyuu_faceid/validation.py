"""Closed-set classification metrics for the selected model's holdout."""

from __future__ import annotations

import numpy as np


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, int | float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if (
        labels.ndim != 1
        or scores.ndim != 2
        or len(labels) != len(scores)
        or not len(labels)
        or not scores.shape[1]
        or not np.isfinite(scores).all()
        or not np.issubdtype(labels.dtype, np.integer)
        or labels.min() < 0
        or labels.max() >= scores.shape[1]
    ):
        raise ValueError("Expected valid integer labels and finite N x classes scores.")
    predictions = scores.argmax(axis=1)
    supported = np.unique(labels)
    f1, recall = [], []
    for label in supported:
        tp = int(((predictions == label) & (labels == label)).sum())
        actual = int((labels == label).sum())
        predicted = int((predictions == label).sum())
        f1.append(2 * tp / (actual + predicted))
        recall.append(tp / actual)
    correct = int((predictions == labels).sum())
    top5 = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    return {
        "samples": len(labels),
        "supported_classes": len(supported),
        "top1_correct": correct,
        "errors": len(labels) - correct,
        "top1_accuracy": correct / len(labels),
        "top5_accuracy": float((top5 == labels[:, None]).any(axis=1).mean()),
        "macro_f1_supported_classes": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
    }
