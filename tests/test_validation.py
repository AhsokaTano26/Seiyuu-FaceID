import numpy as np
import pytest

from seiyuu_faceid.validation import classification_metrics


def test_metrics_include_supported_classes_with_no_correct_predictions():
    metrics = classification_metrics(
        np.array([0, 0, 1]), np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    )
    assert metrics["samples"] == 3
    assert metrics["supported_classes"] == 2
    assert metrics["top1_accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1_supported_classes"] == pytest.approx(0.4)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["top5_accuracy"] == 1


def test_metrics_reject_labels_outside_the_score_columns():
    with pytest.raises(ValueError, match="labels"):
        classification_metrics(np.array([2]), np.array([[0.5, 0.5]]))
