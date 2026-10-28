"""Parity summaries must not hide changed decisions or expose private identities."""

import json

import numpy as np
import pytest

from tools.validate_native import check_model_metadata, compare_crops, compare_images


def test_native_metadata_rejects_swapped_class_labels_with_matching_model_id():
    expected = ["first identity", "second identity"]
    metadata = {"model_id": "same-model", "class_names": expected[::-1]}
    with pytest.raises(ValueError, match="class order"):
        check_model_metadata(metadata, "same-model", expected)


def test_native_metadata_accepts_matching_order_and_rejects_model_mismatch():
    expected = ["first identity", "second identity"]
    metadata = {"model_id": "same-model", "class_names": expected}
    check_model_metadata(metadata, "same-model", expected)
    with pytest.raises(ValueError, match="model identities"):
        check_model_metadata(metadata, "different-model", expected)


def test_crop_parity_flags_threshold_change_even_when_top1_is_unchanged():
    features = np.zeros((1, 1024), dtype=np.float32)
    expected = np.array([[0.950001, 0.049999]], dtype=np.float32)
    actual = np.array([[0.949999, 0.050001]], dtype=np.float32)
    result = compare_crops(features, features, expected, actual, 0.95)
    assert result["top1_matches"] == 1
    assert result["probabilities_allclose"] is True
    assert result["threshold_matches"] == 0
    assert result["passed"] is False


def test_crop_parity_checks_full_top5_order_and_finite_values():
    features = np.zeros((1, 1024), dtype=np.float32)
    expected = np.array([[0.6, 0.2, 0.1, 0.06, 0.04]], dtype=np.float32)
    actual = np.array([[0.6, 0.1, 0.2, 0.06, 0.04]], dtype=np.float32)
    result = compare_crops(features, features, expected, actual, 0.95)
    assert result["top1_matches"] == 1
    assert result["top5_matches"] == 0
    assert result["passed"] is False


def test_threshold_comparison_matches_python_float_candidates():
    features = np.zeros((1, 1024), dtype=np.float32)
    expected = np.array([[0.95, 0.05]], dtype=np.float32)
    actual = expected.tolist()  # Native JSON carries the exact float32 value as a double.
    assert float(expected[0, 0]) < 0.95
    result = compare_crops(features, features, expected, actual, 0.95)
    assert result["threshold_matches"] == 1
    assert result["passed"] is True


def face(score=0.96, identity="private person"):
    return {"bbox": [1, 2, 50, 60], "detector_score": 0.9,
            "quality_score": 1.0, "quality_reasons": [], "status": "recognized",
            "identity": identity, "probability_threshold": 0.95, "decision_reason": None,
            "score_type": "class_probability",
            "candidates": [{"identity": identity, "score": score}]}


def test_image_parity_catches_changed_candidate_and_does_not_expose_names():
    result = compare_images([[face()]], [[face(identity="another private person")]])
    assert result["face_count_matches"] == 1
    assert result["decisions_match"] == 0
    assert result["candidate_order_matches"] == 0
    assert result["passed"] is False
    assert "private person" not in json.dumps(result)


def test_image_parity_rejects_geometry_and_quality_drift():
    changed = face()
    changed["bbox"][0] += 1
    changed["quality_score"] = 0.8
    result = compare_images([[face()]], [[changed]])
    assert result["bbox_max_abs_difference"] == 1
    assert result["quality_max_abs_difference"] > 0.1
    assert result["passed"] is False


def test_image_parity_accepts_empty_images_and_equal_outputs():
    result = compare_images([[], [face()]], [[], [face()]])
    assert result["images"] == 2
    assert result["faces"] == 1
    assert result["passed"] is True
