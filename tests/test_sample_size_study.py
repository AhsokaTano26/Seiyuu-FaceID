"""Protect the sample-size experiment against leakage and optimistic aggregation."""

import numpy as np
import pytest

from seiyuu_faceid.experiments.sample_size import (
    StudyData,
    measure_identity,
    minimum_sustained_size,
    sample_order,
    select_fit_indices,
    validate_data,
)


def fixture_data():
    rows = [
        {
            "sample_id": f"sample-{i}",
            "identity": name,
            "source_group": group,
            "source_sha256": f"source-{i}",
            "crop_sha256": f"crop-{i}",
        }
        for i, (name, group) in enumerate(
            [
                ("alice", "a1"),
                ("alice", "a1"),
                ("alice", "a2"),
                ("bob", "b1"),
                ("bob", "b2"),
                ("alice", "at"),
                ("bob", "bt"),
            ]
        )
    ]
    return StudyData(
        features=np.arange(21, dtype=np.float32).reshape(7, 3),
        labels=np.array([0, 0, 0, 1, 1, 0, 1]),
        class_names=["alice", "bob"],
        rows=rows,
        fit_indices=np.arange(5),
        test_indices=np.array([5, 6]),
    )


def test_data_validation_rejects_source_leakage_and_label_mismatch():
    data = fixture_data()
    validate_data(data)
    data.rows[5]["source_sha256"] = data.rows[0]["source_sha256"]
    with pytest.raises(ValueError, match="source_sha256"):
        validate_data(data)
    data.rows[5]["source_sha256"] = "fixed"
    data.rows[0]["identity"] = "bob"
    with pytest.raises(ValueError, match="identity"):
        validate_data(data)


def test_sampling_counts_source_groups_once_and_is_reproducible():
    data = fixture_data()
    first = sample_order(data, data.fit_indices, 0, seed=42)
    assert len(first) == 2
    assert {data.rows[i]["source_group"] for i in first} == {"a1", "a2"}
    assert first == sample_order(data, data.fit_indices, 0, seed=42)


def test_nested_sampling_keeps_background_examples_and_excludes_test():
    data = fixture_data()
    orders = {0: sample_order(data, data.fit_indices, 0, seed=42)}
    one = select_fit_indices(data, orders, size=1)
    two = select_fit_indices(data, orders, size=2)
    assert set(one).issubset(two)
    assert set(one) & {3, 4} == {3, 4}
    assert not set(two) & {5, 6}
    assert (data.labels[one] == 0).sum() == 1
    assert (data.labels[two] == 0).sum() == 2
    with pytest.raises(ValueError, match="available"):
        select_fit_indices(data, orders, size=3)


def test_metrics_count_review_and_wrong_acceptances_as_failures():
    pytest.importorskip("scipy")
    result = measure_identity(
        np.array([0, 0, 0, 0, 1]),
        np.array([0, 0, 1, 0, 1]),
        np.array([0.95, 0.94, 0.99, 0.999, 0.999]),
        label=0,
        threshold=0.95,
    )
    assert result["test_count"] == 4
    assert result["top1_correct"] == 3
    assert result["correct_returned"] == 2
    assert result["wrong_returned"] == 1
    assert result["review"] == 1
    assert result["correct_return_rate"] == 0.5
    assert result["accepted_precision"] == pytest.approx(2 / 3)


def test_confidence_uses_independent_test_count_not_repeated_predictions():
    pytest.importorskip("scipy")
    one = measure_identity(np.array([0]), np.array([0]), np.array([1.0]), label=0, threshold=0.95)
    sixty = measure_identity(
        np.zeros(60, dtype=int), np.zeros(60, dtype=int), np.ones(60), label=0, threshold=0.95
    )
    assert one["lower95"] == pytest.approx(0.05)
    assert sixty["lower95"] == pytest.approx(0.05 ** (1 / 60))
    assert sixty["lower95"] >= 0.95


def test_minimum_requires_all_larger_sizes_to_pass():
    assert minimum_sustained_size([1, 5, 10, 20], [True, False, True, True]) == 10
    assert minimum_sustained_size([1, 5, 10], [True, True, False]) is None
    assert minimum_sustained_size([1, 5, 10], [False, False, True]) == 10
