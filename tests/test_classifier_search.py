"""Leakage, fold coverage and portable export guarantees for local classifier search."""

import json
from dataclasses import replace

import numpy as np
import pytest

from seiyuu_faceid.experiments import classifier_search as search
from seiyuu_faceid.experiments.sample_size import StudyData
from seiyuu_faceid.vision.classifier import LinearIdentityHead


def dataset():
    rng = np.random.default_rng(17)
    features, labels, rows, fit, test = [], [], [], [], []
    for label in range(4):
        for group in range(2 if label == 3 else 5):
            held = group == (1 if label == 3 else 4)
            for crop in range(1 if label == 3 else 2):
                index = len(rows)
                features.append(rng.normal(0, 0.15, 8) + np.eye(4, 8)[label] * 2)
                labels.append(label)
                rows.append(
                    dict(
                        sample_id=f"s{index}",
                        identity=str(label),
                        source_group=f"g{label}-{group}",
                        source_sha256=f"p{index}",
                        crop_sha256=f"c{index}",
                    )
                )
                (test if held else fit).append(index)
    return StudyData(
        np.array(features, dtype=np.float32),
        np.array(labels),
        list(map(str, range(4))),
        rows,
        np.array(fit),
        np.array(test),
    )


def test_grid_is_fixed_and_baseline_first():
    grid = search.candidate_grid()
    assert len(grid) == len(set(grid)) == 50
    assert grid[0] == search.SearchConfig("standard", 1.0, "balanced", 0.5, False)
    assert sum(c.group_equalized for c in grid) == 6


def test_transitive_book_components_and_mixed_labels_are_rejected():
    data = dataset()
    # Bridge two otherwise distinct components, then bridge to historical test.
    data.rows[0]["known_book_group"] = "book"
    data.rows[2]["known_book_group"] = "book"
    components = search.connected_components(data)
    assert components[0] == components[2]
    assert components[1] == components[3]
    data.rows[int(data.test_indices[0])]["known_book_group"] = "book"
    with pytest.raises(ValueError, match="development/test"):
        search.connected_components(data)
    data = dataset()
    data.rows[0]["known_book_group"] = "conflict"
    data.rows[10]["known_book_group"] = "conflict"
    with pytest.raises(ValueError, match="identit"):
        search.connected_components(data)


def test_folds_pin_singletons_cover_other_samples_once_and_never_touch_test():
    data = dataset()
    components = search.connected_components(data)
    folds, pinned = search.make_folds(data, components, seed=42)
    assert pinned.tolist() == [int(data.fit_indices[-1])]
    validation = np.concatenate([v for _, v in folds])
    assert set(validation) == set(data.fit_indices) - set(pinned)
    assert len(validation) == len(set(validation))
    for train, valid in folds:
        assert set(data.labels[train]) == set(range(4))
        assert not set(components[train]) & set(components[valid])
        assert not set(np.r_[train, valid]) & set(data.test_indices)
        assert set(pinned) <= set(train)
    again, _ = search.make_folds(data, components, seed=42)
    for first, second in zip(folds, again, strict=True):
        np.testing.assert_array_equal(first[1], second[1])


def test_fold_local_scaler_group_weights_and_export_parity(tmp_path):
    pytest.importorskip("sklearn")
    data = dataset()
    components = search.connected_components(data)
    train, valid = search.make_folds(data, components, seed=42)[0][0]
    config = search.SearchConfig("standard", 1.0, "balanced", 0.25, False)
    fitted = search.fit_head(data, train, components, config)
    np.testing.assert_allclose(fitted.mean, data.features[train].mean(0), atol=2e-7)
    altered = replace(data, features=data.features.copy())
    altered.features[valid] += 1000
    other = search.fit_head(altered, train, components, config)
    np.testing.assert_array_equal(fitted.mean, other.mean)
    np.testing.assert_array_equal(fitted.scale, other.scale)
    target = tmp_path / "head.npz"
    search.export_head(fitted, data.class_names, target, data.features)
    runtime = LinearIdentityHead(target)
    np.testing.assert_allclose(
        runtime.predict_proba(data.features),
        fitted.predict_proba(data.features),
        atol=2e-12,
        rtol=0,
    )
    weights = search.component_weights(data.labels[train], components[train], balanced=True)
    assert weights.mean() == pytest.approx(1)
    totals = [weights[data.labels[train] == label].sum() for label in range(4)]
    np.testing.assert_allclose(totals, totals[0])
    with pytest.raises(FileExistsError):
        search.export_head(fitted, data.class_names, target, data.features)


def test_search_selection_ignores_test_features_and_refuses_existing_output(tmp_path):
    pytest.importorskip("sklearn")
    data = dataset()
    components = search.connected_components(data)
    baseline = tmp_path / "baseline.npz"
    config = search.candidate_grid()[0]
    search.export_head(
        search.fit_head(data, data.fit_indices, components, config),
        data.class_names,
        baseline,
        data.features,
    )
    first = tmp_path / "first"
    configs = [config, search.SearchConfig("standard", 0.03, "balanced", 0.5)]
    original_baseline = baseline.read_bytes()
    result = search.run_search(data, baseline, first, candidates=configs, seed=42)
    assert result["winner"] == config.as_dict()
    protocol = json.loads((first / "protocol.json").read_text())
    assert protocol["historical_test_used_for_selection"] is False
    assert (first / "winner.json").is_file()
    assert baseline.read_bytes() == original_baseline
    assert (first / "selection_scores.npz").is_file()
    assert result["holdout"]["baseline"]["samples"] == len(data.test_indices)
    altered = replace(data, features=data.features.copy())
    altered.features[data.test_indices] *= -100
    second = search.run_search(altered, baseline, tmp_path / "second", candidates=configs, seed=42)
    assert second["winner"] == result["winner"]
    assert second["selection_metrics"] == result["selection_metrics"]
    with pytest.raises(FileExistsError):
        search.run_search(data, baseline, first, candidates=[config], seed=42)


def test_group_metric_does_not_let_large_components_dominate():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    metrics = search.score_metrics(labels, scores, components=np.array([0, 0, 0, 1]))
    assert metrics["top1_accuracy"] == 0.75
    assert metrics["group_macro_top1"] == 0.5
    assert metrics["source_components"] == 2
