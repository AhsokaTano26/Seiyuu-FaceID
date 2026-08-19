from __future__ import annotations

import numpy as np

from seiyuu_faceid.gallery.index import Gallery
from seiyuu_faceid.gallery.scoring import (
    aggregate_reference_scores,
    classify_candidates,
    search_gallery,
)


def make_gallery() -> Gallery:
    embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    return Gallery.from_references(embeddings, ["kana", "kana", "inori", "inori"])


def test_identity_scoring_shortlists_with_prototype_then_verifies_references() -> None:
    candidates = search_gallery(make_gallery(), np.array([1, 0], dtype=np.float32), top_k=2)
    assert [candidate.identity for candidate in candidates] == ["kana", "inori"]
    assert candidates[0].score > candidates[1].score


def test_aggregation_methods() -> None:
    scores = np.array([0.2, 0.8, 0.6], dtype=np.float32)
    assert np.isclose(aggregate_reference_scores(scores, "max", 2), 0.8)
    assert np.isclose(aggregate_reference_scores(scores, "top_k_mean", 2), 0.7)


def test_candidate_search_range_is_independent_of_returned_top_k() -> None:
    candidates = search_gallery(
        make_gallery(),
        np.array([1, 0], dtype=np.float32),
        top_k=1,
        candidate_search_top_k=2,
    )
    assert len(candidates) == 1


def test_threshold_logic_has_explicit_unknown_and_uncalibrated_review() -> None:
    candidates = search_gallery(make_gallery(), np.array([1, 0], dtype=np.float32), top_k=2)
    assert classify_candidates(candidates, None, None).status == "review"
    assert classify_candidates(candidates, 0.95, 0.85).status == "recognized"
    assert classify_candidates(candidates, 1.1, 1.05).status == "unknown"
