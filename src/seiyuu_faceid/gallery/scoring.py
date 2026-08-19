"""Identity-level scoring and explicit unknown/review decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seiyuu_faceid.gallery.index import Gallery
from seiyuu_faceid.vision.recognizer import l2_normalize


@dataclass(frozen=True)
class Candidate:
    identity: str
    score: float
    prototype_score: float
    reference_score: float


@dataclass(frozen=True)
class Decision:
    status: str
    identity: str | None
    candidates: tuple[Candidate, ...]


def aggregate_reference_scores(scores: np.ndarray, method: str, top_k: int) -> float:
    """Aggregate a single identity's reference similarities."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not len(values):
        raise ValueError("Cannot aggregate an empty set of reference scores.")
    if method == "max":
        return float(values.max())
    if method == "top_k_mean":
        selected = np.partition(values, -min(top_k, len(values)))[-min(top_k, len(values)) :]
        return float(selected.mean())
    if method == "prototype_score":
        return float(values.max())
    raise ValueError(f"Unknown aggregation method: {method}")


def search_gallery(
    gallery: Gallery,
    query_embedding: np.ndarray,
    *,
    top_k: int,
    aggregation: str = "top_k_mean",
    verification_top_k: int = 3,
    candidate_search_top_k: int | None = None,
) -> list[Candidate]:
    """Use prototypes to shortlist identities, then score their references.

    ``top_k`` controls only returned candidates. The internal shortlist can be
    larger, so the final result is not constrained by the UI's Top-K value.
    """
    if top_k < 1 or verification_top_k < 1:
        raise ValueError("top_k and verification_top_k must be positive.")
    if candidate_search_top_k is not None and candidate_search_top_k < top_k:
        raise ValueError("candidate_search_top_k must be at least top_k.")
    query = l2_normalize(query_embedding)
    if query.size != gallery.embedding_dim:
        raise ValueError(
            f"Query has {query.size} dimensions; gallery expects {gallery.embedding_dim}."
        )
    prototype_scores = gallery.prototypes @ query
    # Multiple future prototypes per identity are collapsed conservatively by max.
    prototype_by_identity: dict[str, float] = {}
    for identity, score in zip(gallery.prototype_identities, prototype_scores, strict=True):
        prototype_by_identity[identity] = max(
            prototype_by_identity.get(identity, -np.inf), float(score)
        )
    shortlist_size = candidate_search_top_k or len(prototype_by_identity)
    shortlisted = sorted(prototype_by_identity, key=prototype_by_identity.get, reverse=True)[
        :shortlist_size
    ]
    reference_scores = gallery.references @ query
    candidates: list[Candidate] = []
    for identity in shortlisted:
        identity_scores = reference_scores[
            np.asarray([item == identity for item in gallery.reference_identities], dtype=bool)
        ]
        reference_score = aggregate_reference_scores(
            identity_scores, aggregation, verification_top_k
        )
        prototype_score = prototype_by_identity[identity]
        final_score = prototype_score if aggregation == "prototype_score" else reference_score
        candidates.append(Candidate(identity, final_score, prototype_score, reference_score))
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:top_k]


def classify_candidates(
    candidates: list[Candidate], high_threshold: float | None, low_threshold: float | None
) -> Decision:
    """Return recognized/review/unknown without silently accepting uncalibrated scores."""
    if high_threshold is not None and low_threshold is not None and high_threshold < low_threshold:
        raise ValueError("high_threshold must be greater than or equal to low_threshold.")
    if not candidates:
        return Decision("unknown", None, ())
    best = candidates[0]
    if high_threshold is None or low_threshold is None:
        return Decision("review", best.identity, tuple(candidates))
    if best.score >= high_threshold:
        return Decision("recognized", best.identity, tuple(candidates))
    if best.score >= low_threshold:
        return Decision("review", best.identity, tuple(candidates))
    return Decision("unknown", None, tuple(candidates))
