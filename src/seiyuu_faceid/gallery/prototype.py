"""Prototype construction with a future-compatible multi-prototype shape."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from seiyuu_faceid.vision.recognizer import l2_normalize


def mean_prototypes(embeddings: np.ndarray, identities: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return one normalized mean prototype per identity in deterministic order."""
    if len(embeddings) != len(identities):
        raise ValueError("embeddings and identities must have equal length.")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for embedding, identity in zip(embeddings, identities, strict=True):
        grouped[identity].append(l2_normalize(embedding))
    keys = sorted(grouped)
    prototypes = np.vstack([l2_normalize(np.mean(grouped[key], axis=0)) for key in keys])
    return prototypes.astype(np.float32), keys
