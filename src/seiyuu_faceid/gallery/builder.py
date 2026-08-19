"""Build a Gallery from aligned image records without retaining images in SQLite."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from seiyuu_faceid.gallery.index import Gallery


class Embedder(Protocol):
    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray: ...


def build_gallery(
    records: list[dict[str, object]], aligned_root: Path, recognizer: Embedder
) -> Gallery:
    """Extract reference embeddings from accepted aligned-face metadata records."""
    import cv2

    embeddings: list[np.ndarray] = []
    identities: list[str] = []
    metadata: list[dict[str, object]] = []
    for record in records:
        if not record.get("accepted", False):
            continue
        identity = record.get("identity")
        relative_path = record.get("aligned_face_path")
        if not isinstance(identity, str) or not isinstance(relative_path, str):
            continue
        image = cv2.imread(str(aligned_root / relative_path))
        if image is None:
            continue
        embeddings.append(recognizer.embed(image))
        identities.append(identity)
        metadata.append(record)
    if not embeddings:
        raise ValueError(
            "No accepted aligned faces found. Run build_dataset.py and inspect metadata."
        )
    return Gallery.from_references(np.vstack(embeddings), identities, metadata)
