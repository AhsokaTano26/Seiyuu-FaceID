"""Portable NumPy/JSON gallery serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seiyuu_faceid.gallery.prototype import mean_prototypes
from seiyuu_faceid.vision.recognizer import l2_normalize


@dataclass(frozen=True)
class Gallery:
    """Normalized reference vectors plus one or more normalized identity prototypes."""

    references: np.ndarray
    reference_identities: tuple[str, ...]
    reference_metadata: tuple[dict[str, object], ...]
    prototypes: np.ndarray
    prototype_identities: tuple[str, ...]

    @property
    def embedding_dim(self) -> int:
        return int(self.references.shape[1])

    @classmethod
    def from_references(
        cls,
        embeddings: np.ndarray,
        identities: list[str],
        metadata: list[dict[str, object]] | None = None,
    ) -> Gallery:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("Gallery needs at least one 2-D reference embedding.")
        if len(vectors) != len(identities):
            raise ValueError("embeddings and identities must have equal length.")
        normalized = np.vstack([l2_normalize(vector) for vector in vectors])
        prototypes, prototype_identities = mean_prototypes(normalized, identities)
        records = metadata or [{} for _ in identities]
        if len(records) != len(identities):
            raise ValueError("metadata and identities must have equal length.")
        return cls(
            references=normalized.astype(np.float32),
            reference_identities=tuple(identities),
            reference_metadata=tuple(records),
            prototypes=prototypes,
            prototype_identities=tuple(prototype_identities),
        )

    def save(self, vectors_path: Path, metadata_path: Path) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(vectors_path, references=self.references, prototypes=self.prototypes)
        payload = {
            "schema_version": 1,
            "embedding_dim": self.embedding_dim,
            "reference_identities": list(self.reference_identities),
            "reference_metadata": list(self.reference_metadata),
            "prototype_identities": list(self.prototype_identities),
        }
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, vectors_path: Path, metadata_path: Path) -> Gallery:
        if not vectors_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "Gallery is incomplete. "
                f"Expected {vectors_path} and {metadata_path}; run build_gallery.py."
            )
        with np.load(vectors_path) as archive:
            references = np.asarray(archive["references"], dtype=np.float32)
            prototypes = np.asarray(archive["prototypes"], dtype=np.float32)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        gallery = cls(
            references=references,
            reference_identities=tuple(payload["reference_identities"]),
            reference_metadata=tuple(payload["reference_metadata"]),
            prototypes=prototypes,
            prototype_identities=tuple(payload["prototype_identities"]),
        )
        if gallery.references.ndim != 2 or gallery.prototypes.ndim != 2:
            raise ValueError("Gallery vectors must be two-dimensional.")
        if len(gallery.references) != len(gallery.reference_identities):
            raise ValueError("Gallery reference vectors and identities do not match.")
        if len(gallery.prototypes) != len(gallery.prototype_identities):
            raise ValueError("Gallery prototype vectors and identities do not match.")
        if not np.allclose(np.linalg.norm(gallery.references, axis=1), 1.0, atol=1e-4):
            raise ValueError("Gallery references are not L2 normalized.")
        return gallery
