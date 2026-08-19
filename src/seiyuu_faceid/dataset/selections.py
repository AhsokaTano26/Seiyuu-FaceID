"""Human-reviewed choices for images that contain more than one face."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from seiyuu_faceid.dataset.metadata import read_jsonl, write_jsonl


def load_face_selections(
    path: str | Path | PurePosixPath, *, allow_incomplete: bool = False
) -> dict[str, list[int]]:
    """Load ``source_image`` → zero-based ``face_indexes`` selections from JSONL.

    The former single ``face_index`` field is accepted for existing review files
    and is normalized to a one-item list.
    """
    selections: dict[str, list[int]] = {}
    for number, record in enumerate(read_jsonl(path), start=1):
        source_image = record.get("source_image")
        if not isinstance(source_image, str) or not _is_safe_relative_path(source_image):
            raise ValueError(f"Selection line {number} needs a relative source_image path.")
        if source_image in selections:
            raise ValueError(f"Selection file has duplicate source_image: {source_image}")
        face_indexes = record.get("face_indexes")
        if face_indexes is None and "face_index" in record:
            face_indexes = [record["face_index"]]
        if face_indexes is None and allow_incomplete:
            continue
        if not isinstance(face_indexes, list):
            raise ValueError(f"Selection line {number} needs a face_indexes list.")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in face_indexes
        ):
            raise ValueError(f"Selection line {number} needs non-negative integer face_indexes.")
        if len(set(face_indexes)) != len(face_indexes):
            raise ValueError(f"Selection line {number} has duplicate face_indexes.")
        selections[source_image] = sorted(face_indexes)
    return selections


def save_face_selections(path: Path, selections: dict[str, list[int]]) -> None:
    """Persist reviewed source-image choices in a stable, human-editable order."""
    write_jsonl(
        path,
        [
            {"source_image": source_image, "face_indexes": sorted(face_indexes)}
            for source_image, face_indexes in sorted(selections.items())
        ],
    )


def parse_review_face_indexes(value: str, detected_face_count: int) -> list[int]:
    """Parse one-based UI face IDs, or ``A``/``all`` for every detected face.

    An empty entry deliberately represents "none of these faces belongs to this
    identity". Stored selections remain zero-based for metadata consistency.
    """
    if detected_face_count < 1:
        raise ValueError("At least one detected face is required.")
    normalized = value.strip().casefold()
    if not normalized:
        return []
    if normalized in {"a", "all", "全部", "全选"}:
        return list(range(detected_face_count))
    tokens = [token for token in re.split(r"[\s,，]+", normalized) if token]
    if not tokens or any(not token.isdigit() for token in tokens):
        raise ValueError("Use face IDs such as 1,3, or enter A for all faces.")
    indexes = [int(token) - 1 for token in tokens]
    if any(index < 0 or index >= detected_face_count for index in indexes):
        raise ValueError(f"Face IDs must be between 1 and {detected_face_count}.")
    if len(set(indexes)) != len(indexes):
        raise ValueError("Face IDs must not be repeated.")
    return sorted(indexes)


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and path.name != "."
