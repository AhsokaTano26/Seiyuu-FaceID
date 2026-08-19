"""Dataset construction: preserve raw paths and write auditable aligned face records."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

from seiyuu_faceid.dataset.detections import DetectionRecord
from seiyuu_faceid.dataset.metadata import write_jsonl
from seiyuu_faceid.dataset.scanner import RawImage, scan_raw_dataset
from seiyuu_faceid.vision.alignment import AlignmentError, align_face
from seiyuu_faceid.vision.detector import FaceDetection
from seiyuu_faceid.vision.quality import assess_face_quality


class Detector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]: ...


def build_aligned_dataset(
    raw_root: Path,
    aligned_root: Path,
    rejected_root: Path,
    detector: Detector | None,
    *,
    min_face_size: int,
    min_sharpness: float,
    face_selections: dict[str, list[int]] | None = None,
    detection_records: Mapping[str, DetectionRecord] | None = None,
) -> tuple[int, int]:
    """Build a gallery dataset without enrolling unreviewed faces from group photos.

    A single detected face is accepted normally. For a source image with several
    faces, ``face_selections`` must name every zero-based face index to keep;
    unselected faces are recorded as rejected rather than silently enrolled.
    """
    import cv2

    accepted_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    selections = face_selections or {}
    cached_detections = detection_records or {}
    for source in scan_raw_dataset(raw_root):
        source_image = source.relative_path.as_posix()
        cached = cached_detections.get(source_image)
        if cached is not None and cached.source_sha256 != source.sha256:
            cached = None
        if cached is not None and cached.error == "decode_failed":
            rejected_records.append(
                {
                    "source_image": source_image,
                    "identity": source.identity,
                    "reason": "decode_failed",
                }
            )
            continue
        image = cv2.imread(str(source.path))
        if image is None:
            rejected_records.append(
                {
                    "source_image": source_image,
                    "identity": source.identity,
                    "reason": "decode_failed",
                }
            )
            continue
        if cached is not None:
            detections = list(cached.detections)
        elif detector is not None:
            detections = detector.detect(image)
        else:
            raise RuntimeError(
                "No current detection cache entry for "
                f"{source_image}; scan raw images before building."
            )
        if not detections:
            rejected_records.append(
                {
                    "source_image": source_image,
                    "identity": source.identity,
                    "reason": "no_face_detected",
                }
            )
            continue
        selected_face_indexes = selections.get(source_image)
        # Compatibility for callers that still pass the former single-index mapping.
        if isinstance(selected_face_indexes, int):
            selected_face_indexes = [selected_face_indexes]
        if len(detections) > 1 and selected_face_indexes is None:
            rejected_records.extend(
                _selection_rejection_records(source, detections, "selection_required")
            )
            continue
        selected_indexes = set(selected_face_indexes or [])
        if any(index >= len(detections) for index in selected_indexes):
            rejected_records.extend(
                _selection_rejection_records(
                    source, detections, "selected_face_index_out_of_range"
                )
            )
            continue
        for face_index, detection in enumerate(detections):
            if len(detections) > 1 and face_index not in selected_indexes:
                rejected_records.append(
                    _selection_rejection_record(source, face_index, detection, "not_selected")
                )
                continue
            quality = assess_face_quality(
                image,
                detection.bbox,
                min_face_size=min_face_size,
                min_sharpness=min_sharpness,
            )
            record: dict[str, object] = {
                "source_image": str(source.relative_path),
                "source_sha256": source.sha256,
                "identity": source.identity,
                "face_index": face_index,
                "bbox": [round(float(value), 3) for value in detection.bbox],
                "detector_score": round(detection.score, 6),
                "quality_score": round(quality.score, 6),
                "quality": {"face_size": quality.face_size, "sharpness": quality.sharpness},
                "accepted": quality.accepted,
            }
            if not quality.accepted:
                record["reasons"] = list(quality.reasons)
                rejected_records.append(record)
                continue
            try:
                aligned = align_face(image, detection.landmarks)
            except AlignmentError as error:
                record["accepted"] = False
                record["reasons"] = ["alignment_failed", str(error)]
                rejected_records.append(record)
                continue
            relative_output = Path(source.identity) / f"{source.sha256[:16]}_face{face_index}.jpg"
            output_path = aligned_root / relative_output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), aligned):
                record["accepted"] = False
                record["reasons"] = ["write_failed"]
                rejected_records.append(record)
                continue
            record["aligned_face_path"] = str(relative_output)
            accepted_records.append(record)
    write_jsonl(aligned_root / "metadata.jsonl", accepted_records)
    write_jsonl(rejected_root / "metadata.jsonl", rejected_records)
    return len(accepted_records), len(rejected_records)


def _selection_rejection_records(
    source: RawImage, detections: list[FaceDetection], reason: str
) -> list[dict[str, object]]:
    return [
        _selection_rejection_record(source, face_index, detection, reason)
        for face_index, detection in enumerate(detections)
    ]


def _selection_rejection_record(
    source: RawImage, face_index: int, detection: FaceDetection, reason: str
) -> dict[str, object]:
    """Build an auditable rejection record before image-quality processing."""
    return {
        "source_image": str(source.relative_path),
        "source_sha256": source.sha256,
        "identity": source.identity,
        "face_index": face_index,
        "bbox": [round(float(value), 3) for value in detection.bbox],
        "detector_score": round(detection.score, 6),
        "accepted": False,
        "reasons": [reason],
    }
