"""Persistent face-detection results for a raw dataset scan.

The manifest is intentionally disposable cache data.  It keeps expensive ONNX
detection separate from the later human-review and alignment stages, while the
source file SHA-256 prevents stale detections from being reused after an image
changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from seiyuu_faceid.dataset.metadata import read_jsonl, write_jsonl
from seiyuu_faceid.dataset.scanner import RawImage, scan_raw_dataset
from seiyuu_faceid.vision.detector import FaceDetection


class Detector(Protocol):
    """The small detector surface needed for a full raw scan."""

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]: ...


@dataclass(frozen=True)
class DetectionRecord:
    """Detection result for one raw source image."""

    source_image: str
    source_sha256: str
    identity: str
    detections: tuple[FaceDetection, ...]
    error: str | None = None


def scan_raw_detections(
    raw_root: Path,
    detector: Detector,
    *,
    progress: Callable[[int, int, RawImage], None] | None = None,
) -> tuple[list[RawImage], dict[str, DetectionRecord]]:
    """Detect every raw image once and return records keyed by relative path."""
    import cv2

    sources = scan_raw_dataset(raw_root)
    records: dict[str, DetectionRecord] = {}
    total = len(sources)
    for number, source in enumerate(sources, start=1):
        image = cv2.imread(str(source.path))
        source_image = source.relative_path.as_posix()
        if image is None:
            record = DetectionRecord(
                source_image=source_image,
                source_sha256=source.sha256,
                identity=source.identity,
                detections=(),
                error="decode_failed",
            )
        else:
            record = DetectionRecord(
                source_image=source_image,
                source_sha256=source.sha256,
                identity=source.identity,
                detections=tuple(detector.detect(image)),
            )
        records[source_image] = record
        if progress is not None:
            progress(number, total, source)
    return sources, records


def save_detection_records(path: Path, records: Mapping[str, DetectionRecord]) -> None:
    """Write a stable JSONL manifest that remains human-inspectable."""
    write_jsonl(
        path,
        [_serialize_record(records[key]) for key in sorted(records)],
    )


def load_detection_records(path: Path) -> dict[str, DetectionRecord]:
    """Load and validate an on-disk detection manifest."""
    records: dict[str, DetectionRecord] = {}
    for number, value in enumerate(read_jsonl(path), start=1):
        try:
            record = _parse_record(value)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid detection record on line {number} in {path}.") from error
        if record.source_image in records:
            raise ValueError(f"Duplicate source_image in detection manifest: {record.source_image}")
        records[record.source_image] = record
    return records


def current_detection_records(
    sources: list[RawImage], records: Mapping[str, DetectionRecord]
) -> dict[str, DetectionRecord] | None:
    """Return records only when they exactly match the current raw dataset."""
    source_paths = {source.relative_path.as_posix() for source in sources}
    if set(records) != source_paths:
        return None
    for source in sources:
        record = records[source.relative_path.as_posix()]
        if record.source_sha256 != source.sha256 or record.identity != source.identity:
            return None
    return dict(records)


def _serialize_record(record: DetectionRecord) -> dict[str, object]:
    return {
        "source_image": record.source_image,
        "source_sha256": record.source_sha256,
        "identity": record.identity,
        "error": record.error,
        "detections": [
            {
                "bbox": [round(float(value), 3) for value in detection.bbox],
                "score": round(float(detection.score), 6),
                "landmarks": [
                    [round(float(value), 3) for value in point] for point in detection.landmarks
                ],
            }
            for detection in record.detections
        ],
    }


def _parse_record(value: dict[str, object]) -> DetectionRecord:
    source_image = value["source_image"]
    source_sha256 = value["source_sha256"]
    identity = value["identity"]
    error = value.get("error")
    detections = value["detections"]
    if not all(
        isinstance(field, str) and field for field in (source_image, source_sha256, identity)
    ):
        raise ValueError("source fields must be non-empty strings")
    if error is not None and not isinstance(error, str):
        raise ValueError("error must be a string or null")
    if not isinstance(detections, list):
        raise ValueError("detections must be a list")
    return DetectionRecord(
        source_image=source_image,
        source_sha256=source_sha256,
        identity=identity,
        error=error,
        detections=tuple(_parse_detection(item) for item in detections),
    )


def _parse_detection(value: object) -> FaceDetection:
    if not isinstance(value, dict):
        raise ValueError("detection must be an object")
    bbox = np.asarray(value["bbox"], dtype=np.float32)
    landmarks = np.asarray(value["landmarks"], dtype=np.float32)
    score = value["score"]
    if bbox.shape != (4,) or landmarks.shape != (5, 2):
        raise ValueError("unexpected detection geometry")
    if not np.isfinite(bbox).all() or not np.isfinite(landmarks).all():
        raise ValueError("detection geometry must be finite")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score):
        raise ValueError("detection score must be finite")
    return FaceDetection(bbox=bbox, score=float(score), landmarks=landmarks)
