from __future__ import annotations

import cv2
import numpy as np

from seiyuu_faceid.dataset.detections import (
    DetectionRecord,
    current_detection_records,
    load_detection_records,
    save_detection_records,
)
from seiyuu_faceid.dataset.scanner import scan_raw_dataset
from seiyuu_faceid.vision.alignment import ARC_FACE_TEMPLATE_112
from seiyuu_faceid.vision.detector import FaceDetection


def _create_raw_image(tmp_path) -> None:
    raw = tmp_path / "raw" / "kana"
    raw.mkdir(parents=True)
    image = np.full((112, 112, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(raw / "001.jpg"), image)


def _record_for_source(source) -> DetectionRecord:
    return DetectionRecord(
        source_image=source.relative_path.as_posix(),
        source_sha256=source.sha256,
        identity=source.identity,
        detections=(
            FaceDetection(
                bbox=np.array([0, 0, 111, 111], dtype=np.float32),
                score=0.99,
                landmarks=ARC_FACE_TEMPLATE_112,
            ),
        ),
    )


def test_detection_cache_round_trip_and_source_change_invalidation(tmp_path) -> None:
    _create_raw_image(tmp_path)
    sources = scan_raw_dataset(tmp_path / "raw")
    records = {sources[0].relative_path.as_posix(): _record_for_source(sources[0])}
    cache_path = tmp_path / "detections.jsonl"

    save_detection_records(cache_path, records)
    loaded = load_detection_records(cache_path)

    assert current_detection_records(sources, loaded) is not None
    assert loaded["kana/001.jpg"].detections[0].landmarks.shape == (5, 2)

    image = np.full((112, 112, 3), 80, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "raw" / "kana" / "001.jpg"), image)
    changed_sources = scan_raw_dataset(tmp_path / "raw")
    assert current_detection_records(changed_sources, loaded) is None
