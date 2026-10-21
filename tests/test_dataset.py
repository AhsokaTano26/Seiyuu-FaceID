from __future__ import annotations

import json

import cv2
import numpy as np

from seiyuu_faceid.dataset.cleaner import build_aligned_dataset
from seiyuu_faceid.vision.alignment import ARC_FACE_TEMPLATE_112
from seiyuu_faceid.vision.detector import FaceDetection


class DummyDetector:
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        landmarks = ARC_FACE_TEMPLATE_112 * (image_bgr.shape[0] / 112)
        return [
            FaceDetection(
                bbox=np.array([10, 10, 118, 118], dtype=np.float32),
                score=0.99,
                landmarks=landmarks,
            )
        ]


class MultiFaceDetector:
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        landmarks = ARC_FACE_TEMPLATE_112 * (image_bgr.shape[0] / 112)
        return [
            FaceDetection(
                bbox=np.array([10, 10, 118, 118], dtype=np.float32),
                score=0.99,
                landmarks=landmarks,
            ),
            FaceDetection(
                bbox=np.array([0, 0, 108, 108], dtype=np.float32),
                score=0.98,
                landmarks=landmarks,
            ),
        ]


def _write_source_image(tmp_path) -> None:
    raw = tmp_path / "raw" / "kana"
    raw.mkdir(parents=True)
    image = np.indices((128, 128)).sum(axis=0) % 2 * 255
    assert cv2.imwrite(
        str(raw / "001.jpg"), np.repeat(image[..., None], 3, axis=2).astype(np.uint8)
    )


def test_dataset_builder_preserves_raw_and_writes_auditable_metadata(tmp_path) -> None:
    raw = tmp_path / "raw" / "kana"
    raw.mkdir(parents=True)
    source = raw / "001.jpg"
    # A checkerboard gives the deterministic blur check a high sharpness score.
    image = np.indices((128, 128)).sum(axis=0) % 2 * 255
    assert cv2.imwrite(str(source), np.repeat(image[..., None], 3, axis=2).astype(np.uint8))
    before = source.read_bytes()
    accepted, rejected = build_aligned_dataset(
        tmp_path / "raw",
        tmp_path / "aligned",
        tmp_path / "rejected",
        DummyDetector(),
        min_face_size=48,
        min_sharpness=20,
        face_selections={"kana/001.jpg": [0]},
    )
    assert (accepted, rejected) == (1, 0)
    assert source.read_bytes() == before
    record = json.loads((tmp_path / "aligned" / "metadata.jsonl").read_text().strip())
    assert record["identity"] == "kana"
    assert record["face_index"] == 0
    assert (tmp_path / "aligned" / record["aligned_face_path"]).is_file()


def test_single_face_source_requires_an_explicit_selection(tmp_path) -> None:
    _write_source_image(tmp_path)
    accepted, rejected = build_aligned_dataset(
        tmp_path / "raw",
        tmp_path / "aligned",
        tmp_path / "rejected",
        DummyDetector(),
        min_face_size=48,
        min_sharpness=20,
    )

    assert (accepted, rejected) == (0, 1)
    record = json.loads((tmp_path / "rejected" / "metadata.jsonl").read_text().strip())
    assert record["reasons"] == ["selection_required"]


def test_multi_face_source_requires_an_explicit_selection(tmp_path) -> None:
    _write_source_image(tmp_path)
    accepted, rejected = build_aligned_dataset(
        tmp_path / "raw",
        tmp_path / "aligned",
        tmp_path / "rejected",
        MultiFaceDetector(),
        min_face_size=48,
        min_sharpness=20,
    )

    assert (accepted, rejected) == (0, 2)
    records = [
        json.loads(line)
        for line in (tmp_path / "rejected" / "metadata.jsonl").read_text().splitlines()
    ]
    assert {record["face_index"] for record in records} == {0, 1}
    assert all(record["reasons"] == ["selection_required"] for record in records)


def test_multi_face_source_enrolls_every_selected_face(tmp_path) -> None:
    _write_source_image(tmp_path)
    accepted, rejected = build_aligned_dataset(
        tmp_path / "raw",
        tmp_path / "aligned",
        tmp_path / "rejected",
        MultiFaceDetector(),
        min_face_size=48,
        min_sharpness=20,
        face_selections={"kana/001.jpg": [0, 1]},
    )

    assert (accepted, rejected) == (2, 0)
    accepted_records = [
        json.loads(line)
        for line in (tmp_path / "aligned" / "metadata.jsonl").read_text().splitlines()
    ]
    assert {record["face_index"] for record in accepted_records} == {0, 1}


def test_multi_face_source_rejects_unselected_faces(tmp_path) -> None:
    _write_source_image(tmp_path)
    accepted, rejected = build_aligned_dataset(
        tmp_path / "raw",
        tmp_path / "aligned",
        tmp_path / "rejected",
        MultiFaceDetector(),
        min_face_size=48,
        min_sharpness=20,
        face_selections={"kana/001.jpg": [1]},
    )

    assert (accepted, rejected) == (1, 1)
    rejected_record = json.loads((tmp_path / "rejected" / "metadata.jsonl").read_text().strip())
    assert rejected_record["face_index"] == 0
    assert rejected_record["reasons"] == ["not_selected"]
