from __future__ import annotations

import numpy as np

from seiyuu_faceid.vision.detector import FaceDetection
from seiyuu_faceid.web.components import FACE_BOX_COLORS, detection_preview
from seiyuu_faceid.web.state import (
    consume_dialog_request,
    parse_group_face_input,
    result_rows_from_record,
)


def test_group_review_shortcuts_parse_to_selected_faces() -> None:
    assert parse_group_face_input("A", 3) == [0, 1, 2]
    assert parse_group_face_input("1, 3", 3) == [0, 2]


def test_recognition_preview_draws_a_recognized_face_in_green() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = FaceDetection(
        bbox=np.array([20, 20, 80, 80], dtype=np.float32),
        score=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
    )
    preview = detection_preview(
        image,
        [detection],
        set(),
        recognition_statuses={0: "recognized"},
        max_long_edge=100,
    )
    assert np.array_equal(preview[20, 20], FACE_BOX_COLORS["recognized"])


def test_result_rows_keep_identity_status_and_candidates_together() -> None:
    rows = result_rows_from_record(
        {
            "faces": [
                {
                    "identity": "kana",
                    "status": "recognized",
                    "detector_score": 0.9,
                    "quality_score": 1.0,
                    "quality_reasons": [],
                    "candidates": [{"identity": "kana", "score": 0.8}],
                }
            ]
        }
    )
    assert rows == [
        {
            "face": 1,
            "result": "kana",
            "status": "recognized",
            "detector_score": 0.9,
            "quality_score": 1.0,
            "quality_flags": "-",
            "top_candidates": "kana (0.800)",
        }
    ]


def test_dialog_request_is_consumed_after_one_open() -> None:
    state: dict[str, object] = {"raw-detail-image": "/tmp/image.jpg"}
    assert consume_dialog_request(state, "raw-detail-image") == "/tmp/image.jpg"
    assert consume_dialog_request(state, "raw-detail-image") is None


def test_probability_review_is_not_displayed_as_an_unknown_identity():
    rows = result_rows_from_record(
        {
            "faces": [
                {
                    "status": "review",
                    "identity": None,
                    "candidates": [{"identity": "alice", "score": 0.94}],
                }
            ]
        }
    )
    assert rows[0]["result"] == "待复核"
    assert rows[0]["status"] == "review"
    assert "alice" in rows[0]["top_candidates"]
