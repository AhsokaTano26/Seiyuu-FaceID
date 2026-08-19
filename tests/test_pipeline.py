from __future__ import annotations

import numpy as np

from seiyuu_faceid.gallery.index import Gallery
from seiyuu_faceid.runtime.pipeline import recognize_detection
from seiyuu_faceid.vision.detector import FaceDetection


class FailingRecognizer:
    def embed(self, aligned: np.ndarray) -> np.ndarray:
        raise AssertionError("A rejected face must not be embedded")


def test_recognize_detection_uses_the_provided_detection() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = FaceDetection(
        bbox=np.array([10, 10, 30, 30], dtype=np.float32),
        score=0.9,
        landmarks=np.array([[14, 16], [26, 16], [20, 21], [16, 26], [24, 26]], dtype=np.float32),
    )
    gallery = Gallery.from_references(np.array([[1, 0]], dtype=np.float32), ["test"])

    result = recognize_detection(
        image,
        detection=detection,
        recognizer=FailingRecognizer(),
        gallery=gallery,
        min_face_size=30,
        min_sharpness=0.0,
        top_k=1,
        candidate_search_top_k=1,
        aggregation="prototype",
        verification_top_k=1,
        high_threshold=None,
        low_threshold=None,
    )

    assert result.detection is detection
    assert result.decision.status == "review"
