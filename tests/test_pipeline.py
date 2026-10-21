from __future__ import annotations

import numpy as np

from seiyuu_faceid.runtime.pipeline import recognize_detection, serialize_result
from seiyuu_faceid.runtime.results import Candidate
from seiyuu_faceid.vision.detector import FaceDetection


class FailingClassifier:
    def classify(self, aligned, *, top_k):
        raise AssertionError("A rejected face must not be classified")


def detection():
    return FaceDetection(
        bbox=np.array([10, 10, 30, 30], dtype=np.float32),
        score=0.9,
        landmarks=np.array([[14, 16], [26, 16], [20, 21], [16, 26], [24, 26]], dtype=np.float32),
    )


def test_rejected_face_keeps_the_provided_detection_without_classifying():
    item = detection()
    result = recognize_detection(
        np.zeros((100, 100, 3), dtype=np.uint8),
        detection=item,
        classifier=FailingClassifier(),
        min_face_size=30,
        min_sharpness=0,
        top_k=1,
    )
    assert result.detection is item
    assert result.decision.status == "review"
    assert result.decision.identity is None
    assert not result.decision.candidates


def test_accepted_face_returns_supervised_classification_and_probability():
    class Classifier:
        def classify(self, aligned, *, top_k):
            assert aligned.shape == (112, 112, 3)
            return [Candidate("alice", 0.95), Candidate("bob", 0.05)][:top_k]

    result = recognize_detection(
        np.zeros((100, 100, 3), dtype=np.uint8),
        detection=detection(),
        classifier=Classifier(),
        min_face_size=10,
        min_sharpness=0,
        top_k=2,
    )
    assert result.decision.status == "recognized"
    assert result.decision.identity == "alice"
    assert serialize_result(result)["candidates"] == [
        {"identity": "alice", "score": 0.95},
        {"identity": "bob", "score": 0.05},
    ]
