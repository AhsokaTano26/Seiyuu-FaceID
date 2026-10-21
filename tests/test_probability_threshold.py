import numpy as np
import pytest

from seiyuu_faceid.runtime.pipeline import recognize_detection, serialize_result
from seiyuu_faceid.runtime.results import Candidate, decide_candidates
from seiyuu_faceid.vision.detector import FaceDetection


@pytest.mark.parametrize(
    "score, status, identity",
    [
        (0.95, "recognized", "alice"),
        (0.949999, "review", None),
    ],
)
def test_probability_boundary_controls_acceptance_without_losing_candidates(
    score, status, identity
):
    candidates = [Candidate("alice", score), Candidate("bob", 1 - score)]
    decision = decide_candidates(candidates, probability_threshold=0.95)
    assert decision.status == status
    assert decision.identity == identity
    assert decision.candidates == tuple(candidates)
    assert decision.reason == ("low_probability" if status == "review" else None)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_probability_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="probability_threshold"):
        decide_candidates([Candidate("alice", 0.9)], probability_threshold=value)


def test_pipeline_applies_configured_threshold_and_exports_review_reason():
    class Classifier:
        def classify(self, aligned, *, top_k):
            return [Candidate("alice", 0.96), Candidate("bob", 0.04)][:top_k]

    detection = FaceDetection(
        bbox=np.array([10, 10, 30, 30], dtype=np.float32),
        score=0.9,
        landmarks=np.array([[14, 16], [26, 16], [20, 21], [16, 26], [24, 26]], dtype=np.float32),
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    common = dict(
        detection=detection, classifier=Classifier(), min_face_size=10, min_sharpness=0, top_k=2
    )
    assert (
        recognize_detection(image, probability_threshold=0.95, **common).decision.identity
        == "alice"
    )
    review = recognize_detection(image, probability_threshold=0.97, **common)
    assert review.decision.status == "review" and review.decision.identity is None
    exported = serialize_result(review)
    assert exported["probability_threshold"] == 0.97
    assert exported["decision_reason"] == "low_probability"
    assert exported["candidates"][0]["identity"] == "alice"
