"""Detect, check quality, align and classify each face independently."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seiyuu_faceid.runtime.results import Decision, decide_candidates
from seiyuu_faceid.vision.alignment import align_face
from seiyuu_faceid.vision.detector import FaceDetection
from seiyuu_faceid.vision.quality import QualityResult, assess_face_quality


@dataclass(frozen=True)
class FaceRecognition:
    detection: FaceDetection
    quality: QualityResult
    decision: Decision
    probability_threshold: float


def recognize_faces(
    image_bgr: np.ndarray,
    *,
    detector: object,
    classifier: object,
    min_face_size: int,
    min_sharpness: float,
    top_k: int,
    probability_threshold: float = 0.95,
) -> list[FaceRecognition]:
    return [
        recognize_detection(
            image_bgr,
            detection=detection,
            classifier=classifier,
            min_face_size=min_face_size,
            min_sharpness=min_sharpness,
            top_k=top_k,
            probability_threshold=probability_threshold,
        )
        for detection in detector.detect(image_bgr)  # type: ignore[attr-defined]
    ]


def recognize_detection(
    image_bgr: np.ndarray,
    *,
    detection: FaceDetection,
    classifier: object,
    min_face_size: int,
    min_sharpness: float,
    top_k: int,
    probability_threshold: float = 0.95,
) -> FaceRecognition:
    quality = assess_face_quality(
        image_bgr,
        detection.bbox,
        min_face_size=min_face_size,
        min_sharpness=min_sharpness,
    )
    if not quality.accepted:
        return FaceRecognition(
            detection,
            quality,
            Decision("review", None, (), "quality_rejected"),
            probability_threshold,
        )
    aligned = align_face(image_bgr, detection.landmarks)
    candidates = classifier.classify(aligned, top_k=top_k)  # type: ignore[attr-defined]
    return FaceRecognition(
        detection,
        quality,
        decide_candidates(candidates, probability_threshold=probability_threshold),
        probability_threshold,
    )


def serialize_result(result: FaceRecognition) -> dict[str, object]:
    """Make classification values safe for JSONL output; scores are probabilities."""
    return {
        "bbox": [round(float(value), 2) for value in result.detection.bbox],
        "detector_score": result.detection.score,
        "quality_score": result.quality.score,
        "quality_reasons": list(result.quality.reasons),
        "status": result.decision.status,
        "identity": result.decision.identity,
        "probability_threshold": result.probability_threshold,
        "decision_reason": result.decision.reason,
        "score_type": "class_probability",
        "candidates": [
            {"identity": candidate.identity, "score": candidate.score}
            for candidate in result.decision.candidates
        ],
    }
