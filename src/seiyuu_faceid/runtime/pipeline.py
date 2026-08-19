"""End-to-end image recognition pipeline with one result per detected face."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seiyuu_faceid.gallery.index import Gallery
from seiyuu_faceid.gallery.scoring import Decision, classify_candidates, search_gallery
from seiyuu_faceid.vision.alignment import align_face
from seiyuu_faceid.vision.detector import FaceDetection
from seiyuu_faceid.vision.quality import QualityResult, assess_face_quality


@dataclass(frozen=True)
class FaceRecognition:
    detection: FaceDetection
    quality: QualityResult
    decision: Decision


def recognize_faces(
    image_bgr: np.ndarray,
    *,
    detector: object,
    recognizer: object,
    gallery: Gallery,
    min_face_size: int,
    min_sharpness: float,
    top_k: int,
    candidate_search_top_k: int,
    aggregation: str,
    verification_top_k: int,
    high_threshold: float | None,
    low_threshold: float | None,
) -> list[FaceRecognition]:
    """Detect, filter, align, embed and compare every face in a BGR image."""
    detections = detector.detect(image_bgr)  # type: ignore[attr-defined]
    return [
        recognize_detection(
            image_bgr,
            detection=detection,
            recognizer=recognizer,
            gallery=gallery,
            min_face_size=min_face_size,
            min_sharpness=min_sharpness,
            top_k=top_k,
            candidate_search_top_k=candidate_search_top_k,
            aggregation=aggregation,
            verification_top_k=verification_top_k,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        for detection in detections
    ]


def recognize_detection(
    image_bgr: np.ndarray,
    *,
    detection: FaceDetection,
    recognizer: object,
    gallery: Gallery,
    min_face_size: int,
    min_sharpness: float,
    top_k: int,
    candidate_search_top_k: int,
    aggregation: str,
    verification_top_k: int,
    high_threshold: float | None,
    low_threshold: float | None,
) -> FaceRecognition:
    """Recognize one already-detected face without running detection again."""
    quality = assess_face_quality(
        image_bgr,
        detection.bbox,
        min_face_size=min_face_size,
        min_sharpness=min_sharpness,
    )
    if not quality.accepted:
        return FaceRecognition(detection, quality, Decision("review", None, ()))
    aligned = align_face(image_bgr, detection.landmarks)
    embedding = recognizer.embed(aligned)  # type: ignore[attr-defined]
    candidates = search_gallery(
        gallery,
        embedding,
        top_k=top_k,
        candidate_search_top_k=candidate_search_top_k,
        aggregation=aggregation,
        verification_top_k=verification_top_k,
    )
    return FaceRecognition(
        detection,
        quality,
        classify_candidates(candidates, high_threshold, low_threshold),
    )


def serialize_result(result: FaceRecognition) -> dict[str, object]:
    """Make result values safe for JSONL batch output."""
    return {
        "bbox": [round(float(value), 2) for value in result.detection.bbox],
        "detector_score": result.detection.score,
        "quality_score": result.quality.score,
        "quality_reasons": list(result.quality.reasons),
        "status": result.decision.status,
        "identity": result.decision.identity,
        "candidates": [
            {
                "identity": candidate.identity,
                "score": candidate.score,
                "prototype_score": candidate.prototype_score,
                "reference_score": candidate.reference_score,
            }
            for candidate in result.decision.candidates
        ],
    }
