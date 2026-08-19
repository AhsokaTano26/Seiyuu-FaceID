"""Inexpensive quality checks used to flag, never relabel, face crops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    score: float
    face_size: float
    sharpness: float
    reasons: tuple[str, ...]


def assess_face_quality(
    face_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    min_face_size: int,
    min_sharpness: float,
) -> QualityResult:
    """Apply deterministic size and blur checks and return auditable metrics."""
    import cv2

    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    face_size = float(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    height, width = face_bgr.shape[:2]
    x1, y1 = np.maximum(np.floor(bbox[:2]).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(bbox[2:]).astype(int), [width, height])
    crop = face_bgr[y1:y2, x1:x2]
    sharpness = 0.0
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []
    if face_size < min_face_size:
        reasons.append("face_too_small")
    if sharpness < min_sharpness:
        reasons.append("too_blurry")
    size_score = min(1.0, face_size / max(min_face_size * 2, 1))
    sharpness_score = min(1.0, sharpness / max(min_sharpness * 2, 1.0))
    return QualityResult(
        accepted=not reasons,
        score=float((size_score + sharpness_score) / 2),
        face_size=face_size,
        sharpness=sharpness,
        reasons=tuple(reasons),
    )
