"""ArcFace-compatible five-point face alignment."""

from __future__ import annotations

import numpy as np

ARC_FACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class AlignmentError(ValueError):
    """Raised when five-point alignment cannot be estimated."""


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """Warp an image to the ArcFace five-point template in BGR color order."""
    import cv2

    source = np.asarray(landmarks, dtype=np.float32)
    if source.shape != (5, 2):
        raise AlignmentError(f"Expected landmarks shaped (5, 2), got {source.shape}.")
    target = ARC_FACE_TEMPLATE_112 * (output_size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if matrix is None:
        raise AlignmentError("Could not estimate a five-point similarity transform.")
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
