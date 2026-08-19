from __future__ import annotations

import numpy as np

from seiyuu_faceid.vision.detector import SCRFDDetector


def test_scrfd_decode_uses_scores_boxes_and_five_landmarks() -> None:
    # Three feature levels for a synthetic 32×32 detector input, one anchor/cell.
    scores = [np.zeros((1, count, 1), dtype=np.float32) for count in (16, 4, 1)]
    scores[0][0, 0, 0] = 0.9
    boxes = [np.ones((1, count, 4), dtype=np.float32) for count in (16, 4, 1)]
    keypoints = [np.zeros((1, count, 10), dtype=np.float32) for count in (16, 4, 1)]
    detector = SCRFDDetector.__new__(SCRFDDetector)
    detector.score_threshold = 0.5
    decoded_boxes, decoded_scores, decoded_points = detector._decode(
        [*scores, *boxes, *keypoints], 32, 32
    )
    assert decoded_boxes.shape == (1, 4)
    assert np.allclose(decoded_boxes[0], [-8, -8, 8, 8])
    assert np.allclose(decoded_scores, [0.9])
    assert decoded_points.shape == (1, 5, 2)
