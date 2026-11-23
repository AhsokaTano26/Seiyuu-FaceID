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


def test_sparse_decode_uses_cell_and_anchor_order_and_request_threshold():
    detector = SCRFDDetector.__new__(SCRFDDetector)
    detector.score_threshold = 0.8
    counts = (32, 8, 2)  # Two anchors per cell on all three feature levels.
    scores = [np.zeros(n, np.float32) for n in counts]
    scores[0][13] = 0.5  # Cell 6 is (2, 1), center (16, 8) at stride 8.
    scores[1][5] = 0.875  # Cell 2 is (0, 1), center (0, 16) at stride 16.
    boxes = [np.ones((n, 4), np.float32) for n in counts]
    offsets = [np.ones((n, 5, 2), np.float32) for n in counts]
    outputs = scores + boxes + offsets
    decoded, confidence, points = detector._decode(outputs, 32, 32, score_threshold=0.5)
    np.testing.assert_array_equal(decoded, [[8, 0, 24, 16], [-16, 0, 16, 32]])
    np.testing.assert_array_equal(confidence, [0.5, 0.875])
    np.testing.assert_array_equal(points[:, 0], [[24, 16], [16, 32]])
    for override in (None, 0.5000000001):
        _, confidence, _ = detector._decode(outputs, 32, 32, score_threshold=override)
        np.testing.assert_array_equal(confidence, [0.875])
