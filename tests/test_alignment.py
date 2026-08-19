from __future__ import annotations

import numpy as np

from seiyuu_faceid.vision.alignment import ARC_FACE_TEMPLATE_112, align_face


def test_alignment_returns_expected_shape() -> None:
    image = np.full((128, 128, 3), 127, dtype=np.uint8)
    aligned = align_face(image, ARC_FACE_TEMPLATE_112)
    assert aligned.shape == (112, 112, 3)
    assert int(aligned[56, 56, 0]) == 127
